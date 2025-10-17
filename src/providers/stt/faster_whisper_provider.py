"""
Faster-Whisper STT Provider

High-performance local speech-to-text using faster-whisper.
Optimized implementation of OpenAI Whisper with CTranslate2.
"""

import numpy as np
import time
from typing import AsyncIterator, Optional, Dict, Any
from pathlib import Path

from ..base import (
    STTProvider,
    AudioChunk,
    TranscriptionResult,
    ProviderStatus,
)


class FasterWhisperProvider(STTProvider):
    """
    Faster-Whisper STT provider using CTranslate2 backend

    Features:
    - Faster inference than original Whisper
    - Lower memory usage
    - Supports streaming transcription
    - Multi-language support with auto-detection
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Faster-Whisper provider

        Args:
            config: Configuration dictionary with:
                - model_size: Model size (tiny, base, small, medium, large, large-v2, large-v3)
                - device: Device to use (cpu, cuda, auto)
                - compute_type: Computation type (int8, float16, float32)
                - num_workers: Number of workers for parallel processing
                - language: Optional language hint
                - beam_size: Beam search size (default: 5)
                - best_of: Number of candidates to consider (default: 5)
                - temperature: Sampling temperature (default: 0.0)
                - vad_filter: Enable VAD filtering (default: False)
        """
        super().__init__(config)

        self.model_size = config.get("model_size", "medium")
        self.device = config.get("device", "auto")
        self.compute_type = config.get("compute_type", "float16")
        self.num_workers = config.get("num_workers", 2)
        self.language = config.get("language")

        # Transcription parameters
        self.beam_size = config.get("beam_size", 5)
        self.best_of = config.get("best_of", 5)
        self.temperature = config.get("temperature", 0.0)
        self.vad_filter = config.get("vad_filter", False)
        self.vad_threshold = config.get("vad_threshold", 0.5)

        # Model instance (initialized in initialize())
        self.model = None

        # Supported languages
        self._supported_languages = [
            "en", "zh", "ja", "ko", "es", "fr", "de", "ru", "pt", "it",
            "ar", "hi", "th", "vi", "id", "tr", "pl", "nl", "sv", "fi",
        ]

    async def initialize(self) -> None:
        """Initialize Faster-Whisper model"""
        if self.status == ProviderStatus.READY:
            return

        self.status = ProviderStatus.INITIALIZING

        try:
            from faster_whisper import WhisperModel

            # Auto-detect device
            if self.device == "auto":
                try:
                    import torch
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    self.device = "cpu"

            # Adjust compute type based on device
            if self.device == "cpu":
                # CPU only supports int8 and float32
                if self.compute_type not in ["int8", "float32"]:
                    self.compute_type = "int8"
            else:
                # GPU supports int8, float16, float32
                if self.compute_type not in ["int8", "float16", "float32"]:
                    self.compute_type = "float16"

            # Load model
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                num_workers=self.num_workers,
            )

            self.status = ProviderStatus.READY

        except ImportError as e:
            self.status = ProviderStatus.ERROR
            raise RuntimeError(
                "faster-whisper not installed. Install with: pip install faster-whisper"
            ) from e
        except Exception as e:
            self.status = ProviderStatus.ERROR
            raise RuntimeError(f"Failed to initialize Faster-Whisper: {e}") from e

    async def cleanup(self) -> None:
        """Clean up Faster-Whisper model"""
        if self.model:
            del self.model
            self.model = None

        self.status = ProviderStatus.STOPPED

    async def transcribe(
        self,
        audio: AudioChunk,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe single audio chunk

        Args:
            audio: Audio chunk to transcribe
            language: Optional language hint (e.g., "zh", "en")

        Returns:
            Transcription result with text and metadata
        """
        if self.status != ProviderStatus.READY:
            raise RuntimeError("Provider not initialized")

        start_time = time.time()

        try:
            # Convert bytes to numpy array
            audio_array = self._bytes_to_array(audio)

            # Use provided language or configured language
            lang = language or self.language

            # Transcribe
            segments, info = self.model.transcribe(
                audio_array,
                language=lang,
                beam_size=self.beam_size,
                best_of=self.best_of,
                temperature=self.temperature,
                vad_filter=self.vad_filter,
                vad_parameters={
                    "threshold": self.vad_threshold,
                } if self.vad_filter else None,
            )

            # Collect all segments
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text)

            full_text = " ".join(text_parts).strip()
            detected_language = info.language

            # Calculate duration and confidence
            duration_ms = (time.time() - start_time) * 1000
            self._record_request(duration_ms, error=False)

            # Create result
            result = TranscriptionResult(
                text=full_text,
                language=detected_language,
                confidence=info.language_probability,
                timestamp=audio.timestamp,
                start_time=audio.timestamp,
                end_time=audio.timestamp + audio.duration_ms / 1000,
                is_final=True,
            )

            return result

        except Exception as e:
            self._record_request(0, error=True)
            raise RuntimeError(f"Transcription failed: {e}") from e

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk],
        language: Optional[str] = None,
    ) -> AsyncIterator[TranscriptionResult]:
        """
        Transcribe audio stream in real-time

        Args:
            audio_stream: Async iterator of audio chunks
            language: Optional language hint

        Yields:
            Transcription results as they become available
        """
        if self.status != ProviderStatus.READY:
            raise RuntimeError("Provider not initialized")

        # Buffer for accumulating audio
        audio_buffer = []
        buffer_duration_ms = 0
        max_buffer_duration_ms = 30000  # 30 seconds max

        async for audio_chunk in audio_stream:
            audio_buffer.append(audio_chunk)
            buffer_duration_ms += audio_chunk.duration_ms

            # Process when buffer reaches threshold or stream ends
            if buffer_duration_ms >= max_buffer_duration_ms:
                # Merge audio chunks
                merged_audio = self._merge_audio_chunks(audio_buffer)

                # Transcribe
                result = await self.transcribe(merged_audio, language=language)

                yield result

                # Clear buffer
                audio_buffer = []
                buffer_duration_ms = 0

        # Process remaining audio
        if audio_buffer:
            merged_audio = self._merge_audio_chunks(audio_buffer)
            result = await self.transcribe(merged_audio, language=language)
            yield result

    def _bytes_to_array(self, audio: AudioChunk) -> np.ndarray:
        """
        Convert audio bytes to numpy array

        Args:
            audio: Audio chunk

        Returns:
            Numpy array of audio samples
        """
        if audio.format == "int16":
            # Convert int16 bytes to float32 array
            audio_array = np.frombuffer(audio.data, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
        elif audio.format == "float32":
            audio_array = np.frombuffer(audio.data, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported audio format: {audio.format}")

        # Ensure correct sample rate (Whisper expects 16kHz)
        if audio.sample_rate != 16000:
            # Simple resampling (should use proper resampling library in production)
            import scipy.signal as signal
            samples = int(len(audio_array) * 16000 / audio.sample_rate)
            audio_array = signal.resample(audio_array, samples)

        return audio_array

    def _merge_audio_chunks(self, chunks: list[AudioChunk]) -> AudioChunk:
        """
        Merge multiple audio chunks into one

        Args:
            chunks: List of audio chunks

        Returns:
            Merged audio chunk
        """
        if not chunks:
            raise ValueError("No audio chunks to merge")

        # Concatenate audio data
        audio_arrays = [self._bytes_to_array(chunk) for chunk in chunks]
        merged_array = np.concatenate(audio_arrays)

        # Convert back to bytes
        if chunks[0].format == "int16":
            merged_array = (merged_array * 32768.0).astype(np.int16)
            merged_data = merged_array.tobytes()
        else:
            merged_data = merged_array.astype(np.float32).tobytes()

        # Create merged chunk
        return AudioChunk(
            data=merged_data,
            sample_rate=chunks[0].sample_rate,
            timestamp=chunks[0].timestamp,
            channels=chunks[0].channels,
            format=chunks[0].format,
        )

    def supports_streaming(self) -> bool:
        """Faster-Whisper supports streaming transcription"""
        return True

    def supported_languages(self) -> list[str]:
        """Get list of supported language codes"""
        return self._supported_languages

    async def health_check(self) -> bool:
        """Check if provider is healthy"""
        if self.status != ProviderStatus.READY:
            return False

        if self.model is None:
            return False

        # Optional: Test transcription with dummy audio
        try:
            # Create 1 second of silence
            dummy_audio = np.zeros(16000, dtype=np.float32)
            segments, _ = self.model.transcribe(
                dummy_audio,
                language="en",
                beam_size=1,
                best_of=1,
            )
            # Just consume the iterator to verify model works
            list(segments)
            return True
        except Exception:
            return False

    def __repr__(self) -> str:
        return (
            f"FasterWhisperProvider(model={self.model_size}, "
            f"device={self.device}, compute_type={self.compute_type}, "
            f"status={self.status.value})"
        )


# Register provider
from ..base import register_provider
register_provider("stt", "faster_whisper", FasterWhisperProvider)
