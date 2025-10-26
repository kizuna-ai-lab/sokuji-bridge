"""
SimulStreaming STT Provider

High-performance streaming speech-to-text using SimulStreaming's AlignAtt policy.
Provides low-latency (<2s) streaming transcription with attention-guided token emission.

Features:
- AlignAtt policy for early token emission (2-4s latency reduction)
- Beam search decoding for higher accuracy
- Context management across 30-second windows
- VAD integration for speech segmentation
- Sliding window audio buffer
"""

import numpy as np
import torch
import time
import librosa
from typing import AsyncIterator, Optional, Dict, Any
from pathlib import Path
from loguru import logger

from src.providers.base import (
    STTProvider,
    AudioChunk,
    TranscriptionResult,
    ProviderStatus,
)

from .simulstreaming import (
    PaddedAlignAttWhisper,
    AlignAttConfig,
    OnlineProcessorInterface,
    VACOnlineASRProcessor,
)


class SimulWhisperOnline(OnlineProcessorInterface):
    """
    Online processor for SimulStreaming ASR

    Implements the streaming interface for progressive transcription results.
    """
    SAMPLING_RATE = 16000

    def __init__(self, model: PaddedAlignAttWhisper):
        self.model = model
        self.init()

    def init(self, offset=None):
        """Initialize or reset the processor state"""
        self.audio_chunks = []
        self.offset = offset if offset is not None else 0
        self.is_last = False
        self.beg = self.offset
        self.end = self.offset
        self.audio_buffer_offset = self.offset
        self.last_ts = -1
        self.model.refresh_segment(complete=True)
        self.unicode_buffer = []  # Hide incomplete unicode characters
        self.accumulated_tokens = []  # Accumulate all tokens for full sentence display

    def insert_audio_chunk(self, audio: np.ndarray):
        """Insert audio chunk for processing"""
        self.audio_chunks.append(torch.from_numpy(audio))

    def hide_incomplete_unicode(self, tokens):
        """
        Hide incomplete unicode characters at the end of token sequence

        Sometimes the last token is an incomplete unicode character (e.g., part of "ň" or "ř").
        This function hides it and adds it back in the next iteration.
        """
        if self.unicode_buffer:
            logger.debug(f"Adding buffered unicode: {self.unicode_buffer}")
            tokens = self.unicode_buffer + tokens
            self.unicode_buffer = []

        chars, _ = self.model.tokenizer.split_tokens_on_unicode(tokens)
        if len(chars) > 0 and chars[-1].endswith('�'):
            self.unicode_buffer = tokens[-1:]
            logger.debug(f"Buffering incomplete unicode: {tokens[-1:]}")
            return tokens[:-1]
        return tokens

    def timestamped_text(self, tokens, generation):
        """Generate word-level timestamps from tokens and attention frames"""
        if not generation:
            return []

        pr = generation["progress"]
        if "result" not in generation or self.unicode_buffer:
            split_words, split_tokens = self.model.tokenizer.split_to_word_tokens(tokens)
        else:
            split_words = generation["result"]["split_words"]
            split_tokens = generation["result"]["split_tokens"]

        frames = [p["most_attended_frames"][0] for p in pr]
        if frames and self.unicode_buffer:
            # Add frames for buffered unicode tokens
            a = [frames[0]] * len(self.unicode_buffer)
            frames = a + frames

        tokens = tokens.copy()
        ret = []
        for sw, st in zip(split_words, split_tokens):
            b = None
            for stt in st:
                t, f = tokens.pop(0), frames.pop(0)
                if t != stt:
                    raise ValueError(f"Token mismatch: {t} != {stt} at frame {f}.")
                if b is None:
                    b = f
            e = f
            out = {
                'start': b * 0.02 + self.audio_buffer_offset,
                'end': e * 0.02 + self.audio_buffer_offset,
                'text': sw,
                'tokens': st
            }
            ret.append(out)
            logger.debug(f"Word timing: {out}")
        return ret

    def process_iter(self):
        """Process accumulated audio and return incremental transcription result"""
        if len(self.audio_chunks) == 0:
            audio = None
        else:
            audio = torch.cat(self.audio_chunks, dim=0)
            if audio.shape[0] == 0:
                audio = None
            else:
                self.end += audio.shape[0] / self.SAMPLING_RATE

        self.audio_chunks = []
        self.audio_buffer_offset += self.model.insert_audio(audio)
        new_tokens, generation_progress = self.model.infer(is_last=self.is_last)

        # Accumulate tokens for full sentence display
        if new_tokens:
            self.accumulated_tokens.extend(new_tokens)

        # Use accumulated tokens for text decoding
        tokens = self.hide_incomplete_unicode(self.accumulated_tokens)
        text = self.model.tokenizer.decode(tokens)

        if len(text) == 0:
            return {}

        # Word-level timestamps (use new tokens for timing)
        ts_words = self.timestamped_text(new_tokens, generation_progress) if new_tokens else []

        if ts_words:
            self.beg = min(word['start'] for word in ts_words)
            self.beg = max(self.beg, self.last_ts + 0.001)  # Non-decreasing timestamps

            if self.is_last:
                e = self.end
            else:
                e = max(word['end'] for word in ts_words)
            e = max(e, self.beg + 0.001)

            self.last_ts = e
        else:
            self.beg = self.last_ts
            e = self.end

        return {
            'start': self.beg,
            'end': e,
            'text': text,
            'tokens': tokens,
            'words': ts_words
        }

    def finish(self):
        """Finish processing and return final result"""
        logger.info("Finishing transcription")
        self.is_last = True
        o = self.process_iter()
        self.is_last = False
        self.model.refresh_segment(complete=True)
        return o


class SimulStreamingProvider(STTProvider):
    """
    SimulStreaming STT provider using AlignAtt policy

    Features:
    - Low-latency streaming (<2s) via attention-guided decoding
    - Beam search for higher accuracy
    - Context preservation across 30s windows
    - VAD integration for speech segmentation
    - Progressive token emission
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SimulStreaming provider

        Args:
            config: Configuration dictionary with:
                - model_size: Model size (large-v2, large-v3)
                - device: Device to use (cpu, cuda, auto)
                - compute_type: Computation type (int8, float16, float32)
                - frame_threshold: Frames before end to stop decoding (default: 30 for <2s latency)
                - beam_size: Beam search size (default: 5)
                - audio_max_len: Max audio buffer length in seconds (default: 30.0)
                - audio_min_len: Min audio length before processing (default: 0.3)
                - min_chunk_size: Minimum chunk size in seconds (default: 0.3)
                - vad_enabled: Enable VAD filtering (default: True)
                - language: Optional language hint
                - max_context_tokens: Max context tokens across windows (default: 224)
                - init_prompt: Initial prompt for terminology
                - static_init_prompt: Static context that doesn't scroll
        """
        super().__init__(config)

        self.model_size = config.get("model_size", "large-v3")
        self.model_path = config.get("model_path", f"./{self.model_size}.pt")
        self.device = config.get("device", "auto")
        self.compute_type = config.get("compute_type", "float16")
        self.language = config.get("language", "auto")

        # AlignAtt configuration for low latency
        self.frame_threshold = config.get("frame_threshold", 30)  # ~0.6s advance
        self.rewind_threshold = config.get("rewind_threshold", 999)  # Disable rewind

        # Audio buffer configuration
        self.audio_max_len = config.get("audio_max_len", 30.0)
        self.audio_min_len = config.get("audio_min_len", 0.3)
        self.min_chunk_size = config.get("min_chunk_size", 0.3)

        # Beam search configuration
        self.beam_size = config.get("beam_size", 5)
        self.decoder_type = "beam" if self.beam_size > 1 else "greedy"

        # Context management
        self.max_context_tokens = config.get("max_context_tokens", 224)
        self.init_prompt = config.get("init_prompt")
        self.static_init_prompt = config.get("static_init_prompt")

        # VAD configuration for automatic sentence segmentation
        self.vad_enabled = config.get("vad_enabled", False)  # Default False for backward compatibility
        self.vad_threshold = config.get("vad_threshold", 0.5)  # Speech detection threshold
        self.vad_min_silence_ms = config.get("vad_min_silence_ms", 500)  # Silence duration before sentence end
        self.vad_speech_pad_ms = config.get("vad_speech_pad_ms", 100)  # Padding around speech segments
        self.vad_min_buffered_length = config.get("vad_min_buffered_length", 1.0)  # Minimum buffer length in seconds
        self.vac_chunk_size = config.get("vac_chunk_size", 0.04)  # VAC processing chunk size

        # CIF configuration (optional)
        self.cif_ckpt_path = config.get("cif_ckpt_path")
        self.never_fire = config.get("never_fire", False)

        # Task
        self.task = config.get("task", "transcribe")

        # Model instance (initialized in initialize())
        self.model = None
        self.online_processor = None

        # Supported languages
        self._supported_languages = [
            "en", "zh", "ja", "ko", "es", "fr", "de", "ru", "pt", "it",
            "ar", "hi", "th", "vi", "id", "tr", "pl", "nl", "sv", "fi",
        ]

    async def initialize(self) -> None:
        """Initialize SimulStreaming model"""
        if self.status == ProviderStatus.READY:
            return

        self.status = ProviderStatus.INITIALIZING

        try:
            # Auto-detect device
            if self.device == "auto":
                try:
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    self.device = "cpu"

            logger.info(f"Initializing SimulStreaming with device: {self.device}")

            # Build AlignAttConfig
            cfg = AlignAttConfig(
                model_path=self.model_path,
                segment_length=self.min_chunk_size,
                frame_threshold=self.frame_threshold,
                rewind_threshold=self.rewind_threshold,
                language=self.language,
                audio_max_len=self.audio_max_len,
                audio_min_len=self.audio_min_len,
                cif_ckpt_path=self.cif_ckpt_path,
                decoder_type=self.decoder_type,
                beam_size=self.beam_size,
                task=self.task,
                never_fire=self.never_fire,
                init_prompt=self.init_prompt,
                static_init_prompt=self.static_init_prompt,
                max_context_tokens=self.max_context_tokens,
                logdir=None,  # Disable debug logging
            )

            # Initialize PaddedAlignAttWhisper
            logger.info(f"Loading model: {self.model_size}")
            self.model = PaddedAlignAttWhisper(cfg)

            # Create online processor (with or without VAD)
            if self.vad_enabled:
                logger.info("🎙️ Enabling VAD for automatic sentence segmentation")
                logger.info(f"📊 VAD Config: threshold={self.vad_threshold}, "
                           f"silence={self.vad_min_silence_ms}ms, "
                           f"padding={self.vad_speech_pad_ms}ms")

                # Create base processor
                base_processor = SimulWhisperOnline(self.model)

                # Wrap with VAC for automatic sentence segmentation
                self.online_processor = VACOnlineASRProcessor(
                    online_chunk_size=self.vac_chunk_size,
                    online=base_processor,
                    min_buffered_length=self.vad_min_buffered_length
                )
                logger.info("✅ VAD wrapper initialized successfully")
            else:
                logger.info("Using standard streaming mode (VAD disabled)")
                self.online_processor = SimulWhisperOnline(self.model)

            # Warmup
            logger.info("Warming up model...")
            warmup_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
            self.model.insert_audio(warmup_audio)
            self.model.infer(is_last=True)
            self.model.refresh_segment(complete=True)

            logger.info("SimulStreaming initialized successfully")
            self.status = ProviderStatus.READY

        except Exception as e:
            self.status = ProviderStatus.ERROR
            logger.error(f"Failed to initialize SimulStreaming: {e}")
            raise RuntimeError(f"Failed to initialize SimulStreaming: {e}") from e

    async def cleanup(self) -> None:
        """Clean up SimulStreaming model"""
        if self.model:
            del self.model
            self.model = None
        if self.online_processor:
            del self.online_processor
            self.online_processor = None

        self.status = ProviderStatus.STOPPED

    async def transcribe(
        self,
        audio: AudioChunk,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe single audio chunk (batch mode)

        Args:
            audio: Audio chunk to transcribe
            language: Optional language hint

        Returns:
            Transcription result
        """
        if self.status != ProviderStatus.READY:
            raise RuntimeError("Provider not initialized")

        start_time = time.time()

        try:
            # Convert to numpy array
            audio_array = self._chunk_to_array(audio)

            # Create temporary online processor
            temp_processor = SimulWhisperOnline(self.model)
            temp_processor.init()

            # Insert and process
            temp_processor.insert_audio_chunk(audio_array)
            temp_processor.is_last = True
            output = temp_processor.process_iter()

            duration_ms = (time.time() - start_time) * 1000
            self._record_request(duration_ms, error=False)

            if not output or not output.get('text'):
                return TranscriptionResult(
                    text="",
                    language=self.model.detected_language or language or "auto",
                    confidence=0.0,
                    timestamp=audio.timestamp,
                    start_time=audio.timestamp,
                    end_time=audio.timestamp + audio.duration_ms / 1000,
                    is_final=True,
                )

            return TranscriptionResult(
                text=output['text'],
                language=self.model.detected_language or language or "auto",
                confidence=1.0,  # SimulStreaming doesn't provide confidence scores
                timestamp=audio.timestamp,
                start_time=output['start'],
                end_time=output['end'],
                is_final=True,
            )

        except Exception as e:
            self._record_request(0, error=True)
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"Transcription failed: {e}") from e

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk],
        language: Optional[str] = None,
    ) -> AsyncIterator[TranscriptionResult]:
        """
        Transcribe audio stream in real-time with progressive results

        Args:
            audio_stream: Async iterator of audio chunks
            language: Optional language hint

        Yields:
            Progressive transcription results
        """
        if self.status != ProviderStatus.READY:
            raise RuntimeError("Provider not initialized")

        # Initialize online processor
        self.online_processor.init()

        logger.info("🎙️ Starting streaming transcription with SimulStreaming")
        logger.info(f"📊 Config: frame_threshold={self.frame_threshold}, "
                   f"beam_size={self.beam_size}, "
                   f"audio_min_len={self.audio_min_len}s, "
                   f"vad_enabled={self.vad_enabled}")

        try:
            chunk_count = 0
            async for audio_chunk in audio_stream:
                chunk_count += 1

                # Convert and insert audio
                audio_array = self._chunk_to_array(audio_chunk)
                self.online_processor.insert_audio_chunk(audio_array)

                logger.debug(f"📦 Chunk {chunk_count}: {audio_chunk.duration_ms:.0f}ms, "
                           f"{len(audio_chunk.data)} bytes")

                # Process and get incremental result
                output = self.online_processor.process_iter()

                if output and output.get('text'):
                    # Check if VAD detected sentence boundary
                    is_sentence_final = False
                    if self.vad_enabled and hasattr(self.online_processor, 'is_currently_final'):
                        is_sentence_final = self.online_processor.is_currently_final
                        if is_sentence_final:
                            logger.info(f"🎯 VAD detected sentence end: {output['text']}")
                        else:
                            logger.info(f"💬 Progressive: {output['text']}")
                    else:
                        logger.info(f"💬 Progressive: {output['text']}")

                    yield TranscriptionResult(
                        text=output['text'],
                        language=self.model.detected_language or language or "auto",
                        confidence=1.0,
                        timestamp=audio_chunk.timestamp,
                        start_time=output['start'],
                        end_time=output['end'],
                        is_final=is_sentence_final,  # True if VAD detected sentence boundary
                    )

            # Finish and get final result
            logger.info("🏁 Finishing transcription stream")
            final_output = self.online_processor.finish()

            if final_output and final_output.get('text'):
                logger.info(f"✅ Final: {final_output['text']}")

                yield TranscriptionResult(
                    text=final_output['text'],
                    language=self.model.detected_language or language or "auto",
                    confidence=1.0,
                    timestamp=time.time(),
                    start_time=final_output['start'],
                    end_time=final_output['end'],
                    is_final=True,
                )

        except Exception as e:
            logger.error(f"Streaming transcription failed: {e}")
            raise RuntimeError(f"Streaming transcription failed: {e}") from e

    def _chunk_to_array(self, audio: AudioChunk) -> np.ndarray:
        """
        Convert audio chunk to numpy array

        Args:
            audio: Audio chunk

        Returns:
            Numpy array of audio samples (float32, [-1, 1])
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
            audio_array = librosa.resample(
                audio_array,
                orig_sr=audio.sample_rate,
                target_sr=16000
            )

        return audio_array

    def supports_streaming(self) -> bool:
        """SimulStreaming supports streaming transcription"""
        return True

    def supported_languages(self) -> list[str]:
        """Get list of supported language codes"""
        return self._supported_languages

    async def health_check(self) -> bool:
        """Check if provider is healthy"""
        if self.status != ProviderStatus.READY:
            return False

        if self.model is None or self.online_processor is None:
            return False

        return True

    def __repr__(self) -> str:
        return (
            f"SimulStreamingProvider(model={self.model_size}, "
            f"device={self.device}, frame_threshold={self.frame_threshold}, "
            f"beam_size={self.beam_size}, status={self.status.value})"
        )


# Register provider
from src.providers.base import register_provider
register_provider("stt", "simulstreaming", SimulStreamingProvider)
