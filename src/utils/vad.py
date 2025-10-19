"""
Voice Activity Detection (VAD) Interface

Abstract interface for VAD implementations.
Prepares for future Silero VAD and WebRTC VAD integration.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from providers.base import AudioChunk


class SpeechState(Enum):
    """Speech detection state"""
    SILENCE = "silence"
    SPEECH = "speech"
    UNKNOWN = "unknown"


@dataclass
class VADResult:
    """VAD detection result"""
    is_speech: bool
    confidence: float  # 0.0 to 1.0
    timestamp: float
    state: SpeechState

    def __str__(self) -> str:
        return f"VAD({self.state.value}, conf={self.confidence:.2f})"


class VADProvider(ABC):
    """
    Abstract base class for Voice Activity Detection

    Implementations:
    - SileroVAD: Deep learning-based VAD using Silero models
    - WebRTCVAD: Fast traditional VAD using WebRTC
    - EnergyVAD: Simple energy-based VAD
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize VAD provider

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._is_initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize VAD model/algorithm

        Raises:
            RuntimeError: If initialization fails
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up VAD resources"""
        pass

    @abstractmethod
    async def detect_speech(self, audio: AudioChunk) -> VADResult:
        """
        Detect speech in audio chunk

        Args:
            audio: Audio chunk to analyze

        Returns:
            VAD result with speech detection and confidence

        Example:
            >>> result = await vad.detect_speech(audio_chunk)
            >>> if result.is_speech:
            >>>     print("Speech detected!")
        """
        pass

    @abstractmethod
    async def segment_audio(
        self,
        audio_stream: AsyncIterator[AudioChunk],
        min_speech_duration_ms: float = 250,
        min_silence_duration_ms: float = 500,
    ) -> AsyncIterator[AudioChunk]:
        """
        Segment audio stream into speech segments

        Intelligently combines consecutive audio chunks with speech
        and splits on silence periods.

        Args:
            audio_stream: Input audio stream
            min_speech_duration_ms: Minimum speech duration to keep
            min_silence_duration_ms: Minimum silence duration to split

        Yields:
            Segmented audio chunks containing complete speech utterances

        Example:
            >>> async for segment in vad.segment_audio(microphone.stream()):
            >>>     # Each segment contains a complete speech utterance
            >>>     result = await stt.transcribe(segment)
        """
        pass

    @property
    def is_initialized(self) -> bool:
        """Check if VAD is initialized"""
        return self._is_initialized

    def get_config(self) -> Dict[str, Any]:
        """Get VAD configuration"""
        return self.config.copy()


class DummyVAD(VADProvider):
    """
    Dummy VAD implementation for testing

    Always detects speech with 100% confidence.
    Useful for testing pipelines without actual VAD.
    """

    async def initialize(self) -> None:
        """Initialize dummy VAD (no-op)"""
        self._is_initialized = True

    async def cleanup(self) -> None:
        """Cleanup dummy VAD (no-op)"""
        self._is_initialized = False

    async def detect_speech(self, audio: AudioChunk) -> VADResult:
        """
        Always return speech detected

        Args:
            audio: Audio chunk (ignored)

        Returns:
            VAD result indicating speech with 100% confidence
        """
        return VADResult(
            is_speech=True,
            confidence=1.0,
            timestamp=audio.timestamp,
            state=SpeechState.SPEECH,
        )

    async def segment_audio(
        self,
        audio_stream: AsyncIterator[AudioChunk],
        min_speech_duration_ms: float = 250,
        min_silence_duration_ms: float = 500,
    ) -> AsyncIterator[AudioChunk]:
        """
        Pass through all audio chunks (no segmentation)

        Args:
            audio_stream: Input audio stream
            min_speech_duration_ms: Ignored
            min_silence_duration_ms: Ignored

        Yields:
            All input audio chunks unchanged
        """
        async for chunk in audio_stream:
            yield chunk

    def __repr__(self) -> str:
        return "DummyVAD(always_speech=True)"


class EnergyVAD(VADProvider):
    """
    Simple energy-based VAD implementation

    Detects speech based on audio energy/volume.
    Fast but less accurate than ML-based methods.
    """

    def __init__(
        self,
        energy_threshold: float = 0.01,
        sample_rate: int = 16000,
        **kwargs
    ):
        """
        Initialize energy-based VAD

        Args:
            energy_threshold: Energy threshold for speech detection
            sample_rate: Expected audio sample rate
            **kwargs: Additional configuration
        """
        super().__init__(kwargs)
        self.energy_threshold = energy_threshold
        self.sample_rate = sample_rate

    async def initialize(self) -> None:
        """Initialize energy VAD"""
        self._is_initialized = True

    async def cleanup(self) -> None:
        """Cleanup energy VAD"""
        self._is_initialized = False

    async def detect_speech(self, audio: AudioChunk) -> VADResult:
        """
        Detect speech based on audio energy

        Args:
            audio: Audio chunk to analyze

        Returns:
            VAD result based on energy threshold
        """
        import numpy as np

        # Convert audio to float
        if audio.format == "int16":
            audio_array = np.frombuffer(audio.data, dtype=np.int16).astype(np.float32) / 32768.0
        elif audio.format == "float32":
            audio_array = np.frombuffer(audio.data, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported audio format: {audio.format}")

        # Calculate energy (RMS)
        energy = np.sqrt(np.mean(audio_array ** 2))

        # Determine speech based on energy threshold
        is_speech = energy > self.energy_threshold

        # Confidence based on how far from threshold
        if is_speech:
            confidence = min(1.0, energy / (self.energy_threshold * 2))
        else:
            confidence = max(0.0, 1.0 - (energy / self.energy_threshold))

        return VADResult(
            is_speech=is_speech,
            confidence=confidence,
            timestamp=audio.timestamp,
            state=SpeechState.SPEECH if is_speech else SpeechState.SILENCE,
        )

    async def segment_audio(
        self,
        audio_stream: AsyncIterator[AudioChunk],
        min_speech_duration_ms: float = 250,
        min_silence_duration_ms: float = 500,
    ) -> AsyncIterator[AudioChunk]:
        """
        Segment audio stream based on energy

        Args:
            audio_stream: Input audio stream
            min_speech_duration_ms: Minimum speech duration to keep
            min_silence_duration_ms: Minimum silence duration to split

        Yields:
            Segmented audio chunks
        """
        import numpy as np

        speech_buffer: List[bytes] = []
        speech_duration_ms = 0.0
        silence_duration_ms = 0.0
        last_state = SpeechState.SILENCE
        first_timestamp = None

        async for chunk in audio_stream:
            result = await self.detect_speech(chunk)

            if result.is_speech:
                # Add to speech buffer
                speech_buffer.append(chunk.data)
                speech_duration_ms += chunk.duration_ms
                silence_duration_ms = 0.0

                if first_timestamp is None:
                    first_timestamp = chunk.timestamp

                last_state = SpeechState.SPEECH

            else:
                # Silence detected
                silence_duration_ms += chunk.duration_ms

                if last_state == SpeechState.SPEECH:
                    # Check if we have enough silence to end segment
                    if silence_duration_ms >= min_silence_duration_ms:
                        # Emit speech segment if long enough
                        if speech_duration_ms >= min_speech_duration_ms and speech_buffer:
                            combined_audio = b''.join(speech_buffer)
                            yield AudioChunk(
                                data=combined_audio,
                                sample_rate=chunk.sample_rate,
                                timestamp=first_timestamp or chunk.timestamp,
                                channels=chunk.channels,
                                format=chunk.format,
                            )

                        # Reset buffer
                        speech_buffer.clear()
                        speech_duration_ms = 0.0
                        silence_duration_ms = 0.0
                        first_timestamp = None

                last_state = SpeechState.SILENCE

        # Emit remaining speech if any
        if speech_buffer and speech_duration_ms >= min_speech_duration_ms:
            combined_audio = b''.join(speech_buffer)
            yield AudioChunk(
                data=combined_audio,
                sample_rate=chunk.sample_rate,
                timestamp=first_timestamp or 0.0,
                channels=chunk.channels,
                format=chunk.format,
            )

    def __repr__(self) -> str:
        return f"EnergyVAD(threshold={self.energy_threshold})"


async def test_vad():
    """Test VAD implementations"""
    print("Testing VAD implementations...")
    print()

    # Test DummyVAD
    print("1. Testing DummyVAD...")
    dummy_vad = DummyVAD()
    await dummy_vad.initialize()
    print(f"   {dummy_vad}")

    # Create test audio
    import numpy as np
    audio_samples = np.random.randn(16000).astype(np.float32) * 0.1
    audio_data = (audio_samples * 32768).astype(np.int16).tobytes()
    test_chunk = AudioChunk(audio_data, 16000, 0.0, 1, "int16")

    result = await dummy_vad.detect_speech(test_chunk)
    print(f"   Result: {result}")
    await dummy_vad.cleanup()
    print()

    # Test EnergyVAD
    print("2. Testing EnergyVAD...")
    energy_vad = EnergyVAD(energy_threshold=0.05)
    await energy_vad.initialize()
    print(f"   {energy_vad}")

    # Test with loud audio (should detect speech)
    loud_audio = (np.random.randn(16000).astype(np.float32) * 0.5)
    loud_data = (loud_audio * 32768).astype(np.int16).tobytes()
    loud_chunk = AudioChunk(loud_data, 16000, 0.0, 1, "int16")

    loud_result = await energy_vad.detect_speech(loud_chunk)
    print(f"   Loud audio: {loud_result}")

    # Test with quiet audio (should detect silence)
    quiet_audio = (np.random.randn(16000).astype(np.float32) * 0.01)
    quiet_data = (quiet_audio * 32768).astype(np.int16).tobytes()
    quiet_chunk = AudioChunk(quiet_data, 16000, 0.0, 1, "int16")

    quiet_result = await energy_vad.detect_speech(quiet_chunk)
    print(f"   Quiet audio: {quiet_result}")

    await energy_vad.cleanup()
    print()

    print("VAD tests complete!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_vad())
