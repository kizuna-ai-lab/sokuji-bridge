"""
Audio I/O Abstract Base Classes

Defines unified interfaces for audio input and output operations.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, Dict, Any, List
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.providers.base import AudioChunk, SynthesisResult


@dataclass
class AudioDevice:
    """Audio device information"""
    index: int
    name: str
    channels: int
    sample_rate: int
    is_input: bool
    is_output: bool
    is_default: bool = False

    def __str__(self) -> str:
        device_type = []
        if self.is_input:
            device_type.append("input")
        if self.is_output:
            device_type.append("output")
        default = " [DEFAULT]" if self.is_default else ""
        return f"{self.name} ({'/'.join(device_type)}){default}"


class AudioInput(ABC):
    """
    Abstract base class for audio input sources

    Implementations: MicrophoneInput, AudioFileReader, NetworkAudioInput
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize audio input

        Args:
            config: Configuration dictionary with device settings
        """
        self.config = config or {}
        self._is_running = False

    @abstractmethod
    async def start(self) -> None:
        """
        Start audio input stream

        Raises:
            RuntimeError: If starting input fails
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """
        Stop audio input stream
        """
        pass

    @abstractmethod
    async def stream(self) -> AsyncIterator[AudioChunk]:
        """
        Generate audio chunks from input source

        Yields:
            AudioChunk objects with audio data

        Example:
            >>> async for chunk in audio_input.stream():
            >>>     print(f"Received {chunk.duration_ms}ms of audio")
        """
        pass

    @property
    def is_running(self) -> bool:
        """Check if input is currently running"""
        return self._is_running

    @staticmethod
    @abstractmethod
    def get_devices() -> List[AudioDevice]:
        """
        List available input devices

        Returns:
            List of available audio input devices
        """
        pass

    async def __aenter__(self):
        """Context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.stop()


class AudioOutput(ABC):
    """
    Abstract base class for audio output destinations

    Implementations: SpeakerOutput, AudioFileWriter, NetworkAudioOutput
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize audio output

        Args:
            config: Configuration dictionary with device settings
        """
        self.config = config or {}
        self._is_running = False

    @abstractmethod
    async def start(self) -> None:
        """
        Start audio output stream

        Raises:
            RuntimeError: If starting output fails
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """
        Stop audio output stream and flush buffers
        """
        pass

    @abstractmethod
    async def play(self, audio: SynthesisResult) -> None:
        """
        Play a single audio chunk

        Args:
            audio: Synthesized audio result to play

        Raises:
            RuntimeError: If playback fails
        """
        pass

    @abstractmethod
    async def play_stream(self, audio_stream: AsyncIterator[SynthesisResult]) -> None:
        """
        Play audio stream

        Args:
            audio_stream: Async iterator of synthesis results

        Example:
            >>> async for result in pipeline.process_audio_stream(input_stream):
            >>>     await speaker.play_stream([result])
        """
        pass

    @property
    def is_running(self) -> bool:
        """Check if output is currently running"""
        return self._is_running

    @staticmethod
    @abstractmethod
    def get_devices() -> List[AudioDevice]:
        """
        List available output devices

        Returns:
            List of available audio output devices
        """
        pass

    async def __aenter__(self):
        """Context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.stop()


# Utility functions for device management

def list_audio_devices(input_only: bool = False, output_only: bool = False) -> List[AudioDevice]:
    """
    List all available audio devices

    Args:
        input_only: Only list input devices
        output_only: Only list output devices

    Returns:
        List of audio devices
    """
    import sounddevice as sd

    devices = []
    for idx, device_info in enumerate(sd.query_devices()):
        is_input = device_info['max_input_channels'] > 0
        is_output = device_info['max_output_channels'] > 0

        # Filter based on parameters
        if input_only and not is_input:
            continue
        if output_only and not is_output:
            continue

        device = AudioDevice(
            index=idx,
            name=device_info['name'],
            channels=max(device_info['max_input_channels'], device_info['max_output_channels']),
            sample_rate=int(device_info['default_samplerate']),
            is_input=is_input,
            is_output=is_output,
            is_default=(idx == sd.default.device[0] if is_input else idx == sd.default.device[1])
        )
        devices.append(device)

    return devices


def get_default_input_device() -> Optional[AudioDevice]:
    """
    Get the default audio input device

    Returns:
        Default input device or None if not found
    """
    devices = list_audio_devices(input_only=True)
    for device in devices:
        if device.is_default:
            return device
    return devices[0] if devices else None


def get_default_output_device() -> Optional[AudioDevice]:
    """
    Get the default audio output device

    Returns:
        Default output device or None if not found
    """
    devices = list_audio_devices(output_only=True)
    for device in devices:
        if device.is_default:
            return device
    return devices[0] if devices else None
