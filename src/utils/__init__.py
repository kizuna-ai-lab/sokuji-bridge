"""
Audio I/O Utilities for Sokuji-Bridge

Provides audio input/output abstractions and implementations for:
- Microphone input
- Speaker output
- Audio file reading/writing
- Voice Activity Detection (VAD)
"""

# Core audio I/O abstractions
from src.utils.audio_io import (
    AudioInput,
    AudioOutput,
    AudioDevice,
    list_audio_devices,
    get_default_input_device,
    get_default_output_device,
)

# Microphone input
# from utils.microphone import MicrophoneInput  # Temporarily disabled

# Speaker output
from src.utils.speaker import SpeakerOutput

# Audio file I/O
from src.utils.audio_file import AudioFileReader, AudioFileWriter

# Voice Activity Detection
from src.utils.vad import (
    VADProvider,
    VADResult,
    SpeechState,
    DummyVAD,
    EnergyVAD,
)

__all__ = [
    # Core abstractions
    "AudioInput",
    "AudioOutput",
    "AudioDevice",
    "list_audio_devices",
    "get_default_input_device",
    "get_default_output_device",
    # Implementations
    # "MicrophoneInput",  # Temporarily disabled
    "SpeakerOutput",
    "AudioFileReader",
    "AudioFileWriter",
    # VAD
    "VADProvider",
    "VADResult",
    "SpeechState",
    "DummyVAD",
    "EnergyVAD",
]
