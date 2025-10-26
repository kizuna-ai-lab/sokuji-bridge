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
from src.utils.microphone import MicrophoneInput

# Speaker output
from src.utils.speaker import SpeakerOutput

# Audio file I/O
from src.utils.audio_file import AudioFileReader, AudioFileWriter

# Voice Activity Detection
# from src.utils.vad import (
#     VADProvider,
#     VADResult,
#     SpeechState,
#     DummyVAD,
#     EnergyVAD,
# )

__all__ = [
    # Core abstractions
    "AudioInput",
    "AudioOutput",
    "AudioDevice",
    "list_audio_devices",
    "get_default_input_device",
    "get_default_output_device",
    # Implementations
    "MicrophoneInput",
    "SpeakerOutput",
    "AudioFileReader",
    "AudioFileWriter",
    # VAD - Disabled until vad.py is implemented
    # "VADProvider",
    # "VADResult",
    # "SpeechState",
    # "DummyVAD",
    # "EnergyVAD",
]
