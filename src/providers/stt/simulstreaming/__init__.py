"""
SimulStreaming - Vendored core components for streaming ASR

This package contains vendored code from:
https://github.com/ufal/SimulStreaming

Core components:
- simul_whisper: PaddedAlignAttWhisper with AlignAtt policy
- token_buffer: Context management across 30s windows
- whisper: Custom Whisper implementation
- whisper_streaming: Online processing interface and VAD
"""

from .simul_whisper import PaddedAlignAttWhisper
from .config import AlignAttConfig, SimulWhisperConfig
from .token_buffer import TokenBuffer
from .whisper_streaming.base import ASRBase, OnlineProcessorInterface
from .whisper_streaming.vac_online_processor import VACOnlineASRProcessor

__all__ = [
    "PaddedAlignAttWhisper",
    "AlignAttConfig",
    "SimulWhisperConfig",
    "TokenBuffer",
    "ASRBase",
    "OnlineProcessorInterface",
    "VACOnlineASRProcessor",
]
