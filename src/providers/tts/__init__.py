"""
TTS Provider Module

Available providers:
- piper: Fast local TTS with Piper
- cosyvoice: Multi-lingual TTS with cross-lingual synthesis (CosyVoice2)
"""

# Import providers to trigger registration
from . import piper_provider
from . import cosyvoice_provider

__all__ = ["piper_provider", "cosyvoice_provider"]
