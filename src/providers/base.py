"""
Base Provider Classes for Sokuji-Bridge

Defines abstract base classes for STT, Translation, and TTS providers,
ensuring a unified interface for all implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Optional, Dict, Any
import time


class ProviderStatus(Enum):
    """Provider operational status"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class AudioChunk:
    """Audio data chunk with metadata"""
    data: bytes
    sample_rate: int
    timestamp: float
    channels: int = 1
    format: str = "int16"  # int16, float32, etc.

    @property
    def duration_ms(self) -> float:
        """Calculate duration in milliseconds"""
        bytes_per_sample = 2 if self.format == "int16" else 4
        samples = len(self.data) // (bytes_per_sample * self.channels)
        return (samples / self.sample_rate) * 1000


@dataclass
class TranscriptionResult:
    """Speech-to-text transcription result"""
    text: str
    language: str
    confidence: float
    timestamp: float
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    is_final: bool = True

    def __str__(self) -> str:
        return f"[{self.language}] {self.text} (conf: {self.confidence:.2f})"


@dataclass
class TranslationResult:
    """Translation result with metadata"""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: float
    timestamp: float
    model_name: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.source_language}→{self.target_language}: {self.translated_text}"


@dataclass
class SynthesisResult:
    """Text-to-speech synthesis result"""
    audio_data: bytes
    sample_rate: int
    text: str
    voice_id: str
    timestamp: float
    duration_ms: float
    format: str = "int16"

    def __str__(self) -> str:
        return f"Synthesized {len(self.text)} chars → {self.duration_ms:.0f}ms audio"


class BaseProvider(ABC):
    """Base class for all providers"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.status = ProviderStatus.UNINITIALIZED
        self._metrics: Dict[str, Any] = {
            "total_requests": 0,
            "total_errors": 0,
            "total_duration_ms": 0.0,
        }

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the provider (load models, establish connections, etc.)

        Raises:
            RuntimeError: If initialization fails
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """
        Clean up resources (unload models, close connections, etc.)
        """
        pass

    async def health_check(self) -> bool:
        """
        Check if the provider is healthy and ready to process requests

        Returns:
            True if healthy, False otherwise
        """
        return self.status == ProviderStatus.READY

    def get_metrics(self) -> Dict[str, Any]:
        """Get provider performance metrics"""
        return self._metrics.copy()

    def _record_request(self, duration_ms: float, error: bool = False) -> None:
        """Record request metrics"""
        self._metrics["total_requests"] += 1
        if error:
            self._metrics["total_errors"] += 1
        else:
            self._metrics["total_duration_ms"] += duration_ms
            # Calculate average
            successful_requests = self._metrics["total_requests"] - self._metrics["total_errors"]
            if successful_requests > 0:
                self._metrics["avg_duration_ms"] = (
                    self._metrics["total_duration_ms"] / successful_requests
                )


class STTProvider(BaseProvider):
    """
    Abstract base class for Speech-to-Text providers

    Implementations: faster-whisper, Whisper API, Azure STT, Google STT
    """

    @abstractmethod
    async def transcribe(
        self,
        audio: AudioChunk,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe a single audio chunk

        Args:
            audio: Audio chunk to transcribe
            language: Optional language hint (e.g., "zh", "en", "ja")

        Returns:
            Transcription result with text and metadata

        Raises:
            RuntimeError: If transcription fails
        """
        pass

    @abstractmethod
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

        Raises:
            RuntimeError: If streaming transcription fails
        """
        pass

    def supports_streaming(self) -> bool:
        """Check if provider supports streaming transcription"""
        return True

    def supported_languages(self) -> list[str]:
        """Get list of supported language codes"""
        return []  # Override in implementation


class TranslationProvider(BaseProvider):
    """
    Abstract base class for Translation providers

    Implementations: NLLB, DeepL API, Google Translate, GPT-4
    """

    @abstractmethod
    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: Optional[str] = None,
    ) -> TranslationResult:
        """
        Translate text from source to target language

        Args:
            text: Text to translate
            source_lang: Source language code (e.g., "zh", "en")
            target_lang: Target language code
            context: Optional context for better translation

        Returns:
            Translation result with translated text and metadata

        Raises:
            RuntimeError: If translation fails
        """
        pass

    async def translate_batch(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
    ) -> list[TranslationResult]:
        """
        Translate multiple texts in batch (more efficient)

        Default implementation calls translate() for each text.
        Override for batch optimization.

        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            List of translation results
        """
        results = []
        for text in texts:
            result = await self.translate(text, source_lang, target_lang)
            results.append(result)
        return results

    async def translate_stream(
        self,
        text_stream: AsyncIterator[str],
        source_lang: str,
        target_lang: str,
    ) -> AsyncIterator[TranslationResult]:
        """
        Translate text stream (if provider supports streaming)

        Default implementation calls translate() for each text.
        Override for streaming optimization (e.g., GPT-4 streaming).

        Args:
            text_stream: Async iterator of texts
            source_lang: Source language code
            target_lang: Target language code

        Yields:
            Translation results as they become available
        """
        async for text in text_stream:
            result = await self.translate(text, source_lang, target_lang)
            yield result

    def supports_batch(self) -> bool:
        """Check if provider supports efficient batch translation"""
        return False  # Override if supported

    def supports_streaming(self) -> bool:
        """Check if provider supports streaming translation"""
        return False  # Override if supported

    def supported_languages(self) -> list[str]:
        """Get list of supported language codes"""
        return []  # Override in implementation


class TTSProvider(BaseProvider):
    """
    Abstract base class for Text-to-Speech providers

    Implementations: Piper, Kokoro, XTTS v2, ElevenLabs
    """

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice_id: str,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> SynthesisResult:
        """
        Synthesize speech from text

        Args:
            text: Text to synthesize
            voice_id: Voice identifier (provider-specific)
            language: Optional language hint
            **kwargs: Provider-specific parameters (speed, pitch, etc.)

        Returns:
            Synthesis result with audio data and metadata

        Raises:
            RuntimeError: If synthesis fails
        """
        pass

    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str],
        voice_id: str,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[SynthesisResult]:
        """
        Synthesize speech from text stream

        Default implementation calls synthesize() for each text.
        Override for streaming optimization.

        Args:
            text_stream: Async iterator of texts
            voice_id: Voice identifier
            language: Optional language hint
            **kwargs: Provider-specific parameters

        Yields:
            Synthesis results as they become available
        """
        async for text in text_stream:
            result = await self.synthesize(text, voice_id, language, **kwargs)
            yield result

    async def get_voices(self) -> list[Dict[str, Any]]:
        """
        Get available voices for this provider

        Returns:
            List of voice dictionaries with id, name, language, etc.
        """
        return []  # Override in implementation

    def supports_streaming(self) -> bool:
        """Check if provider supports streaming synthesis"""
        return False  # Override if supported

    def supports_voice_cloning(self) -> bool:
        """Check if provider supports voice cloning"""
        return False  # Override if supported

    def supported_languages(self) -> list[str]:
        """Get list of supported language codes"""
        return []  # Override in implementation


# Provider registry for dynamic loading
_provider_registry: Dict[str, type[BaseProvider]] = {}


def register_provider(category: str, name: str, provider_class: type[BaseProvider]) -> None:
    """
    Register a provider for dynamic loading

    Args:
        category: Provider category ("stt", "translation", "tts")
        name: Provider name (e.g., "faster_whisper", "deepl")
        provider_class: Provider class
    """
    key = f"{category}.{name}"
    _provider_registry[key] = provider_class


def get_provider_class(category: str, name: str) -> Optional[type[BaseProvider]]:
    """
    Get provider class by category and name

    Args:
        category: Provider category ("stt", "translation", "tts")
        name: Provider name

    Returns:
        Provider class or None if not found
    """
    key = f"{category}.{name}"
    return _provider_registry.get(key)


def list_providers(category: Optional[str] = None) -> list[str]:
    """
    List registered providers

    Args:
        category: Optional category filter

    Returns:
        List of provider keys
    """
    if category:
        prefix = f"{category}."
        return [k for k in _provider_registry.keys() if k.startswith(prefix)]
    return list(_provider_registry.keys())
