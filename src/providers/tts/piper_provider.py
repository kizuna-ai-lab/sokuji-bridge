"""
Piper TTS Provider

Fast, local text-to-speech using Piper.
Optimized for CPU with high-quality neural voices.

Uses piper1-gpl from https://github.com/OHF-Voice/piper1-gpl
"""

import time
import io
import subprocess
import sys
from typing import AsyncIterator, Optional, Dict, Any
from pathlib import Path

from providers.base import (
    TTSProvider,
    SynthesisResult,
    ProviderStatus,
)


class PiperProvider(TTSProvider):
    """
    Piper TTS provider for fast local speech synthesis

    Features:
    - Ultra-fast CPU inference
    - High-quality neural voices
    - Low memory footprint
    - Multiple language support
    - No API costs
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Piper provider

        Args:
            config: Configuration dictionary with:
                - model: Model name (e.g., "en_US-lessac-medium")
                - speaker_id: Speaker ID for multi-speaker models (default: 0)
                - length_scale: Speech speed multiplier (default: 1.0)
                - noise_scale: Variation in speech (default: 0.667)
                - noise_w: Variation in phoneme durations (default: 0.8)
                - model_path: Optional custom model path
        """
        super().__init__(config)

        self.model_name = config.get("model", "en_US-lessac-medium")
        self.speaker_id = config.get("speaker_id", 0)
        self.length_scale = config.get("length_scale", 1.0)
        self.noise_scale = config.get("noise_scale", 0.667)
        self.noise_w = config.get("noise_w", 0.8)
        self.model_path = config.get("model_path")

        # Piper voice instance (initialized in initialize())
        self.voice = None
        self.sample_rate = 22050  # Piper default

        # Supported voices (language -> model names)
        self._available_voices = {
            "en": ["en_US-lessac-medium", "en_US-amy-medium", "en_GB-alan-medium"],
            "zh": ["zh_CN-huayan-medium"],
            "ja": ["ja_JP-natsu-medium"],
            "es": ["es_ES-sharvard-medium"],
            "fr": ["fr_FR-siwis-medium"],
            "de": ["de_DE-thorsten-medium"],
        }

    def _download_model(self, model_name: str, download_dir: Path) -> None:
        """
        Download a Piper voice model using piper's download utility

        Args:
            model_name: Name of the model to download (e.g., "en_US-lessac-medium")
            download_dir: Directory to download model into
        """
        try:
            print(f"📥 Downloading Piper model: {model_name}")
            print(f"📁 Download directory: {download_dir}")

            # Create directory if it doesn't exist
            download_dir.mkdir(parents=True, exist_ok=True)

            # Run download command with specified directory
            subprocess.run(
                [
                    sys.executable, "-m", "piper.download_voices",
                    model_name,
                    "--download-dir", str(download_dir)
                ],
                check=True,
                capture_output=True,
                text=True
            )
            print(f"✓ Model {model_name} downloaded successfully")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to download model {model_name}: {e.stderr}"
            ) from e

    async def initialize(self) -> None:
        """Initialize Piper voice model"""
        if self.status == ProviderStatus.READY:
            return

        self.status = ProviderStatus.INITIALIZING

        try:
            from piper import PiperVoice

            # Load voice model
            if self.model_path:
                model_path = Path(self.model_path)
            else:
                # Try to find the model
                model_path = self._get_model_path(self.model_name)

            # Auto-download if model doesn't exist
            if not model_path.exists():
                print(f"⚠️  Model not found at {model_path}")
                print(f"🔄 Attempting to download {self.model_name}...")
                try:
                    # Use project-local models directory for downloads
                    download_dir = Path.cwd() / "models" / "piper"
                    self._download_model(self.model_name, download_dir)
                    # Re-check model path after download
                    model_path = self._get_model_path(self.model_name)
                    if not model_path.exists():
                        raise FileNotFoundError(
                            f"Model still not found after download: {model_path}"
                        )
                except Exception as download_error:
                    raise FileNotFoundError(
                        f"Piper model not found: {model_path}. "
                        f"Auto-download failed: {download_error}\n"
                        "Please download manually from: "
                        "https://github.com/OHF-Voice/piper1-gpl or use: "
                        f"python -m piper.download_voices {self.model_name} --download-dir models/piper"
                    ) from download_error

            # Load voice using new piper1-gpl API
            print(f"📂 Loading model from: {model_path}")
            self.voice = PiperVoice.load(str(model_path))
            self.sample_rate = self.voice.config.sample_rate

            self.status = ProviderStatus.READY
            print(f"✓ Piper initialized: {self.model_name} @ {self.sample_rate}Hz")

        except ImportError as e:
            self.status = ProviderStatus.ERROR
            raise RuntimeError(
                "piper-tts not installed. Install with: pip install piper-tts"
            ) from e
        except Exception as e:
            self.status = ProviderStatus.ERROR
            raise RuntimeError(f"Failed to initialize Piper: {e}") from e

    async def cleanup(self) -> None:
        """Clean up Piper voice"""
        if self.voice:
            del self.voice
            self.voice = None

        self.status = ProviderStatus.STOPPED

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
            voice_id: Voice identifier (not used, uses configured model)
            language: Optional language hint
            **kwargs: Additional parameters (speed, etc.)

        Returns:
            Synthesis result with audio data
        """
        if self.status != ProviderStatus.READY:
            raise RuntimeError("Provider not initialized")

        if not text or not text.strip():
            # Return silence for empty text
            silence = b"\x00\x00" * int(self.sample_rate * 0.1)  # 100ms silence
            return SynthesisResult(
                audio_data=silence,
                sample_rate=self.sample_rate,
                text=text,
                voice_id=voice_id,
                timestamp=time.time(),
                duration_ms=100,
                format="int16",
            )

        start_time = time.time()

        try:
            from piper import SynthesisConfig

            # Create synthesis configuration using new API
            length_scale = kwargs.get("speed", self.length_scale)

            # Note: piper1-gpl uses different parameter names
            # length_scale controls speech speed (same as before)
            # But noise_scale and noise_w are not in SynthesisConfig
            # SynthesisConfig supports: volume, length_scale, normalize_audio
            config = SynthesisConfig(
                length_scale=length_scale,
                volume=kwargs.get("volume", 1.0),
                normalize_audio=kwargs.get("normalize_audio", True),
            )

            # Synthesize audio using new API
            audio_stream = io.BytesIO()
            sample_count = 0

            # Use new synthesize() method that returns audio chunks
            for chunk in self.voice.synthesize(text):
                # Extract audio bytes from chunk
                audio_bytes = chunk.audio_int16_bytes
                audio_stream.write(audio_bytes)
                sample_count += len(audio_bytes) // 2  # int16 = 2 bytes per sample

            audio_data = audio_stream.getvalue()

            # Calculate duration
            duration_ms = (sample_count / self.sample_rate) * 1000
            synthesis_time_ms = (time.time() - start_time) * 1000

            self._record_request(synthesis_time_ms, error=False)

            return SynthesisResult(
                audio_data=audio_data,
                sample_rate=self.sample_rate,
                text=text,
                voice_id=voice_id or self.model_name,
                timestamp=time.time(),
                duration_ms=duration_ms,
                format="int16",
            )

        except Exception as e:
            self._record_request(0, error=True)
            raise RuntimeError(f"Synthesis failed: {e}") from e

    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str],
        voice_id: str,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[SynthesisResult]:
        """
        Synthesize speech from text stream

        Args:
            text_stream: Async iterator of texts
            voice_id: Voice identifier
            language: Optional language hint
            **kwargs: Additional parameters

        Yields:
            Synthesis results as they become available
        """
        async for text in text_stream:
            result = await self.synthesize(text, voice_id, language, **kwargs)
            yield result

    async def get_voices(self) -> list[Dict[str, Any]]:
        """
        Get available voices

        Returns:
            List of voice dictionaries
        """
        voices = []
        for lang, models in self._available_voices.items():
            for model in models:
                voices.append({
                    "id": model,
                    "name": model.replace("_", " ").title(),
                    "language": lang,
                    "gender": self._infer_gender(model),
                    "quality": "medium",
                })
        return voices

    def _get_model_path(self, model_name: str) -> Path:
        """
        Get path to model file

        Args:
            model_name: Model name

        Returns:
            Path to model file
        """
        # Check common model directories for piper1-gpl
        model_dirs = [
            # Project root (where download_voices downloads by default)
            Path.cwd(),
            # Project-local models directory
            Path.cwd() / "models" / "piper",
            # User home locations
            Path.home() / ".local" / "share" / "piper",
            # Legacy piper locations
            Path.home() / ".cache" / "piper" / "models",
            Path.home() / ".local" / "share" / "piper" / "models",
            Path("/usr/share/piper/models"),
        ]

        for model_dir in model_dirs:
            # Try direct path (piper1-gpl format)
            model_path = model_dir / f"{model_name}.onnx"
            if model_path.exists():
                return model_path

            # Try with subdirectory (some installations organize by language)
            # e.g., ~/.local/share/piper/en_US-lessac-medium/en_US-lessac-medium.onnx
            model_subdir_path = model_dir / model_name / f"{model_name}.onnx"
            if model_subdir_path.exists():
                return model_subdir_path

        # Return default path for download (project-local models directory)
        return Path.cwd() / "models" / "piper" / f"{model_name}.onnx"

    def _infer_gender(self, model_name: str) -> str:
        """Infer gender from model name"""
        name_lower = model_name.lower()
        female_names = ["amy", "jenny", "lessac", "siwis", "huayan", "natsu"]
        male_names = ["alan", "joe", "thorsten"]

        for name in female_names:
            if name in name_lower:
                return "female"
        for name in male_names:
            if name in name_lower:
                return "male"
        return "neutral"

    def supports_streaming(self) -> bool:
        """Piper supports streaming synthesis"""
        return True

    def supports_voice_cloning(self) -> bool:
        """Piper does not support voice cloning"""
        return False

    def supported_languages(self) -> list[str]:
        """Get list of supported language codes"""
        return list(self._available_voices.keys())

    async def health_check(self) -> bool:
        """Check if provider is healthy"""
        if self.status != ProviderStatus.READY:
            return False

        if self.voice is None:
            return False

        # Test synthesis with simple text
        try:
            result = await self.synthesize("Hello", "default")
            return len(result.audio_data) > 0
        except Exception:
            return False

    def __repr__(self) -> str:
        return (
            f"PiperProvider(model={self.model_name}, "
            f"sample_rate={self.sample_rate}Hz, "
            f"status={self.status.value})"
        )


# Register provider
from providers.base import register_provider
register_provider("tts", "piper", PiperProvider)
