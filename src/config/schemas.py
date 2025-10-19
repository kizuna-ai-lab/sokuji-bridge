"""
Configuration Schemas for Sokuji-Bridge

Pydantic models for type-safe configuration management.
"""

from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator
from pathlib import Path


class VADConfig(BaseModel):
    """Voice Activity Detection configuration"""
    enabled: bool = True
    model: Literal["silero", "webrtc"] = "silero"
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_speech_duration_ms: int = Field(default=250, ge=0)
    max_speech_duration_s: int = Field(default=30, ge=1)
    min_silence_duration_ms: int = Field(default=300, ge=0)


class AudioConfig(BaseModel):
    """Audio processing configuration"""
    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    chunk_duration_ms: int = Field(default=30, ge=10, le=1000)
    format: Literal["int16", "float32"] = "int16"
    channels: int = Field(default=1, ge=1, le=2)


class SegmentationConfig(BaseModel):
    """Audio segmentation configuration"""
    strategy: Literal["sentence", "pause", "hybrid"] = "hybrid"
    max_segment_length: int = Field(default=1000, ge=100)
    accumulate_until_punctuation: bool = True


class PipelineConfig(BaseModel):
    """Main pipeline configuration"""
    name: str = "default"
    mode: Literal["streaming", "batch"] = "streaming"
    source_language: str = "auto"
    target_language: str = "en"

    vad: VADConfig = Field(default_factory=VADConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)

    @field_validator("source_language", "target_language")
    @classmethod
    def validate_language_code(cls, v: str) -> str:
        """Validate language code format"""
        if v != "auto" and len(v) not in [2, 5]:  # "zh" or "zh-CN"
            raise ValueError(f"Invalid language code: {v}")
        return v.lower()


class STTConfig(BaseModel):
    """Speech-to-Text provider configuration"""
    provider: str = "faster_whisper"
    config: Dict[str, Any] = Field(default_factory=dict)

    # Common STT options
    model: Optional[str] = None
    device: Literal["cpu", "cuda", "auto"] = "auto"
    language: Optional[str] = None

    # Performance optimization
    optimization: Optional[Dict[str, Any]] = None


class TranslationConfig(BaseModel):
    """Translation provider configuration"""
    provider: str = "nllb_local"
    fallback: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)

    # Common translation options
    model: Optional[str] = None
    device: Literal["cpu", "cuda", "auto"] = "auto"

    # Batching and caching
    batching: Optional[Dict[str, Any]] = None
    cache: Optional[Dict[str, Any]] = None


class TTSConfig(BaseModel):
    """Text-to-Speech provider configuration"""
    provider: str = "piper"
    config: Dict[str, Any] = Field(default_factory=dict)

    # Common TTS options
    model: Optional[str] = None
    voice: Optional[str] = None
    device: Literal["cpu", "cuda", "auto"] = "auto"

    # Voice cloning (for XTTS, etc.)
    voice_cloning: Optional[Dict[str, Any]] = None


class MonitoringConfig(BaseModel):
    """Monitoring and observability configuration"""
    enabled: bool = True
    prometheus_port: int = Field(default=9090, ge=1024, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Latency targets (ms) for alerting
    latency_targets: Dict[str, int] = Field(
        default_factory=lambda: {
            "stt_ms": 500,
            "translation_ms": 400,
            "tts_ms": 200,
            "total_ms": 2000,
        }
    )


class APIKeysConfig(BaseModel):
    """API keys configuration (loaded from environment)"""
    deepl: Optional[str] = None
    openai: Optional[str] = None
    elevenlabs: Optional[str] = None
    azure_speech_key: Optional[str] = None
    azure_speech_region: Optional[str] = None
    google_application_credentials: Optional[str] = None


class SokujiBridgeConfig(BaseModel):
    """Root configuration for Sokuji-Bridge"""
    version: str = "0.1.0"

    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    api_keys: APIKeysConfig = Field(default_factory=APIKeysConfig)

    # Model cache directory
    model_cache_dir: Path = Field(default=Path.home() / ".cache" / "sokuji-bridge" / "models")

    @field_validator("model_cache_dir")
    @classmethod
    def ensure_cache_dir_exists(cls, v: Path) -> Path:
        """Ensure model cache directory exists"""
        v.mkdir(parents=True, exist_ok=True)
        return v

    def model_dump_yaml(self) -> str:
        """Export configuration as YAML string"""
        import yaml
        return yaml.dump(
            self.model_dump(mode="json", exclude_none=True),
            default_flow_style=False,
            sort_keys=False,
        )


# Predefined profile configurations

def get_fast_profile() -> SokujiBridgeConfig:
    """Fast local profile (default) - lowest latency, local only"""
    return SokujiBridgeConfig(
        pipeline=PipelineConfig(
            name="fast_local",
            source_language="auto",
            target_language="en",
        ),
        stt=STTConfig(
            provider="faster_whisper",
            config={
                "model_size": "medium",
                "compute_type": "float16",
                "num_workers": 2,
                "language": None,
                "initial_prompt": None,
                "beam_size": 5,
                "best_of": 5,
                "temperature": 0.0,
                "condition_on_previous_text": False,  # Disable context to reduce hallucinations
                "vad_filter": True,        # Enable Whisper's built-in VAD
                "vad_threshold": 0.95,     # Very strict threshold to prevent audio hallucinations
            },
            device="cuda",
        ),
        translation=TranslationConfig(
            provider="nllb_local",
            config={
                "model": "facebook/nllb-200-distilled-1.3B",
                "precision": "float16",
                "num_beams": 4,
            },
            device="cuda",
            batching={
                "enabled": True,
                "max_batch_size": 4,
                "timeout_ms": 500,
            },
            cache={
                "enabled": True,
                "ttl_seconds": 3600,
                "max_entries": 10000,
            },
        ),
        tts=TTSConfig(
            provider="piper",
            config={
                "model": "en_US-lessac-medium",
                "speaker_id": 0,
                "length_scale": 1.0,
            },
            device="cpu",
        ),
    )


def get_hybrid_profile() -> SokujiBridgeConfig:
    """Hybrid profile - local STT/TTS, API translation"""
    return SokujiBridgeConfig(
        pipeline=PipelineConfig(
            name="hybrid_quality",
            source_language="auto",
            target_language="en",
        ),
        stt=STTConfig(
            provider="faster_whisper",
            config={
                "model_size": "medium",
                "compute_type": "float16",
            },
            device="cuda",
        ),
        translation=TranslationConfig(
            provider="deepl_api",
            fallback="nllb_local",
            config={
                "formality": "default",
                "preserve_formatting": True,
            },
            cache={
                "enabled": True,
                "ttl_seconds": 7200,
            },
        ),
        tts=TTSConfig(
            provider="piper",
            config={
                "model": "en_US-lessac-medium",
            },
            device="cpu",
        ),
    )


def get_quality_profile() -> SokujiBridgeConfig:
    """Quality profile - best quality with API usage"""
    return SokujiBridgeConfig(
        pipeline=PipelineConfig(
            name="maximum_quality",
            source_language="auto",
            target_language="en",
        ),
        stt=STTConfig(
            provider="faster_whisper",
            config={
                "model_size": "large-v3",
                "compute_type": "float16",
                "beam_size": 5,
                "best_of": 5,
            },
            device="cuda",
        ),
        translation=TranslationConfig(
            provider="openai_translator",
            fallback="deepl_api",
            config={
                "model": "gpt-4o-mini",
                "temperature": 0.3,
                "context_aware": True,
            },
        ),
        tts=TTSConfig(
            provider="xtts",
            config={
                "model": "tts_models/multilingual/multi-dataset/xtts_v2",
            },
            device="cuda",
            voice_cloning={
                "enabled": False,
                "reference_audio": None,
            },
        ),
    )


def get_cpu_only_profile() -> SokujiBridgeConfig:
    """CPU-only profile - fallback for systems without GPU"""
    return SokujiBridgeConfig(
        pipeline=PipelineConfig(
            name="cpu_fallback",
        ),
        stt=STTConfig(
            provider="faster_whisper",
            config={
                "model_size": "base",
                "compute_type": "int8",
            },
            device="cpu",
        ),
        translation=TranslationConfig(
            provider="google_translate",  # API fallback
            config={},
        ),
        tts=TTSConfig(
            provider="piper",
            device="cpu",
        ),
    )


# Profile registry
PROFILES: Dict[str, callable] = {
    "fast": get_fast_profile,
    "hybrid": get_hybrid_profile,
    "quality": get_quality_profile,
    "cpu": get_cpu_only_profile,
}


def get_profile(name: str) -> SokujiBridgeConfig:
    """Get predefined configuration profile"""
    if name not in PROFILES:
        raise ValueError(f"Unknown profile: {name}. Available: {list(PROFILES.keys())}")
    return PROFILES[name]()
