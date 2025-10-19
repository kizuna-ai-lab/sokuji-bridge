"""
Configuration Schemas for Sokuji-Bridge

Pydantic models for type-safe configuration management.
"""

from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator
from pathlib import Path


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
