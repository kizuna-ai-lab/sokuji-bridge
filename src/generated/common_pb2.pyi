from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class AudioChunk(_message.Message):
    __slots__ = ["data", "sample_rate", "timestamp", "channels", "format"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    CHANNELS_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    sample_rate: int
    timestamp: float
    channels: int
    format: str
    def __init__(self, data: _Optional[bytes] = ..., sample_rate: _Optional[int] = ..., timestamp: _Optional[float] = ..., channels: _Optional[int] = ..., format: _Optional[str] = ...) -> None: ...

class TranscriptionResult(_message.Message):
    __slots__ = ["text", "language", "confidence", "timestamp", "start_time", "end_time", "is_final"]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    END_TIME_FIELD_NUMBER: _ClassVar[int]
    IS_FINAL_FIELD_NUMBER: _ClassVar[int]
    text: str
    language: str
    confidence: float
    timestamp: float
    start_time: float
    end_time: float
    is_final: bool
    def __init__(self, text: _Optional[str] = ..., language: _Optional[str] = ..., confidence: _Optional[float] = ..., timestamp: _Optional[float] = ..., start_time: _Optional[float] = ..., end_time: _Optional[float] = ..., is_final: bool = ...) -> None: ...

class TranslationResult(_message.Message):
    __slots__ = ["original_text", "translated_text", "source_language", "target_language", "confidence", "timestamp", "model_name"]
    ORIGINAL_TEXT_FIELD_NUMBER: _ClassVar[int]
    TRANSLATED_TEXT_FIELD_NUMBER: _ClassVar[int]
    SOURCE_LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    TARGET_LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: float
    timestamp: float
    model_name: str
    def __init__(self, original_text: _Optional[str] = ..., translated_text: _Optional[str] = ..., source_language: _Optional[str] = ..., target_language: _Optional[str] = ..., confidence: _Optional[float] = ..., timestamp: _Optional[float] = ..., model_name: _Optional[str] = ...) -> None: ...

class SynthesisResult(_message.Message):
    __slots__ = ["audio_data", "sample_rate", "text", "voice_id", "timestamp", "duration_ms", "format"]
    AUDIO_DATA_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    VOICE_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    audio_data: bytes
    sample_rate: int
    text: str
    voice_id: str
    timestamp: float
    duration_ms: float
    format: str
    def __init__(self, audio_data: _Optional[bytes] = ..., sample_rate: _Optional[int] = ..., text: _Optional[str] = ..., voice_id: _Optional[str] = ..., timestamp: _Optional[float] = ..., duration_ms: _Optional[float] = ..., format: _Optional[str] = ...) -> None: ...

class HealthCheckResponse(_message.Message):
    __slots__ = ["healthy", "status", "provider_name", "details"]
    class DetailsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_NAME_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    healthy: bool
    status: str
    provider_name: str
    details: _containers.ScalarMap[str, str]
    def __init__(self, healthy: bool = ..., status: _Optional[str] = ..., provider_name: _Optional[str] = ..., details: _Optional[_Mapping[str, str]] = ...) -> None: ...

class MetricsResponse(_message.Message):
    __slots__ = ["total_requests", "total_errors", "avg_duration_ms", "total_duration_ms", "additional"]
    class AdditionalEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    TOTAL_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_ERRORS_FIELD_NUMBER: _ClassVar[int]
    AVG_DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    ADDITIONAL_FIELD_NUMBER: _ClassVar[int]
    total_requests: int
    total_errors: int
    avg_duration_ms: float
    total_duration_ms: float
    additional: _containers.ScalarMap[str, str]
    def __init__(self, total_requests: _Optional[int] = ..., total_errors: _Optional[int] = ..., avg_duration_ms: _Optional[float] = ..., total_duration_ms: _Optional[float] = ..., additional: _Optional[_Mapping[str, str]] = ...) -> None: ...

class Empty(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...
