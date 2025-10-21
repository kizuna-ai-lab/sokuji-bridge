import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SynthesizeRequest(_message.Message):
    __slots__ = ["text", "voice_id", "language", "speed", "pitch", "parameters"]
    class ParametersEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    TEXT_FIELD_NUMBER: _ClassVar[int]
    VOICE_ID_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    SPEED_FIELD_NUMBER: _ClassVar[int]
    PITCH_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    text: str
    voice_id: str
    language: str
    speed: float
    pitch: float
    parameters: _containers.ScalarMap[str, str]
    def __init__(self, text: _Optional[str] = ..., voice_id: _Optional[str] = ..., language: _Optional[str] = ..., speed: _Optional[float] = ..., pitch: _Optional[float] = ..., parameters: _Optional[_Mapping[str, str]] = ...) -> None: ...

class Voice(_message.Message):
    __slots__ = ["id", "name", "language", "supported_languages", "gender", "metadata"]
    class MetadataEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_LANGUAGES_FIELD_NUMBER: _ClassVar[int]
    GENDER_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    language: str
    supported_languages: _containers.RepeatedScalarFieldContainer[str]
    gender: str
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., language: _Optional[str] = ..., supported_languages: _Optional[_Iterable[str]] = ..., gender: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class VoiceListResponse(_message.Message):
    __slots__ = ["voices"]
    VOICES_FIELD_NUMBER: _ClassVar[int]
    voices: _containers.RepeatedCompositeFieldContainer[Voice]
    def __init__(self, voices: _Optional[_Iterable[_Union[Voice, _Mapping]]] = ...) -> None: ...

class LanguageListResponse(_message.Message):
    __slots__ = ["language_codes", "supports_streaming", "supports_voice_cloning"]
    LANGUAGE_CODES_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_STREAMING_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_VOICE_CLONING_FIELD_NUMBER: _ClassVar[int]
    language_codes: _containers.RepeatedScalarFieldContainer[str]
    supports_streaming: bool
    supports_voice_cloning: bool
    def __init__(self, language_codes: _Optional[_Iterable[str]] = ..., supports_streaming: bool = ..., supports_voice_cloning: bool = ...) -> None: ...
