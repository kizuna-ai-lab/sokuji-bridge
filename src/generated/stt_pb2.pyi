import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TranscribeRequest(_message.Message):
    __slots__ = ["audio", "language"]
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    audio: _common_pb2.AudioChunk
    language: str
    def __init__(self, audio: _Optional[_Union[_common_pb2.AudioChunk, _Mapping]] = ..., language: _Optional[str] = ...) -> None: ...

class LanguageListResponse(_message.Message):
    __slots__ = ["language_codes"]
    LANGUAGE_CODES_FIELD_NUMBER: _ClassVar[int]
    language_codes: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, language_codes: _Optional[_Iterable[str]] = ...) -> None: ...
