import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TranslateRequest(_message.Message):
    __slots__ = ["text", "source_language", "target_language", "context"]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    SOURCE_LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    TARGET_LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    text: str
    source_language: str
    target_language: str
    context: str
    def __init__(self, text: _Optional[str] = ..., source_language: _Optional[str] = ..., target_language: _Optional[str] = ..., context: _Optional[str] = ...) -> None: ...

class TranslateBatchRequest(_message.Message):
    __slots__ = ["texts", "source_language", "target_language"]
    TEXTS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    TARGET_LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    texts: _containers.RepeatedScalarFieldContainer[str]
    source_language: str
    target_language: str
    def __init__(self, texts: _Optional[_Iterable[str]] = ..., source_language: _Optional[str] = ..., target_language: _Optional[str] = ...) -> None: ...

class TranslateBatchResponse(_message.Message):
    __slots__ = ["results"]
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[_common_pb2.TranslationResult]
    def __init__(self, results: _Optional[_Iterable[_Union[_common_pb2.TranslationResult, _Mapping]]] = ...) -> None: ...

class LanguageListResponse(_message.Message):
    __slots__ = ["language_codes", "supports_batch", "supports_streaming"]
    LANGUAGE_CODES_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_BATCH_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_STREAMING_FIELD_NUMBER: _ClassVar[int]
    language_codes: _containers.RepeatedScalarFieldContainer[str]
    supports_batch: bool
    supports_streaming: bool
    def __init__(self, language_codes: _Optional[_Iterable[str]] = ..., supports_batch: bool = ..., supports_streaming: bool = ...) -> None: ...
