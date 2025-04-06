from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class CleanupRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class CleanupResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class InstanceRequest(_message.Message):
    __slots__ = ("name", "payload")
    NAME_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    name: str
    payload: bytes
    def __init__(self, name: _Optional[str] = ..., payload: _Optional[bytes] = ...) -> None: ...

class InstanceResponse(_message.Message):
    __slots__ = ("uuid",)
    UUID_FIELD_NUMBER: _ClassVar[int]
    uuid: int
    def __init__(self, uuid: _Optional[int] = ...) -> None: ...

class CommandRequest(_message.Message):
    __slots__ = ("uuid", "method", "payload")
    UUID_FIELD_NUMBER: _ClassVar[int]
    METHOD_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    uuid: int
    method: str
    payload: bytes
    def __init__(self, uuid: _Optional[int] = ..., method: _Optional[str] = ..., payload: _Optional[bytes] = ...) -> None: ...

class CommandResponse(_message.Message):
    __slots__ = ("result", "direct")
    RESULT_FIELD_NUMBER: _ClassVar[int]
    DIRECT_FIELD_NUMBER: _ClassVar[int]
    result: bytes
    direct: bool
    def __init__(self, result: _Optional[bytes] = ..., direct: bool = ...) -> None: ...

class ImportRequest(_message.Message):
    __slots__ = ("package", "as_name", "item")
    PACKAGE_FIELD_NUMBER: _ClassVar[int]
    AS_NAME_FIELD_NUMBER: _ClassVar[int]
    ITEM_FIELD_NUMBER: _ClassVar[int]
    package: str
    as_name: str
    item: str
    def __init__(self, package: _Optional[str] = ..., as_name: _Optional[str] = ..., item: _Optional[str] = ...) -> None: ...

class ImportResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
