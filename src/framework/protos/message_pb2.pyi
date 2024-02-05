from framework.protos import node_pb2 as _node_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Code(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    OK: _ClassVar[Code]
    ERROR: _ClassVar[Code]

class MessageType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PLAIN: _ClassVar[MessageType]
    CREATE_JOB: _ClassVar[MessageType]
    QUERY_JOB: _ClassVar[MessageType]
    FINISH_TASK: _ClassVar[MessageType]
    START_TASK: _ClassVar[MessageType]
    UNREGISTER: _ClassVar[MessageType]
OK: Code
ERROR: Code
PLAIN: MessageType
CREATE_JOB: MessageType
QUERY_JOB: MessageType
FINISH_TASK: MessageType
START_TASK: MessageType
UNREGISTER: MessageType

class Message(_message.Message):
    __slots__ = ("node", "code", "message", "data", "type")
    NODE_FIELD_NUMBER: _ClassVar[int]
    CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    node: _node_pb2.Node
    code: Code
    message: str
    data: AggregationValue
    type: MessageType
    def __init__(self, node: _Optional[_Union[_node_pb2.Node, _Mapping]] = ..., code: _Optional[_Union[Code, str]] = ..., message: _Optional[str] = ..., data: _Optional[_Union[AggregationValue, _Mapping]] = ..., type: _Optional[_Union[MessageType, str]] = ...) -> None: ...

class Value(_message.Message):
    __slots__ = ("double", "sint64", "bool", "string", "bytes")
    DOUBLE_FIELD_NUMBER: _ClassVar[int]
    SINT64_FIELD_NUMBER: _ClassVar[int]
    BOOL_FIELD_NUMBER: _ClassVar[int]
    STRING_FIELD_NUMBER: _ClassVar[int]
    BYTES_FIELD_NUMBER: _ClassVar[int]
    double: float
    sint64: int
    bool: bool
    string: str
    bytes: bytes
    def __init__(self, double: _Optional[float] = ..., sint64: _Optional[int] = ..., bool: bool = ..., string: _Optional[str] = ..., bytes: _Optional[bytes] = ...) -> None: ...

class AggregationValue(_message.Message):
    __slots__ = ("named_values",)
    class NamedValuesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Value, _Mapping]] = ...) -> None: ...
    NAMED_VALUES_FIELD_NUMBER: _ClassVar[int]
    named_values: _containers.MessageMap[str, Value]
    def __init__(self, named_values: _Optional[_Mapping[str, Value]] = ...) -> None: ...
