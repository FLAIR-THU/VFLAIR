from framework.protos import node_pb2 as _node_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

CREATE_JOB: MessageType
DESCRIPTOR: _descriptor.FileDescriptor
ERROR: Code
FINISH_TASK: MessageType
OK: Code
PLAIN: MessageType
QUERY_JOB: MessageType
START_TASK: MessageType
UNREGISTER: MessageType

class AggregationValue(_message.Message):
    __slots__ = ["named_values"]
    class NamedValuesEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Value, _Mapping]] = ...) -> None: ...
    NAMED_VALUES_FIELD_NUMBER: _ClassVar[int]
    named_values: _containers.MessageMap[str, Value]
    def __init__(self, named_values: _Optional[_Mapping[str, Value]] = ...) -> None: ...

class Message(_message.Message):
    __slots__ = ["code", "data", "message", "node", "type"]
    CODE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    NODE_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    code: Code
    data: AggregationValue
    message: str
    node: _node_pb2.Node
    type: MessageType
    def __init__(self, node: _Optional[_Union[_node_pb2.Node, _Mapping]] = ..., code: _Optional[_Union[Code, str]] = ..., message: _Optional[str] = ..., data: _Optional[_Union[AggregationValue, _Mapping]] = ..., type: _Optional[_Union[MessageType, str]] = ...) -> None: ...

class Value(_message.Message):
    __slots__ = ["bool", "bytes", "double", "sint64", "string"]
    BOOL_FIELD_NUMBER: _ClassVar[int]
    BYTES_FIELD_NUMBER: _ClassVar[int]
    DOUBLE_FIELD_NUMBER: _ClassVar[int]
    SINT64_FIELD_NUMBER: _ClassVar[int]
    STRING_FIELD_NUMBER: _ClassVar[int]
    bool: bool
    bytes: bytes
    double: float
    sint64: int
    string: str
    def __init__(self, double: _Optional[float] = ..., sint64: _Optional[int] = ..., bool: bool = ..., string: _Optional[str] = ..., bytes: _Optional[bytes] = ...) -> None: ...

class Code(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class MessageType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
