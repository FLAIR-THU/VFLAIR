from framework.protos import node_pb2 as _node_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

CLOSE_JOB: MessageType
CREATE_JOB: MessageType
DESCRIPTOR: _descriptor.FileDescriptor
ERROR: Code
FINISH_TASK: MessageType
LOAD_MODEL: MessageType
OK: Code
PLAIN: MessageType
QUERY_JOB: MessageType
START_TASK: MessageType
STREAM_END: MessageType
UNREGISTER: MessageType
UPDATE_MODEL_DATA: MessageType

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

class HiddenStates(_message.Message):
    __slots__ = ["attention_mask", "inputs_embeds", "output_hidden_states", "position_ids", "requires_grads", "use_cache"]
    ATTENTION_MASK_FIELD_NUMBER: _ClassVar[int]
    INPUTS_EMBEDS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_HIDDEN_STATES_FIELD_NUMBER: _ClassVar[int]
    POSITION_IDS_FIELD_NUMBER: _ClassVar[int]
    REQUIRES_GRADS_FIELD_NUMBER: _ClassVar[int]
    USE_CACHE_FIELD_NUMBER: _ClassVar[int]
    attention_mask: tensor_double
    inputs_embeds: tensor_double
    output_hidden_states: bool
    position_ids: tensor_int
    requires_grads: _containers.RepeatedScalarFieldContainer[str]
    use_cache: bool
    def __init__(self, inputs_embeds: _Optional[_Union[tensor_double, _Mapping]] = ..., attention_mask: _Optional[_Union[tensor_double, _Mapping]] = ..., position_ids: _Optional[_Union[tensor_int, _Mapping]] = ..., requires_grads: _Optional[_Iterable[str]] = ..., use_cache: bool = ..., output_hidden_states: bool = ...) -> None: ...

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

class TensorData(_message.Message):
    __slots__ = ["data", "requires_grad"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    REQUIRES_GRAD_FIELD_NUMBER: _ClassVar[int]
    data: tensor_double
    requires_grad: bool
    def __init__(self, data: _Optional[_Union[tensor_double, _Mapping]] = ..., requires_grad: bool = ...) -> None: ...

class Value(_message.Message):
    __slots__ = ["bool", "bytes", "double", "hidden_states", "sint64", "string", "tensor"]
    BOOL_FIELD_NUMBER: _ClassVar[int]
    BYTES_FIELD_NUMBER: _ClassVar[int]
    DOUBLE_FIELD_NUMBER: _ClassVar[int]
    HIDDEN_STATES_FIELD_NUMBER: _ClassVar[int]
    SINT64_FIELD_NUMBER: _ClassVar[int]
    STRING_FIELD_NUMBER: _ClassVar[int]
    TENSOR_FIELD_NUMBER: _ClassVar[int]
    bool: bool
    bytes: bytes
    double: float
    hidden_states: HiddenStates
    sint64: int
    string: str
    tensor: TensorData
    def __init__(self, double: _Optional[float] = ..., sint64: _Optional[int] = ..., bool: bool = ..., string: _Optional[str] = ..., bytes: _Optional[bytes] = ..., hidden_states: _Optional[_Union[HiddenStates, _Mapping]] = ..., tensor: _Optional[_Union[TensorData, _Mapping]] = ...) -> None: ...

class tensor_double(_message.Message):
    __slots__ = ["dtype", "shape", "value"]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    dtype: str
    shape: _containers.RepeatedScalarFieldContainer[int]
    value: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, value: _Optional[_Iterable[float]] = ..., shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ...) -> None: ...

class tensor_int(_message.Message):
    __slots__ = ["shape", "value"]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    shape: _containers.RepeatedScalarFieldContainer[int]
    value: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, value: _Optional[_Iterable[int]] = ..., shape: _Optional[_Iterable[int]] = ...) -> None: ...

class Code(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class MessageType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
