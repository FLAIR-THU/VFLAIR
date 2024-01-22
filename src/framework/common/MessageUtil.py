import framework.protos.message_pb2 as fpm


class MessageUtil:

    @staticmethod
    def create(node, data: dict, msg_type=0):
        value = fpm.AggregationValue(named_values=data)
        # value.named_values
        message = fpm.Message(
            node=node,
            code=fpm.Code.OK,
            message="",
            data=value,
            type=msg_type
        )
        return message

