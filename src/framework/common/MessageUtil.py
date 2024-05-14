import framework.protos.message_pb2 as fpm


class MessageUtil:

    @staticmethod
    def create(node, data: dict, msg_type=fpm.PLAIN):
        value = fpm.AggregationValue(named_values=data)
        message = fpm.Message(
            node=node,
            code=fpm.Code.OK,
            message="",
            data=value,
            type=msg_type
        )
        return message

    @staticmethod
    def error(node, error, msg_type=fpm.PLAIN):
        value = fpm.AggregationValue(named_values={})
        message = fpm.Message(
            node=node,
            code=fpm.Code.ERROR,
            message=error,
            data=value,
            type=msg_type
        )
        return message
