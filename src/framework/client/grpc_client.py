import framework.protos.service_pb2_grpc as fps
import framework.protos.node_pb2 as fpn
import framework.protos.message_pb2 as fpm
import framework.client.PassiveMessageService as fcp
import grpc
from framework.common import MessageUtil as mu
import framework.common.logger_util as logger_util
import argparse
import json
from framework.common.yaml_loader import load_yaml
from framework.database.repository.JobRepository import job_repository
from framework.database.model.Task import Task
import framework.credentials.credentials as credentials

logger = logger_util.get_logger("grpc_client")

channel_credential = grpc.ssl_channel_credentials(
    credentials.ROOT_CERTIFICATE
)

MAX_MESSAGE_LENGTH = 2000 * 1024 * 1000


class GrpcClient():
    _message_service = None
    _node = None

    def __init__(self, client_id, client_index, host, port, compression=None):
        self.id = client_id
        self.index = client_index
        self.host = host
        self.port = port
        self._node = fpn.Node(node_id=f"{self.id}")
        self._message_service = fcp.PassiveMessageService(self)

        compression_algorithm = grpc.Compression.NoCompression
        if compression == 'GZIP':
            compression_algorithm = grpc.Compression.Gzip
        self.compression_algorithm = compression_algorithm

        options = [
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ]
        self.channel = grpc.secure_channel(f"{self.host}:{self.port}", channel_credential, options, compression=self.compression_algorithm)
        self.stub = fps.MessageServiceStub(self.channel)

    def send_server_stream(self, task, hidden_states=None):
        msg = self._create_msg(task, hidden_states)
        response_iterator = self.stub.send_server_stream(msg)
        for response in response_iterator:
            print(
                "code(%s), message: %s"
                % (response.code, response.data)
            )
            self._message_service.parse_message(response)
        return response_iterator

    def send_batch(self, task, messages):
        def request_messages():
            for msg in messages:
                yield self._create_msg(task, msg)

        response_iterator = self.stub.send_batch(request_messages())
        return next(response_iterator).data

    def send_stream(self, messages):
        def request_messages():
            for msg in messages:
                yield msg

        response_iterator = self.stub.send_stream(request_messages())
        for response in response_iterator:
            print(
                "code(%s), message: %s"
                % (response.code, response.data)
            )
        return response_iterator

    def _create_msg(self, task, hidden_states=None):
        job = job_repository.get_by_id(task.job_id)
        config_value = fpm.Value()
        config_value.string = job.params

        task_value = fpm.Value()
        task_value.string = json.dumps(task.to_dict())

        data_value = fpm.Value()
        if hidden_states:
            data_value = hidden_states
        msg = mu.MessageUtil.create(self._node, {"config": config_value, "task": task_value, "data": data_value},
                                    fpm.START_TASK)
        return msg

    def send(self, stub, task, hidden_states=None):
        msg = self._create_msg(task, hidden_states)

        response = stub.send(msg)
        return response.data

    def parse_message(self, response):
        return self._message_service.parse_message(response)

    def register(self, stub):
        msg = mu.MessageUtil.create(self._node, {})
        response_iterator = stub.register(msg)
        for response in response_iterator:
            try:
                if response.code is fpm.ERROR:
                    logger.error("received msg from server, code(%s), message: %s, please try again." % (
                    response.code, response.message))
                    self.unregister(stub)
                    return
                self._message_service.parse_message(response)
            except (grpc.RpcError, Exception) as e:
                logger.exception(e)

    def unregister(self, stub):
        msg = mu.MessageUtil.create(self._node, {}, fpm.UNREGISTER)
        stub.unregister(msg)

    def open_and_send(self, task, hidden_states=None):
        if isinstance(task, Task):
            return self.send(self.stub, task, hidden_states=hidden_states)
        else:
            response = self.stub.send(task)
            return response.data

    def open_and_register(self):
        self.register(self.stub)

    def close(self):
        logger.info("Closing channel")
        self.channel.close()


def main(main_args):
    if main_args.config is not None:
        config = load_yaml(main_args.config)
        host = config["server"]["host"]
        port = config["server"]["port"]
        client_id = config["client"]["id"]
        client_index = config["client"]["index"]
    else:
        raise ValueError("Please specify --config")

    with grpc.secure_channel(f"{host}:{port}", channel_credential, options=[
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
    ]) as channel:
        stub = fps.MessageServiceStub(channel)
        GrpcClient(client_id, int(client_index), host, port).register(stub)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("GrpcClient")
    parser.add_argument('--config', default='./client_config.yml')
    args = parser.parse_args()
    main(args)
