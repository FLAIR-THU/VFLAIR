import framework.protos.service_pb2_grpc as fps
import framework.protos.node_pb2 as fpn
import framework.protos.message_pb2 as fpm
import framework.client.PassiveMessageService as fcp
import grpc
from framework.common import MessageUtil as mu
import framework.common.logger_util as logger_util
import argparse
import os
import json
from framework.common.yaml_loader import load_yaml
from framework.database.repository.JobRepository import job_repository
import framework.credentials.credentials as credentials

logger = logger_util.get_logger("grpc_client")

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

channel_credential = grpc.ssl_channel_credentials(
    credentials.ROOT_CERTIFICATE
)

MAX_MESSAGE_LENGTH = 800*1024*1000

class GrpcClient():
    _message_service = None
    _node = None

    def __init__(self, client_id, client_index, host, port):
        self.id = client_id
        self.index = client_index
        self.host = host
        self.port = port
        self._node = fpn.Node(node_id=f"{self.id}")

    def send_stream(self, stub):
        def request_messages():
            msg = mu.MessageUtil.create(self._node, {})
            yield msg

        response_iterator = stub.send_stream(request_messages())
        for response in response_iterator:
            print(
                "code(%s), message: %s"
                % (response.code, response.data)
            )

    def send(self, stub, task):
        job = job_repository.get_by_id(task.job_id)
        config_value = fpm.Value()
        config_value.string = job.params

        task_value = fpm.Value()
        task_value.string = json.dumps(task.to_dict())
        msg = mu.MessageUtil.create(self._node, {"config": config_value, "task": task_value}, fpm.START_TASK)

        response = stub.send(msg)
        return response.data

    def parse_message(self, response):
        if self._message_service is None:
            self._message_service = fcp.PassiveMessageService(self)
        return self._message_service.parse_message(response)

    def register(self, stub):
        msg = mu.MessageUtil.create(self._node, {})
        response_iterator = stub.register(msg)
        if self._message_service is None:
            self._message_service = fcp.PassiveMessageService(self)
        for response in response_iterator:
            try:
                if response.code is fpm.ERROR:
                    logger.error("received msg from server, code(%s), message: %s, please try again." % (response.code, response.message))
                    self.unregister(stub)
                    return
                self._message_service.parse_message(response)
            except (grpc.RpcError, Exception) as e:
                logger.exception(e)

    def unregister(self, stub):
        msg = mu.MessageUtil.create(self._node, {}, fpm.UNREGISTER)
        stub.unregister(msg)

    def open_and_send(self, task):
        options = [
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ]
        with grpc.secure_channel(f"{self.host}:{self.port}", channel_credential, options) as channel:
            stub = fps.MessageServiceStub(channel)
            return self.send(stub, task)

    def open_and_register(self):
        options = [
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ]
        with grpc.secure_channel(f"{self.host}:{self.port}", channel_credential, options) as channel:
            stub = fps.MessageServiceStub(channel)
            self.register(stub)

def main(main_args):
    if main_args.config is not None:
        config = load_yaml(main_args.config)
        host = config["server"]["host"]
        port = config["server"]["port"]
        client_id = config["client"]["id"]
        client_index = config["client"]["index"]
    else:
        raise ValueError("Please specify --config")

    with grpc.secure_channel(f"{host}:{port}",  channel_credential, options=[
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
