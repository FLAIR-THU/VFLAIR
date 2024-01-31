import framework.protos.service_pb2_grpc as fps
import framework.protos.node_pb2 as fpn
import framework.client.PassiveMessageService as fcp
import grpc
import yaml
from framework.common import MessageUtil as mu
import framework.common.logger_util as logger_util
import argparse
import os
logger = logger_util.get_logger("grpc_client")

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class GrpcClient():
    _message_service = None
    _node = None

    def __init__(self, client_id, host, port):
        self.id = client_id
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

    def send(self, stub, msg):
        response = stub.send(msg)
        logger.info(
            "code(%s), message: %s"
            % (response.code, response.data)
        )
        return response.data


    def register(self, stub):
        msg = mu.MessageUtil.create(self._node, {})
        response_iterator = stub.register(msg)
        if self._message_service is None:
            self._message_service = fcp.PassiveMessageService(self)
        for response in response_iterator:
            try:
                self._message_service.parse_message(response)
            except (grpc.RpcError, Exception) as e:
                logger.exception(e)

    def open_and_send(self, msg):
        with grpc.insecure_channel(f"{self.host}:{self.port}") as channel:
            stub = fps.MessageServiceStub(channel)
            return self.send(stub, msg)

    def open_and_register(self):
        with grpc.insecure_channel(f"{self.host}:{self.port}") as channel:
            stub = fps.MessageServiceStub(channel)
            self.register(stub)

def main(main_args):
    if main_args.config is not None:
        config = yaml.safe_load(open(main_args.config))
        host = config["server"]["host"]
        port = config["server"]["port"]
        client_id = config["client"]["id"]
    else:
        raise ValueError("Please specify --config")

    MAX_MESSAGE_LENGTH = 100*1024*1000
    with grpc.insecure_channel(f"{host}:{port}",  options=[
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
    ]) as channel:
        stub = fps.MessageServiceStub(channel)
        GrpcClient(client_id, host, port).register(stub)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("GrpcClient")
    parser.add_argument('--config', default='./client_config.yml')
    args = parser.parse_args()
    main(args)
