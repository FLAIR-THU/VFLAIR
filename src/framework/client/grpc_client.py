import framework.protos.service_pb2_grpc as fps
import framework.protos.message_pb2 as fpm
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

    def __init__(self, client_id, host, port):
        self.id = client_id
        self.host = host
        self.port = port

    def send_stream(self, stub):
        def request_messages():
            value = fpm.Value()
            value.sint64 = 123
            msg = mu.MessageUtil.create({"id": value})
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
        msg = mu.MessageUtil.create(fpn.Node(node_id=f"{self.id}"), {})
        response_iterator = stub.register(msg)
        if self._message_service is None:
            self._message_service = fcp.PassiveMessageService(self)
        for response in response_iterator:
            try:
                self._message_service.parse_message(response)
            except (grpc.RpcError, Exception) as e:
                logger.error(e)

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
        logger.info("config: %s", config)
        host = config["server"]["host"]
        port = config["server"]["port"]
    elif main_args.port is not None and main_args.host is not None:
        host = main_args.host
        port = main_args.port
    else:
        raise ValueError("Please specify either --config or --host and --port")

    if main_args.config is not None:
        client_id = config["client"]["id"]
    else:
        client_id = main_args.clientId

    MAX_MESSAGE_LENGTH = 100*1024*1000
    with grpc.insecure_channel(f"{host}:{port}",  options=[
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
    ]) as channel:
        grpc.max_send_message_length = 1024 * 1024
        stub = fps.MessageServiceStub(channel)
        GrpcClient(client_id, host, port).register(stub)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("GrpcClient")
    parser.add_argument('--clientId', default='1')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', default='3333', type=int)
    parser.add_argument('--config', default='./client_config.yml')
    args = parser.parse_args()
    main(args)
