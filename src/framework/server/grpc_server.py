import os

import framework.protos.service_pb2_grpc as fps
import framework.protos.message_pb2 as fpm
import framework.protos.node_pb2 as fpn
import grpc
from concurrent import futures
from threading import Thread
import framework.common.MessageQueue as mq
import framework.common.logger_util as logger_util
from framework.common.yaml_loader import load_yaml
import argparse
import json
import queue
from framework.common import MessageUtil as mu
import framework.server.ActiveMessageService as fsm
import framework.credentials.credentials as credentials
from framework.client.DistributedCommunication import merge_data

logger = logger_util.get_logger("grpc_server")

ACTIVE_PARTY = 'active'


class GrpcServer(fps.MessageServiceServicer):
    _clients = set()
    _queues = {}
    _node = fpn.Node(node_id="grpc server")
    _message_service = None

    def __init__(self):
        self._queues[ACTIVE_PARTY] = queue.Queue(0)

    def register(self, request, context):
        node_id = request.node.node_id
        if not self._clients.__contains__(node_id):
            self._clients.add(node_id)
            the_queue = queue.Queue(0)
            self._queues[node_id] = the_queue
            response = {}
            the_queue.put(response)
        else:
            msg = "Node {} already registered".format(node_id)
            logger.warning(msg)
            yield mu.MessageUtil.error(self._node, msg)

        try:
            while self._clients.__contains__(node_id):
                the_queue = self._queues[node_id]
                task = the_queue.get(True)
                logger.info("sending message: {}".format(task))
                value = fpm.Value()
                config = fpm.Value()
                msg_type = fpm.PLAIN
                if not isinstance(task, dict):
                    value.string = json.dumps(task.to_dict())
                    # job = job_repository.get_by_id(task.job_id)
                    # config.string = job.params
                    msg_type = fpm.START_TASK
                yield mu.MessageUtil.create(self._node, {"task": value, "config": config}, msg_type)
        except GeneratorExit as e:
            logger.error("generator exception: {}".format(e))
        finally:
            logger.info("stream closed for {}".format(node_id))
            self._clients.remove(node_id)
            self._queues.pop(node_id)

    def unregister(self, request, context):
        node_id = request.node.node_id
        self._clients.remove(node_id)
        return mu.MessageUtil.create(self._node, {}, fpm.PLAIN)

    def send(self, request, context):
        if self._message_service is None:
            self._message_service = fsm.MessageService(self._queues)

        try:
            result = self._message_service.parse_message(request)
            if result is None:
                response = mu.MessageUtil.create(self._node, {})
                return response
            response = mu.MessageUtil.create(self._node, result)
            return response
        except Exception as e:
            logger.exception(f"Exception occurred: {e}")
            response = mu.MessageUtil.error(self._node, e.__str__())
            return response

    def send_stream(self, request_iterator, context):
        # 开启一个子线程去接收数据
        # Open a sub thread to receive data
        def parse_request():
            for request in request_iterator:
                mq.message_queue.put(request)

        t = Thread(target=parse_request)
        t.start()

        while True:
            msg = mq.message_queue.get()
            logger.info(msg)
            yield msg

    def send_server_stream(self, request, context):
        node_id = request.node.node_id
        self._message_service.parse_message(request)
        while True:
            msg = mq.message_queue.get()
            logger.info(msg)
            if msg.type == fpm.STREAM_END:
                return
            yield msg

    def send_batch(self, request_iterator, context):
        data_segments = []
        data = {}
        for i, request in enumerate(request_iterator):
            if i == 0:
                data = {**request.data.named_values}
            data_segments.append(request.data.named_values['data'])
        data['data'] = merge_data(data_segments)
        msg = mu.MessageUtil.create(self._node, data, fpm.START_TASK)

        result = self._message_service.parse_message(msg)
        if result is None:
            response = mu.MessageUtil.create(self._node, {})
        else:
            response = mu.MessageUtil.create(self._node, result)
        yield response


def main(main_args):
    if main_args.config is not None:
        config = load_yaml(main_args.config)
        logger.info("config: %s", config)
        host = config["server"]["host"]
        port = config["server"]["port"]
        compression = config["server"]["compression"]
    else:
        raise ValueError("Please specify --config")

    MAX_MESSAGE_LENGTH = 2000 * 1024 * 1000
    server_credentials = grpc.ssl_server_credentials(
        (
            (
                credentials.SERVER_CERTIFICATE_KEY,
                credentials.SERVER_CERTIFICATE,
            ),
        )
    )
    compression_algorithm = grpc.Compression.NoCompression
    if compression == 'GZIP':
        compression_algorithm = grpc.Compression.Gzip
    server = grpc.server(futures.ThreadPoolExecutor(), options=[
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
    ], compression=compression_algorithm)
    grpc_server = GrpcServer()
    fps.add_MessageServiceServicer_to_server(grpc_server, server)
    server.add_secure_port("{}:{}".format(host, port), server_credentials)

    server.start()
    logger.info("--------GRPC server started-----")
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("GrpcServer")
    parser.add_argument('--config', default='./server_config.yml')
    args = parser.parse_args()
    main(args)
