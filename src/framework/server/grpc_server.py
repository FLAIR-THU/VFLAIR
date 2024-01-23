import os

import framework.protos.service_pb2_grpc as fps
import framework.protos.message_pb2 as fpm
import framework.protos.node_pb2 as fpn
import grpc
from concurrent import futures
from threading import Thread
import framework.common.MessageQueue as mq
import framework.common.logger_util as logger_util
import yaml
import argparse
import json
import queue

from framework.common import MessageUtil as mu
import framework.server.MessageService as fsm
from framework.database.repository.JobRepository import job_repository

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

logger = logger_util.get_logger("grpc_server")
class GrpcServer(fps.MessageServiceServicer):
    _clients = set()
    _queues = {}
    _node = fpn.Node(node_id="grpc server")
    _message_service = None

    def __init__(self):
        self._queues['active'] = queue.Queue(0)

    def register(self, request, context):
        node_id = request.node.node_id
        if not self._clients.__contains__(node_id):
            self._clients.add(node_id)
            the_queue = queue.Queue(0)
            self._queues[node_id] = the_queue
            response = {}
            the_queue.put(response)
        else:
            logger.warning("Node {} already registered".format(node_id))
            return

        try:
            while self._clients.__contains__(node_id):
                the_queue = self._queues[node_id]
                task = the_queue.get(True)
                logger.info("sending message: {}".format(task))
                value = fpm.Value()
                config = fpm.Value()
                msg_type = 0
                if not isinstance(task, dict):
                    value.string = json.dumps(task.to_dict())
                    job = job_repository.get_by_id(task.job_id)
                    config.string = job.params
                    msg_type = 1
                yield mu.MessageUtil.create(self._node, {"task": value, "config": config}, msg_type)
        except GeneratorExit as e:
            logger.error("generator exception: {}".format(e))
        finally:
            logger.info("finished register")

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
                logger.info(
                    "code(%s), message: %s"
                    % (request.code, request.data)
                )
                mq.message_queue.put(request)

        t = Thread(target=parse_request)
        t.start()

        while True:
            msg = mq.message_queue.get()
            logger.info(msg)
            yield msg


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

    MAX_MESSAGE_LENGTH = 100 * 1024 * 1000
    server = grpc.server(futures.ThreadPoolExecutor(),  options=[
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
    ])
    fps.add_MessageServiceServicer_to_server(GrpcServer(), server)
    server.add_insecure_port("{}:{}".format(host, port))

    server.start()
    logger.info("--------GRPC server started-----")
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("GrpcServer")
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', default='3333', type=int)
    parser.add_argument('--config', default='./server_config.yml')
    args = parser.parse_args()
    main(args)
