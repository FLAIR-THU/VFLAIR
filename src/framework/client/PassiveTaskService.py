import framework.common.logger_util as logger_util
from framework.ml.passive_party import PassiveParty_LLM
import framework.protos.message_pb2 as fpm
import framework.protos.node_pb2 as fpn
import framework.common.MessageUtil as mu
from argparse import Namespace
import json

logger = logger_util.get_logger()


class PassiveTaskService:
    def __init__(self, data, client):
        self._data = data
        self._client = client
        self._node = fpn.Node(node_id=client.id)
        args = Namespace(**self._data)
        self._party = PassiveParty_LLM(args, self._client)

    def run(self, task):
        if hasattr(self._party, task['run']):
            target_func = getattr(self._party, task['run'])
            result = target_func()
            return self._send_message(task["id"], result)

    def _send_message(self, task_id, data):
        value = fpm.Value()
        value.string = json.dumps(data)
        id_value = fpm.Value()
        id_value.sint64 = task_id
        msg = mu.MessageUtil.create(self._node, {"task_id": id_value, "result": value}, 3)
        response = self._client.open_and_send(msg)
        return response
