import framework.common.logger_util as logger_util
from party.passive_party import PassiveParty_LLM
import framework.protos.message_pb2 as fpm
import framework.protos.node_pb2 as fpn
import framework.common.MessageUtil as mu
from load.LoadConfigs import load_llm_configs
import json
from .DistributedCommunication import DistributedCommunication

logger = logger_util.get_logger('passive_task_service')


class PassiveTaskService:
    _parties = {}
    _client = None

    def _get_party(self, task):
        return self._parties[task['job_id']]

    def add_job(self, job_id, data):
        if job_id not in self._parties:
            args = load_llm_configs(data)
            party = PassiveParty_LLM(args, self._client.index)
            party.init_communication(DistributedCommunication(self._client, job_id))
            self._parties[job_id] = party

    def __init__(self, client):
        self._client = client
        self._node = fpn.Node(node_id=client.id)

    def run(self, task):
        party = self._get_party(task)
        if hasattr(party, task['run']):
            target_func = getattr(party, task['run'])
            result = target_func()
            return self._send_message(task["id"], result)

    def _send_message(self, task_id, data):
        value = fpm.Value()
        value.string = json.dumps(data)
        id_value = fpm.Value()
        id_value.sint64 = task_id
        msg = mu.MessageUtil.create(self._node, {"task_id": id_value, "result": value}, fpm.FINISH_TASK)
        response = self._client.open_and_send(msg)
        return response
