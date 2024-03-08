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
            params = task['params']
            if params is not None:
                params_value = json.loads(params)
                result = target_func(params_value)
            else:
                result = target_func()
            return self._send_message(task["job_id"], result, task["run"])

    def _send_message(self, job_id, data, run):
        value = fpm.Value()
        value.string = json.dumps(data)
        run_value = fpm.Value()
        run_value.string = run
        id_value = fpm.Value()
        id_value.sint64 = job_id
        logger.info("sending message: {}".format(data))
        msg = mu.MessageUtil.create(self._node, {"job_id": id_value, "result": value, "run": run_value}, fpm.FINISH_TASK)
        response = self._client.open_and_send(msg)
        return response
