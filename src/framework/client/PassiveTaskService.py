import framework.common.logger_util as logger_util
from load.LoadParty import get_class_constructor
import framework.protos.message_pb2 as fpm
import framework.protos.node_pb2 as fpn
import framework.common.MessageUtil as mu
from load.LoadConfigs import load_llm_configs
import json
from .DistributedCommunication import DistributedCommunication
from framework.server.RemoteActiveParty import RemoteActiveParty
from evaluates.MainTaskVFL_LLM import MainTaskVFL_LLM
from framework.database.repository.JobRepository import job_repository

logger = logger_util.get_logger('passive_task_service')


class PassiveTaskService:
    _parties = {}
    _client = None
    _main_tasks = {}

    def _get_party(self, task):
        return self._parties[task['job_id']]

    def add_job(self, job_id, data):
        if job_id not in self._parties:
            args = load_llm_configs(data)
            passive_party = get_class_constructor(args.passive_party_class)(args, self._client.index)
            passive_party.init_communication(DistributedCommunication(self._client, job_id))
            self._parties[job_id] = passive_party

    def _init_parties(self, args, job_id):
        passive_party = get_class_constructor(args.passive_party_class)(args, self._client.index)
        passive_party.init_communication(DistributedCommunication(self._client, job_id))
        parties = [passive_party, RemoteActiveParty(self._client, job_id, args)]
        return parties

    def run_job(self, job_id, data, params=None):
        params = params if params else {}
        args = load_llm_configs(data)
        args.parties = self._init_parties(args, job_id)
        # self._main_tasks.append(MainTaskVFL_LLM(args, job_id))
        # self.run_next(job_id)
        main_task = MainTaskVFL_LLM(args, job_id)
        self._main_tasks.setdefault(str(job_id), main_task)
        if args.pipeline == 'pretrained':
            result = main_task.inference(messages=params)
        elif args.pipeline == 'finetune':
            result = main_task.start_train()
        else:
            raise NotImplementedError
        self._save_job_result(job_id, result)
        del self._main_tasks[str(job_id)]

    def _save_job_result(self, job_id, result):
        job_repository.change_status(job_id, 1, result)

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
                result = target_func(**params_value)
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
