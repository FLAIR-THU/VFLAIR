import gc
import framework.common.logger_util as logger_util
from load.LoadParty import get_class_constructor
import framework.protos.message_pb2 as fpm
import framework.protos.node_pb2 as fpn
import framework.common.MessageUtil as mu
from load.LoadConfigs import load_llm_configs
import json
from .DistributedCommunication import DistributedCommunication
from framework.server.RemoteActiveParty import RemoteActiveParty
from evaluates.MainTaskVFL_LLM import *
from framework.database.repository.JobRepository import job_repository
from framework.database.repository.PretrainedModelRepository import pretrained_model_repository
from framework.database.model.PretrainedModel import PretrainedModel
from main_pipeline_llm import get_cls_ancestor, create_exp_dir_and_file
from load.QwenModelLoader import QwenModelLoader

from utils import timer, recorder

# logger = logger_util.get_logger('passive_task_service')
from loguru import logger
from datetime import datetime


class PassiveTaskService:
    _parties = {}
    _model_data = None
    _client = None

    def _get_party(self, task):
        return self._parties[task['job_id']]

    def add_job(self, job_id, data):
        if job_id not in self._parties:
            args = load_llm_configs(data)
            passive_party = get_class_constructor(args.passive_party_class)(args, self._client.index, need_model=False)
            self._parties[job_id] = passive_party

    def _init_parties(self, args, job_id, need_model):
        passive_party = get_class_constructor(args.passive_party_class)(args, self._client.index, need_model=need_model)
        parties = [passive_party, RemoteActiveParty(self._client, job_id, args)]
        return parties

    @timer()
    def run_job(self, job_id, data, params=None):
        try:
            self._run_job(job_id, data, params)
        except Exception as e:
            job_repository.change_status(job_id, -1, repr(e))
            gc.collect()
            self._close_job(job_id)
        finally:
            logger.info(str(recorder))

    def _close_job(self, job_id):
        logger.info("sending closing message...")

        id_value = fpm.Value()
        id_value.sint64 = job_id
        msg = mu.MessageUtil.create(self._node, {"job_id": id_value}, fpm.CLOSE_JOB)
        self._client.open_and_send(msg)

    @logger.catch(reraise=True)
    def _run_job(self, job_id, data, params=None):
        params = params if params else {}
        args = load_llm_configs(data)
        need_model = args.model_type.lower() != 'qwen2' or args.pipeline != 'pretrained'
        args.parties = self._init_parties(args, job_id, need_model)
        if not need_model:
            args.generation_config = self._model_data['generation_config']
            args.config = self._model_data['config']
        model_name = args.model_list[str(0)]["type"]  # .replace('/','-')
        exp_res_dir, exp_res_path = create_exp_dir_and_file(args.dataset, args.Q, model_name, args.pipeline, args.defense_name, args.defense_param)
        args.exp_res_dir = exp_res_dir
        args.exp_res_path = exp_res_path

        # self._main_tasks.append(MainTaskVFL_LLM(args, job_id))
        # self.run_next(job_id)

        ancestor_cls = get_cls_ancestor(args.config.model_type)
        main_task_creator = create_main_task(ancestor_cls)
        main_task = main_task_creator(args, job_id)
        main_task.init_communication(DistributedCommunication(self._client, job_id))
        if args.pipeline == 'pretrained':
            if not need_model:
                for party in args.parties:
                    party.update_model_data(self._model_data)
            result = main_task.inference(messages=params)
        elif args.pipeline == 'finetune':
            model_id = main_task.create_model_id()
            base_model_id = main_task.get_base_model()
            logger.info('Model id: {}'.format(model_id))
            logger.info('Base model id: {}'.format(base_model_id))
            result = main_task.train_vfl(model_id=model_id, save_model=False)
            model_path = args.parties[0].get_model_folder() + model_id
            self._save_trained_model(base_model_id, model_id, args.model_type, model_path)
        else:
            raise NotImplementedError
        self._save_job_result(job_id, result)

        self._close_job(job_id)

    def _save_job_result(self, job_id, result):
        job_repository.change_status(job_id, 1, result)

    def _save_trained_model(self, base_model_id, model_id, model_type, model_path):
        trained_model = PretrainedModel()
        trained_model.model_id = model_id
        trained_model.model_type = model_type
        trained_model.base_model_id = base_model_id
        trained_model.path = model_path
        trained_model.name = model_id
        trained_model.status = 1
        trained_model.create_time = datetime.now()
        pretrained_model_repository.create(trained_model)

    def __init__(self, client):
        self._client = client
        self._node = fpn.Node(node_id=client.id)

    def load_model(self, model_id):
        model = pretrained_model_repository.get_by_model_id(model_id)
        self._send_load_model_message(model.model_type, model_id)
        if model.model_type.lower() == 'qwen2':
            loader = QwenModelLoader()  # TODO: use interface instead
            self._model_data = loader.load(model.path, False)
        else:
            raise NotImplementedError

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
        msg = mu.MessageUtil.create(self._node, {"job_id": id_value, "result": value, "run": run_value},
                                    fpm.FINISH_TASK)
        response = self._client.open_and_send(msg)
        return response

    def _send_load_model_message(self, model_type, model_id):
        value = fpm.Value()
        value.string = model_id

        type_value = fpm.Value()
        type_value.string = model_type

        logger.info("sending load model message, model_id: {}".format(model_id))
        msg = mu.MessageUtil.create(self._node, {"model_id": value, "model_type": type_value}, fpm.LOAD_MODEL)
        self._client.open_and_send(msg)
