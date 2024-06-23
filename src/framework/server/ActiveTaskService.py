import gc
import threading

import torch

import framework.common.logger_util as logger_util
from load.LoadConfigs import load_llm_configs
from load.LoadParty import get_class_constructor
from load.QwenModelLoader import QwenModelLoader
from utils import recorder

logger = logger_util.get_logger('active_task_service')


class ActiveTaskService(threading.Thread):
    _queues = {}
    _model_data = {}
    _active_parties = {}
    _last_result = None

    def __init__(self, queues):
        threading.Thread.__init__(self)
        self._queues = queues

    def _get_active_party(self, job_id, config):
        key = str(job_id)
        if key in self._active_parties:
            return self._active_parties[key]
        elif config is not None:
            args = load_llm_configs(config)
            need_model = args.model_type.lower() != 'qwen2' or args.pipeline != 'pretrained'
            active_party = get_class_constructor(args.active_party_class)(args, args.k - 1, need_model=need_model)
            self._active_parties[key] = active_party
            return active_party
        return None

    def run(self):
        while True:
            task = self._queues['active'].get()
            party = self._get_active_party(task.job_id)
            logger.info(f"Running task: {task.run}")
            logger.info(f"Party: {party}")
            if hasattr(party, task.run):
                target_func = getattr(party, task.run)
                result = target_func(last_task_result=self._last_result)
                self._last_result = result
                logger.info(f"Finished task: {task.run}")

    def run_specific(self, task, config, data=None):
        party = self._get_active_party(task['job_id'], config)
        if party.global_model is None:
            party.update_model_data(self._model_data)
        logger.info(f"running specific task: {task['run']}")
        logger.debug(f"Party: {party}")
        if hasattr(party, task['run']):
            target_func = getattr(party, task['run'])
            current_grad_enabled = None
            if task['params'] and 'grad_enabled' in task['params']:
                logger.info(f"Setting grad_enabled: {task['params']['grad_enabled']}")
                current_grad_enabled = torch.is_grad_enabled()
                torch.set_grad_enabled(task['params']['grad_enabled'])

            if data is not None:
                result = target_func(data)
            else:
                result = target_func()
            logger.info(f"Finished specific task: {task['run']}")
            logger.info(str(recorder))
            if current_grad_enabled is not None:
                torch.set_grad_enabled(current_grad_enabled)
            return result

    def close_job(self, job_id):
        logger.info(f"Closing job: {job_id}")
        key = str(job_id)
        if key in self._active_parties:
            del self._active_parties[key]
            gc.collect()
            logger.info(f"current parties: {self._active_parties}")

    def load_model(self, model_type, model_id):
        logger.info(f"Loading model: {model_type} {model_id}")
        model_path = f'/shared/model/Qwen/{model_id}'  # TODO: need to change path
        # model_path = f'/home/shannon/dev/tools/nlp/models/{model_id}'
        if model_type.lower() == 'qwen2':
            loader = QwenModelLoader()  # TODO: use interface instead
            self._model_data = loader.load(model_path, True)
        else:
            raise NotImplementedError

    def update_model_data(self, job_id):
        party = self._get_active_party(job_id, None)
        party.update_model_data(self._model_data)
