import threading
from argparse import Namespace

import framework.common.logger_util as logger_util
from evaluates.MainTaskVFL_LLM import MainTaskVFL_LLM
from framework.client.RemotePassiveParty import RemotePassiveParty
from framework.database.repository.JobRepository import job_repository
from framework.database.repository.TaskRepository import task_repository
from load.LoadConfigs import load_llm_configs
from load.LoadParty import get_class_constructor
from party.active_party import ActiveParty_LLM

logger = logger_util.get_logger('active_task_service')


class ActiveTaskService(threading.Thread):
    _queues = {}
    _active_parties = {}
    _last_result = None

    def __init__(self, queues):
        threading.Thread.__init__(self)
        self._queues = queues

    def _get_active_party(self, job_id, config):
        key = str(job_id)
        if key in self._active_parties:
            return self._active_parties[key]
        else:
            args = load_llm_configs(config)
            active_party = get_class_constructor(args.active_party_class)(args, args.k - 1)
            self._active_parties[key] = active_party
            return active_party

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

    def run_specific(self, task, data, config):
        party = self._get_active_party(task['job_id'], config)
        logger.info(f"running specific task: {task['run']}")
        logger.info(f"Party: {party}")
        if hasattr(party, task['run']):
            target_func = getattr(party, task['run'])
            if data is not None:
                result = target_func(data)
            else:
                result = target_func()
            logger.info(f"Finished specific task: {task['run']}")
            return result
