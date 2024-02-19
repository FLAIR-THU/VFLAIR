import threading

import framework.common.logger_util as logger_util
from party.active_party import ActiveParty_LLM
from evaluates.MainTaskVFL_LLM import MainTaskVFL_LLM
from load.LoadConfigs import load_llm_configs

from framework.database.repository.TaskRepository import task_repository

logger = logger_util.get_logger('active_task_service')


class ActiveTaskService(threading.Thread):
    _queues = {}
    _party = None
    _last_result = None
    _main_task = None

    def __init__(self, queues, data, job_id):
        threading.Thread.__init__(self)
        self._queues = queues
        self._data = data
        self._job_id = job_id

    def run(self):
        args = load_llm_configs(self._data)
        if self._main_task is None:
            self._main_task = MainTaskVFL_LLM(args)
            self._party = ActiveParty_LLM(args, 1)
            self._main_task.set_active_party(self._party)

        while True:
            task = self._queues['active'].get()
            logger.info(f"Running task: {task}")
            logger.info(f"Party: {self._party}")
            if hasattr(self._party, task.run):
                target_func = getattr(self._party, task.run)
                result = target_func(last_task_result=self._last_result)
                self._last_result = result
                task_repository.change_status(task.id, 1, result)
                logger.info(f"Finished task: {task}")
                if not self.run_next():
                    break

    def run_specific(self, task, data):
        logger.info(f"running specific task: {task}")
        logger.info(f"Party: {self._party}")
        if hasattr(self._party, task['run']):
            target_func = getattr(self._party, task['run'])
            if data is not None:
                result = target_func(data)
            else:
                result = target_func()
            logger.info(f"Finished specific task: {task}")
            return result

    def run_next(self):
        task = task_repository.find_next(self._job_id)
        if task is None:
            return False
        party = task.party
        if party in self._queues:
            self._queues[party].put(task)
            task_repository.change_start_time(task.id)
            return True
        return False

    def save_and_next(self, data):
        logger.info(f"Saving {data}")
        task_id, result = data['task_id'].sint64, data['result'].string
        task_repository.change_status(task_id, 1, result)
        self._last_result = result
        self.run_next()
