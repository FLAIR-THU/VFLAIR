import threading

import framework.common.logger_util as logger_util

from evaluates.MainTaskVFL_LLM import MainTaskVFL_LLM
from load.LoadConfigs import load_llm_configs

from framework.database.repository.TaskRepository import task_repository

logger = logger_util.get_logger('active_task_service')


class ActiveTaskService(threading.Thread):
    _queues = {}
    _main_tasks = []

    def __init__(self, queues):
        threading.Thread.__init__(self)
        self._queues = queues

    def add_job(self, job_id, data):
        args = load_llm_configs(data)
        self._main_tasks.append(MainTaskVFL_LLM(args, job_id))
        self.run_next(job_id)

    def _get_main_task(self, task):
        for main_task in self._main_tasks:
            if main_task.job_id == task['job_id']:
                return main_task

    def run(self):
        while True:
            task = self._queues['active'].get()
            main_task = self._get_main_task(task)
            party = main_task.active_party
            logger.info(f"Running task: {task}")
            logger.info(f"Party: {party}")
            if hasattr(party, task.run):
                target_func = getattr(party, task.run)
                result = target_func(last_task_result=main_task.get_last_result())
                main_task.set_last_result(result)
                task_repository.change_status(task.id, 1, result)
                logger.info(f"Finished task: {task}")
                if not self.run_next(task.job_id):
                    break

    def run_specific(self, task, data):
        party = self._get_main_task(task).active_party
        logger.info(f"running specific task: {task}")
        logger.info(f"Party: {party}")
        if hasattr(party, task['run']):
            target_func = getattr(party, task['run'])
            if data is not None:
                result = target_func(data)
            else:
                result = target_func()
            logger.info(f"Finished specific task: {task}")
            return result

    def run_next(self, job_id):
        task = task_repository.find_next(job_id)
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
        task = task_repository.change_status(task_id, 1, result)
        main_task = self._get_main_task(task)
        main_task.set_last_result(result)
        self.run_next(task.job_id)
