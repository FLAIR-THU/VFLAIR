import threading

import framework.common.logger_util as logger_util

from evaluates.MainTaskVFL_LLM import MainTaskVFL_LLM
from framework.client.RemotePassiveParty import RemotePassiveParty
from load.LoadConfigs import load_llm_configs
from party.active_party import ActiveParty_LLM
from framework.database.repository.TaskRepository import task_repository

logger = logger_util.get_logger('active_task_service')


class ActiveTaskService(threading.Thread):
    _queues = {}
    _main_tasks = []

    def __init__(self, queues):
        threading.Thread.__init__(self)
        self._queues = queues

    def _init_parties(self, args, job_id):
        active_party = ActiveParty_LLM(args, 1)
        parties = []
        for i in range(args.k - 1):
            client_id = f'c{i + 1}'
            queue = self._queues[client_id]
            parties.append(RemotePassiveParty(client_id, queue, job_id))
        parties.append(active_party)
        return parties

    def add_job(self, job_id, data):
        args = load_llm_configs(data)
        args.parties = self._init_parties(args, job_id)
        # self._main_tasks.append(MainTaskVFL_LLM(args, job_id))
        # self.run_next(job_id)
        main_task = MainTaskVFL_LLM(args, job_id)
        self._main_tasks.append(main_task)
        if args.pipeline == 'pretrained':
            main_task.inference()
        elif args.pipeline == 'finetune':
            main_task.start_train()

    def _get_main_task(self, job_id):
        for main_task in self._main_tasks:
            if main_task.job_id == job_id:
                return main_task

    def run(self):
        while True:
            task = self._queues['active'].get()
            main_task = self._get_main_task(task.job_id)
            party = main_task.get_active_party()
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
        party = self._get_main_task(task['job_id']).get_active_party()
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
        job_id = task_repository.change_status(task_id, 1, result)
        main_task = self._get_main_task(job_id)
        main_task.set_last_result(result)
        self.run_next(job_id)
