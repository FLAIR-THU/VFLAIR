import threading

import framework.common.logger_util as logger_util

from evaluates.MainTaskVFL_LLM import MainTaskVFL_LLM
from framework.client.RemotePassiveParty import RemotePassiveParty
from load.LoadConfigs import load_llm_configs
from load.LoadParty import get_class_constructor
from framework.database.repository.TaskRepository import task_repository
from framework.database.repository.JobRepository import job_repository

logger = logger_util.get_logger('active_task_service')


class ActiveTaskService(threading.Thread):
    _queues = {}
    _main_tasks = {}

    def __init__(self, queues):
        threading.Thread.__init__(self)
        self._queues = queues

    def _init_parties(self, args, job_id):
        active_party = get_class_constructor(args.active_party_class)(args, args.k-1)
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
        self._main_tasks.setdefault(str(job_id), main_task)
        if args.pipeline == 'pretrained':
            result = main_task.inference()
        elif args.pipeline == 'finetune':
            result = main_task.start_train()
        else:
            raise NotImplementedError
        self._save_job_result(job_id, result)
        del self._main_tasks[str(job_id)]

    def _get_main_task(self, job_id):
        return self._main_tasks[str(job_id)]

    def _save_job_result(self, job_id, result):
        job_repository.change_status(job_id, 1, result)

    def run(self):
        while True:
            task = self._queues['active'].get()
            main_task = self._get_main_task(task.job_id)
            party = main_task.get_active_party()
            logger.info(f"Running task: {task.run}")
            logger.info(f"Party: {party}")
            if hasattr(party, task.run):
                target_func = getattr(party, task.run)
                result = target_func(last_task_result=main_task.get_last_result())
                main_task.set_last_result(result)
                task_repository.change_status(task.id, 1, result)
                logger.info(f"Finished task: {task.run}")
                if not self.run_next(task.job_id):
                    break

    def run_specific(self, task, data):
        party = self._get_main_task(task['job_id']).get_active_party()
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
        job_id, result, run = data['job_id'].sint64, data['result'].string, data['run'].string
        # job_id = task_repository.change_status(task_id, 1, result)
        main_task = self._get_main_task(job_id)
        main_task.set_last_result(result, run)
        self.run_next(job_id)
