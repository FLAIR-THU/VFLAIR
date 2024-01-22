import threading

import framework.common.logger_util as logger_util
from party.active_party import ActiveParty_LLM

from argparse import Namespace

from framework.database.repository.TaskRepository import task_repository

logger = logger_util.get_logger('active_task_service')


class ActiveTaskService(threading.Thread):
    _queues = {}
    _party = None
    _last_result = None

    def __init__(self, queues, data, job_id):
        threading.Thread.__init__(self)
        self._queues = queues
        self._data = data
        self._job_id = job_id

    def run(self):
        args = Namespace(**self._data)
        if self._party is None:
            self._party = ActiveParty_LLM(args)

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
        if hasattr(self._party, task.run):
            target_func = getattr(self._party, task.run)
            result = target_func(data)
            logger.info(f"Finished specific task: {task}")
            return {
                # "loss": result.total_loss.float(),
                "start_logits": result.start_logits.tolist(),
                "end_logits": result.end_logits.tolist(),
                # "hidden_states": result.outputs.hidden_states,
                # "attentions": result.outputs.attentions,
            }

    def run_next(self):
        task = task_repository.find_next(self._job_id)
        if task is None:
            return False
        party = task.party
        if party in self._queues:
            self._queues[party].put(task)
            return True
        return False

    def save_and_next(self, data):
        logger.info(f"Saving {data}")
        task_id, result = data['task_id'].sint64, data['result'].string
        task_repository.change_status(task_id, 1, result)
        self._last_result = result
        self.run_next()
