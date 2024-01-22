from datetime import datetime

import framework.common.logger_util as logger_util
import json
import framework.server.ActiveTaskService as fst
import framework.protos.message_pb2 as fpm

from framework.database.repository.JobRepository import job_repository
from framework.database.repository.TaskRepository import task_repository
import framework.database.model.Job as Job
import framework.database.model.Task as Task

logger = logger_util.get_logger()

class MessageService:
    _queues = {}
    _task_service = None

    def __init__(self, queues):
        self._queues = queues

    def _queue_tasks(self, data):
        tasks = data['tasks']
        task = tasks[0]
        client_queue = self._queues[task['party']]
        client_queue.put(task)

    def _run_task(self, config):
        logger.info("received config: {}".format(config.named_values))

        params = config.named_values['config']
        data = json.loads(params.string)
        job_id = self._create_job(data)
        if data['fl_type'] == 'VFL':
            self._task_service = fst.ActiveTaskService(self._queues, data, job_id)
            self._task_service.run_next()
            self._task_service.start()


    def parse_message(self, message):
        if message.type == 1:
            # start job
            self._run_task(message.data)
            return None
        elif message.type == 2:
            # query job detail
            return {}
        elif message.type == 3:
            # client finish tasks
            self._task_service.save_and_next(message.data.named_values)
            return {}
        elif message.type == 4:
            # client sending task to active
            params = message.data.named_values['pred_list']
            data = json.loads(params.string)
            task = self._init_task()
            result = self._task_service.run_specific(task, data)
            value = fpm.Value()
            if result is None:
                result = {}
            value.string = json.dumps(result)
            return {"test_logit": value}

    def _init_task(self):
        task = Task.Task()
        task.run = "aggregate_remote"
        task.party = 'active'
        return task

    def _create_job(self, data):
        job = Job.Job()
        job.name = data['fl_type']+"任务"
        job.fl_type = data['fl_type']
        job.params = json.dumps(data)
        job.create_time = datetime.now()
        job_id = job_repository.create(job)
        job.id = job_id

        for task_data in data['tasks']:
            task = Task.Task()
            task.task_id = task_data['id']
            task.run = task_data['run']
            task.job_id = job_id
            task.create_time = datetime.now()
            task.status = 0
            task.party = task_data['party']
            task_id = task_repository.create(task)
            task.id = task_id

        return job_id
