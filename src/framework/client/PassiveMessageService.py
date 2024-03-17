import framework.common.logger_util as logger_util
import json
import framework.client.PassiveTaskService as fcp
import framework.protos.message_pb2 as fpm
import threading
import framework.database.model.Job as Job
from framework.database.repository.JobRepository import job_repository
from datetime import datetime

logger = logger_util.get_logger()


class PassiveMessageService:
    _task_service = None
    _client = None

    def __init__(self, client):
        self._client = client

    def _run_task(self, data):
        logger.info("received data: {}".format(data.named_values))

        task_str = data.named_values['task'].string
        config_str = data.named_values['config'].string
        task = json.loads(task_str)
        config = json.loads(config_str)
        if config['fl_type'] == 'VFL':
            if self._task_service is None:
                self._task_service = fcp.PassiveTaskService(self._client)
            self._task_service.add_job(task['job_id'], config)
            return self._task_service.run(task)

    def _run_job(self, data_dict):
        logger.info("received config: {}".format(data_dict))

        params = data_dict['config']
        data = json.loads(params)
        messages = data_dict['messages'] if 'messages' in data_dict else None

        job_id = self._create_job(data)
        if data['fl_type'] == 'VFL':
            if self._task_service is None:
                self._task_service = fcp.PassiveTaskService(self._client)
        threading.Thread(target=self._task_service.run_job, args=(job_id, data, messages)).start()
        return job_id

    def parse_message(self, message):
        if message.type == fpm.CREATE_JOB:
            # start job
            job_id = self._run_job(message.data)
            return {"job_id": job_id}
        if message.type == fpm.PLAIN:
            logger.info("received data: {}".format(message.data))
            return {}
        if message.type == fpm.START_TASK:
            # start task
            return self._run_task(message.data)

    def _create_job(self, data):
        job = Job.Job()
        job.name = data['fl_type'] + "任务"
        job.fl_type = data['fl_type']
        job.params = json.dumps(data)
        job.create_time = datetime.now()
        job_id = job_repository.create(job)
        job.id = job_id

        return job_id

