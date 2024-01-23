import framework.common.logger_util as logger_util
import json
import framework.client.PassiveTaskService as fcp

logger = logger_util.get_logger()

class PassiveMessageService:
    _task_service = None
    _client = None

    def __init__(self, client):
        self._client = client

    def _run_task(self, config):
        logger.info("received data: {}".format(config.named_values))

        task_str = config.named_values['task'].string
        config_str = config.named_values['config'].string
        task = json.loads(task_str)
        config = json.loads(config_str)
        if config['fl_type'] == 'VFL':
            if self._task_service is None:
                self._task_service = fcp.PassiveTaskService(config, self._client)
            logger.info(f"task service: {self._task_service}")
            return self._task_service.run(task)


    def parse_message(self, message):
        if message.type == 0:
            logger.info("received data: {}".format(message.data))
            return {}
        if message.type == 1:
            # start task
            return self._run_task(message.data)
