import framework.common.logger_util as logger_util
import json
import framework.server.ActiveTaskService as fst
import framework.protos.message_pb2 as fpm

import framework.database.model.Task as Task

logger = logger_util.get_logger()


class MessageService:
    _queues = {}
    _task_service = None

    def __init__(self, queues):
        self._queues = queues
        self._task_service = fst.ActiveTaskService(self._queues)
        self._task_service.start()

    def _queue_tasks(self, data):
        tasks = data['tasks']
        task = tasks[0]
        client_queue = self._queues[task['party']]
        client_queue.put(task)

    def parse_message(self, message):
        if message.type == fpm.QUERY_JOB:
            # query job detail
            return self.show_job(message)
        elif message.type == fpm.START_TASK:
            # client sending task to active
            task_value = message.data.named_values['task']
            task = json.loads(task_value.string)

            data = task['params']
            config_value = message.data.named_values['config']
            config = json.loads(config_value.string)

            result = self._task_service.run_specific(task, data, config)
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


