import framework.common.logger_util as logger_util
import json
import framework.server.ActiveTaskService as fst
import framework.protos.message_pb2 as fpm

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
        if message.type == fpm.LOAD_MODEL:
            model_id = message.data.named_values['model_id']
            model_type = message.data.named_values['model_type']
            return self._task_service.load_model(model_type.string, model_id.string)
        elif message.type == fpm.CLOSE_JOB:
            # close job
            job_id = message.data.named_values['job_id']
            return self._task_service.close_job(job_id.sint64)
        elif message.type == fpm.START_TASK:
            # client sending task to active
            task_value = message.data.named_values['task']
            task = json.loads(task_value.string)

            config_value = message.data.named_values['config']
            config = json.loads(config_value.string)

            data_value = message.data.named_values['data']
            hidden_states = data_value.hidden_states
            tensor = data_value.tensor

            if len(hidden_states.inputs_embeds.value) > 0:
                data = hidden_states
            elif len(tensor.data.value) > 0:
                data = tensor
            else:
                data = task['params']

            result = self._task_service.run_specific(task, config, data=data)
            value = fpm.Value()
            if result is None:
                result = {}
            if isinstance(result, fpm.Value):
                value = result
            else:
                value.string = json.dumps(result)
            return {"test_logit": value}  # todo change name
        elif message.type == fpm.UPDATE_MODEL_DATA:
            job_id = message.data.named_values['job_id']
            return self._task_service.update_model_data(job_id.sint64)
