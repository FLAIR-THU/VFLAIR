from party.ICommunication import ICommunication
import json
from framework.database.model.Task import Task


class DistributedCommunication(ICommunication):
    _client = None
    _job_id = None

    def __init__(self, client, job_id):
        self._client = client
        self._job_id = job_id

    def send_pred_message(self, pred_list, parse_result_fn, use_cache=False, test="True"):
        task = Task()
        task.run = "aggregate_remote"
        task.party = "active"
        task.job_id = self._job_id

        new_list = [item.tolist() for item in pred_list[0]]
        task.params = new_list

        response = self._client.open_and_send(task)
        result = response.named_values['test_logit'].string
        test_logit = json.loads(result)

        if parse_result_fn is not None:
            return parse_result_fn(test_logit)
        return test_logit

    def send_global_backward_message(self):
        task = Task()
        task.run = "global_backward"
        task.party = "active"
        task.job_id = self._job_id

        response = self._client.open_and_send(task)

    def send_global_loss_and_gradients(self, loss, gradients):
        task = Task()
        task.run = "receive_loss_and_gradients_remote"
        task.party = "active"
        task.job_id = self._job_id
        task.params = {"loss": loss.item(), "gradients": gradients.tolist()}

        response = self._client.open_and_send(task)

    def send_cal_passive_local_gradient_message(self, pred):
        task = Task()
        task.run = "cal_passive_local_gradient"
        task.party = "active"
        task.job_id = self._job_id
        new_list = [item.tolist() for item in pred]
        task.params = new_list

        response = self._client.open_and_send(task)

    def send_global_lr_decay(self, i_epoch):
        task = Task()
        task.run = "global_LR_decay"
        task.party = "active"
        task.job_id = self._job_id
        task.params = str(i_epoch)

        response = self._client.open_and_send(task)

    def send_global_model_train_message(self):
        task = Task()
        task.run = "train_model"
        task.party = "active"
        task.job_id = self._job_id

        response = self._client.open_and_send(task)
