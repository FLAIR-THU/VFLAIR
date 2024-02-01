from party.ICommunication import ICommunication
import json
import framework.protos.message_pb2 as fpm
import framework.protos.node_pb2 as fpn
import framework.common.MessageUtil as mu
import torch
from load.LoadModels import QuestionAnsweringModelOutput
from framework.database.model.Task import Task

class DistributedCommunication(ICommunication):
    _client = None
    _device = None
    _node = None

    def __init__(self, client, device):
        self._client = client
        self._device = device
        self._node = fpn.Node(node_id=self._client.id)

    def send_pred_message(self, pred_list):
        task = Task()
        task.run = "aggregate_remote"
        task.party = "active"
        task_value = fpm.Value()
        task_value.string = json.dumps(task.to_dict())

        value = fpm.Value()
        new_list = [item.tolist() for item in pred_list[0]]
        value.string = json.dumps(new_list)

        msg = mu.MessageUtil.create(self._node, {"data": value, "task": task_value}, fpm.START_TASK)
        response = self._client.open_and_send(msg)
        result = response.named_values['test_logit'].string
        test_logit = json.loads(result)

        start_logits = torch.Tensor(test_logit['start_logits'])
        end_logits = torch.Tensor(test_logit['end_logits'])

        test_logit_output = QuestionAnsweringModelOutput(
            loss=None,
            start_logits=start_logits.to(self._device),
            end_logits=end_logits.to(self._device),
            hidden_states=None,
            attentions=None,
        )
        return test_logit_output

    def send_global_backward_message(self):
        task = Task()
        task.run = "global_backward"
        task.party = "active"
        task_value = fpm.Value()
        task_value.string = json.dumps(task.to_dict())
        msg = mu.MessageUtil.create(self._node, {"task": task_value}, fpm.START_TASK)
        response = self._client.open_and_send(msg)

    def send_global_loss_and_gradients(self, loss, gradients):
        task = Task()
        task.run = "receive_loss_and_gradients"
        task.party = "active"
        task_value = fpm.Value()
        task_value.string = json.dumps(task.to_dict())

        data_value = fpm.Value()
        data_value.string = json.dumps({"loss": loss, "gradients": gradients})

        msg = mu.MessageUtil.create(self._node, {"task": task_value, "data": data_value}, fpm.START_TASK)
        response = self._client.open_and_send(msg)

    def send_cal_passive_local_gradient_message(self, pred):
        task = Task()
        task.run = "cal_passive_local_gradient"
        task.party = "active"
        task_value = fpm.Value()
        task_value.string = json.dumps(task.to_dict())

        value = fpm.Value()
        new_list = [item.tolist() for item in pred]
        value.string = json.dumps(new_list)

        msg = mu.MessageUtil.create(self._node, {"task": task_value, "data": value}, fpm.START_TASK)
        response = self._client.open_and_send(msg)

    def send_global_lr_decay(self, i_epoch):
        task = Task()
        task.run = "global_LR_decay"
        task.party = "active"
        task_value = fpm.Value()
        task_value.string = json.dumps(task.to_dict())

        data_value = fpm.Value()
        data_value.sint64 = i_epoch
        msg = mu.MessageUtil.create(self._node, {"task": task_value, "data": data_value}, fpm.START_TASK)
        response = self._client.open_and_send(msg)

    def send_global_modal_train_message(self):
        task = Task()
        task.run = "train_model"
        task.party = "active"
        task_value = fpm.Value()
        task_value.string = json.dumps(task.to_dict())
        msg = mu.MessageUtil.create(self._node, {"task": task_value}, fpm.START_TASK)
        response = self._client.open_and_send(msg)

