import torch

from party.ICommunication import ICommunication
import json
from framework.database.model.Task import Task
from framework.protos.message_pb2 import Value
from utils import timer,get_total_size
from loguru import logger

@timer()
def convert_pred_to_msg(pred_list):
    logger.debug(f"{get_total_size(pred_list)} MB")
    data_value = Value()
    data_value.hidden_states.inputs_embeds.shape.extend(pred_list['inputs_embeds'].shape)
    data_value.hidden_states.inputs_embeds.value.extend(pred_list['inputs_embeds'].flatten().tolist())

    data_value.hidden_states.attention_mask.shape.extend(pred_list['attention_mask'].shape)
    data_value.hidden_states.attention_mask.value.extend(pred_list['attention_mask'].flatten().tolist())

    data_value.hidden_states.position_ids.shape.extend(pred_list['position_ids'].shape)
    data_value.hidden_states.position_ids.value.extend(pred_list['position_ids'].flatten().tolist())

    data_value.hidden_states.output_hidden_states = False
    data_value.hidden_states.use_cache = False
    if pred_list['inputs_embeds'].requires_grad:
        data_value.hidden_states.requires_grads.append('inputs_embeds')
    if pred_list['attention_mask'].requires_grad:
        data_value.hidden_states.requires_grads.append('attention_mask')

    return data_value

@timer()
def convert_msg_to_pred(pred, device, dtype=torch.bfloat16):
    inputs_embeds = torch.tensor(pred.inputs_embeds.value,dtype=dtype)
    inputs_embeds = inputs_embeds.view(torch.Size(pred.inputs_embeds.shape))
    if 'inputs_embeds' in pred.requires_grads:
        inputs_embeds.requires_grad = True

    attention_mask = torch.tensor(pred.attention_mask.value,dtype=dtype)
    attention_mask = attention_mask.view(torch.Size(pred.attention_mask.shape))
    if 'attention_mask' in pred.requires_grads:
        attention_mask.requires_grad = True

    position_ids = torch.tensor(pred.position_ids.value)
    position_ids = position_ids.view(torch.Size(pred.position_ids.shape))

    new_dict = {
        "past_key_values": None,
        "output_hidden_states": pred.output_hidden_states,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "use_cache": pred.use_cache
    }
    return new_dict


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

        new_pred = convert_pred_to_msg(pred_list[0])

        task.params = {"grad_enabled": torch.is_grad_enabled()}

        response = self._client.open_and_send(task, new_pred)
        result = response.named_values['test_logit']
        if result.hidden_states:
            test_logit = result.hidden_states
        else:
            test_logit = json.loads(result.string)

        if parse_result_fn is not None:
            return parse_result_fn(test_logit)
        return test_logit

    def send_global_backward_message(self):
        task = Task()
        task.run = "global_backward"
        task.party = "active"
        task.job_id = self._job_id

        response = self._client.open_and_send(task)

    def send_global_loss_and_gradients(self, gradients):
        task = Task()
        task.run = "receive_loss_and_gradients_remote"
        task.party = "active"
        task.job_id = self._job_id
        task.params = {"gradients": gradients.tolist()}

        response = self._client.open_and_send(task)

    def send_cal_passive_local_gradient_message(self, index):
        task = Task()
        task.run = "cal_passive_local_gradient"
        task.party = "active"
        task.job_id = self._job_id
        task.params = index

        response = self._client.open_and_send(task)
        result = response.named_values['test_logit'].string
        test_logit = json.loads(result)
        return test_logit

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
