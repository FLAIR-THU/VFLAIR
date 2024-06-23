import json

import torch
from loguru import logger

from framework.database.model.Task import Task
from framework.protos.message_pb2 import Value
from party.ICommunication import ICommunication
from utils import timer, get_total_size


@timer()
def convert_tensor_to_msg(logits):
    get_total_size({'tensor': logits})
    data_value = Value()
    data_value.tensor.data.shape.extend(logits.shape)
    data_value.tensor.data.value.extend(logits.flatten().tolist())
    data_value.tensor.data.dtype = str(logits.dtype)
    data_value.tensor.requires_grad = logits.requires_grad

    return data_value


@timer()
def convert_msg_to_tensor(msg):
    dtype = getattr(torch, msg.data.dtype.split(".")[1])
    logits = torch.tensor(msg.data.value, dtype=dtype)
    logits = logits.view(torch.Size(msg.data.shape))
    logits.requires_grad = msg.requires_grad
    return logits


@timer()
def convert_pred_to_batch_msg(pred_list):
    SPLIT_SIZE = 10
    total_size = get_total_size(pred_list)
    batch_size = int(total_size / SPLIT_SIZE) + 1
    logger.info(f"split data into {batch_size} parts")

    result = []
    for i in range(batch_size):
        data_value = Value()
        data_value.hidden_states.inputs_embeds.shape.extend(pred_list['inputs_embeds'].shape)
        inputs_embeds = pred_list['inputs_embeds'].flatten().tolist()
        start, end = _compute_range(i, len(inputs_embeds), batch_size)
        data_value.hidden_states.inputs_embeds.value.extend(inputs_embeds[start:end])
        data_value.hidden_states.inputs_embeds.dtype = str(pred_list['inputs_embeds'].dtype)

        if 'attention_mask' in pred_list:
            data_value.hidden_states.attention_mask.shape.extend(pred_list['attention_mask'].shape)
            attention_mask = pred_list['attention_mask'].flatten().tolist()
            start, end = _compute_range(i, len(attention_mask), batch_size)
            data_value.hidden_states.attention_mask.value.extend(attention_mask[start:end])
            data_value.hidden_states.attention_mask.dtype = str(pred_list['attention_mask'].dtype)
            if pred_list['attention_mask'].requires_grad:
                data_value.hidden_states.requires_grads.append('attention_mask')

        if 'position_ids' in pred_list:
            data_value.hidden_states.position_ids.shape.extend(pred_list['position_ids'].shape)
            position_ids = pred_list['position_ids'].flatten().tolist()
            start, end = _compute_range(i, len(position_ids), batch_size)
            data_value.hidden_states.position_ids.value.extend(position_ids[start:end])

        data_value.hidden_states.output_hidden_states = False
        data_value.hidden_states.use_cache = False
        if pred_list['inputs_embeds'].requires_grad:
            data_value.hidden_states.requires_grads.append('inputs_embeds')
        result.append(data_value)

    return result


def merge_data(data_list):
    data_value = Value()
    for i, data in enumerate(data_list):
        if i == 0:
            data_value.hidden_states.inputs_embeds.shape.extend(data.hidden_states.inputs_embeds.shape)
            data_value.hidden_states.inputs_embeds.dtype = data.hidden_states.inputs_embeds.dtype

            data_value.hidden_states.attention_mask.shape.extend(data.hidden_states.attention_mask.shape)
            data_value.hidden_states.attention_mask.dtype = data.hidden_states.attention_mask.dtype

            data_value.hidden_states.position_ids.shape.extend(data.hidden_states.position_ids.shape)

            data_value.hidden_states.output_hidden_states = data.hidden_states.output_hidden_states
            data_value.hidden_states.use_cache = data.hidden_states.use_cache
            data_value.hidden_states.requires_grads.extend(data.hidden_states.requires_grads)

        data_value.hidden_states.inputs_embeds.value.extend(data.hidden_states.inputs_embeds.value)
        data_value.hidden_states.attention_mask.value.extend(data.hidden_states.attention_mask.value)
        data_value.hidden_states.position_ids.value.extend(data.hidden_states.position_ids.value)
    return data_value


def _compute_range(i, total, batch):
    split_size = int(total / batch)
    end = (i + 1) * split_size
    if end > total:
        end = total
    start = i*split_size
    logger.info(f"total: {total}, batch:{batch}, start:{start}, end:{end}")
    return start, end

@timer()
def convert_pred_to_msg(pred_list):
    get_total_size(pred_list)
    data_value = Value()
    data_value.hidden_states.inputs_embeds.shape.extend(pred_list['inputs_embeds'].shape)
    data_value.hidden_states.inputs_embeds.value.extend(pred_list['inputs_embeds'].flatten().tolist())
    data_value.hidden_states.inputs_embeds.dtype = str(pred_list['inputs_embeds'].dtype)

    if 'attention_mask' in pred_list and pred_list['attention_mask'] is not None:
        data_value.hidden_states.attention_mask.shape.extend(pred_list['attention_mask'].shape)
        data_value.hidden_states.attention_mask.value.extend(pred_list['attention_mask'].flatten().tolist())
        data_value.hidden_states.attention_mask.dtype = str(pred_list['attention_mask'].dtype)
        if pred_list['attention_mask'].requires_grad:
            data_value.hidden_states.requires_grads.append('attention_mask')

    if 'position_ids' in pred_list:
        data_value.hidden_states.position_ids.shape.extend(pred_list['position_ids'].shape)
        data_value.hidden_states.position_ids.value.extend(pred_list['position_ids'].flatten().tolist())

    data_value.hidden_states.output_hidden_states = False
    data_value.hidden_states.use_cache = False
    if pred_list['inputs_embeds'].requires_grad:
        data_value.hidden_states.requires_grads.append('inputs_embeds')

    return data_value

@timer()
def convert_msg_to_pred(pred):
    dtype = getattr(torch, pred.inputs_embeds.dtype.split(".")[1])
    inputs_embeds = torch.tensor(pred.inputs_embeds.value, dtype=dtype)
    inputs_embeds = inputs_embeds.view(torch.Size(pred.inputs_embeds.shape))
    if 'inputs_embeds' in pred.requires_grads:
        inputs_embeds.requires_grad = True

    attention_mask = None
    if len(pred.attention_mask.value) > 0:
        dtype = getattr(torch, pred.attention_mask.dtype.split(".")[1])
        attention_mask = torch.tensor(pred.attention_mask.value, dtype=dtype)
        attention_mask = attention_mask.view(torch.Size(pred.attention_mask.shape))
        if 'attention_mask' in pred.requires_grads:
            attention_mask.requires_grad = True

    position_ids = None
    if len(pred.position_ids.value) > 0:
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

        # response = self._client.send_batch(task, new_pred)
        response = self._client.open_and_send(task, new_pred)
        result = response.named_values['test_logit']
        if len(result.hidden_states.inputs_embeds.value) > 0:
            test_logit = result.hidden_states
        elif len(result.tensor.data.value) > 0:
            test_logit = result.tensor
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

        gradients = convert_tensor_to_msg(gradients)

        response = self._client.open_and_send(task, gradients)

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
