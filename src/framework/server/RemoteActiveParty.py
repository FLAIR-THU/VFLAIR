import json

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from framework.database.model.Task import Task
from party.IParty import IParty


class RemoteActiveParty(IParty):
    _client = None
    _job_id = None
    _args = None
    local_model = None
    global_model = None

    def __init__(self, client, job_id, args):
        self._client = client
        self._job_id = job_id
        self._args = args

    def load_dataset(self):
        pass

    def load_model(self, pred_list, parse_result_fn):
        pass

    def prepare_data_loader(self):
        task = Task()
        task.run = "prepare_data_loader"
        task.party = "active"
        task.job_id = self._job_id
        response = self._client.open_and_send(task)

    def eval(self):
        task = Task()
        task.run = "eval"
        task.party = "active"
        task.job_id = self._job_id
        response = self._client.open_and_send(task)

    def predict(self, value):
        for k, v in value.items():
            if isinstance(v, torch.Tensor):
                value.update({k: v.tolist()})

        task = Task()
        task.run = "distributed_predict"
        task.party = "active"
        task.job_id = self._job_id
        task.params = value
        response = self._client.open_and_send(task)
        result = response.named_values['test_logit'].string
        resp = json.loads(result)
        output = CausalLMOutputWithPast(
            loss=resp['loss'],
            logits=torch.Tensor(resp["logits"]).to(self._args.device),
            hidden_states=resp["hidden_states"],
            attentions=resp["attentions"],
            past_key_values=resp["past_key_values"]
        )
        return output

    def train(self, *args, **kwargs):
        task = Task()
        task.run = "train"
        task.party = "active"
        task.job_id = self._job_id
        response = self._client.open_and_send(task)

    '''
    compatible with local mode
    '''

    def receive_pred(self, pred_list, ik):
        pass

    def update_model_data(self, model_data):
        pass

    def save_pretrained(self, model_index, model_id, **kwargs):
        task = Task()
        task.run = "save_pretrained_remote"
        task.party = "active"
        task.job_id = self._job_id
        task.params = {"model_index": model_index, "model_id": model_id}
        if kwargs:
            task.params.update(kwargs)
        response = self._client.open_and_send(task)

    def __call__(self, *args, **kwargs):
        return self.predict(kwargs)
