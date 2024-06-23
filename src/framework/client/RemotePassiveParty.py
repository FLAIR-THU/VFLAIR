from party.IParty import IParty
from framework.database.model.Task import Task
import json


class RemotePassiveParty(IParty):
    _client_id = None
    _job_id = None
    _queue = None
    local_model = None
    global_model = None

    def __init__(self, client_id, queue, job_id):
        self._client_id = client_id
        self._queue = queue
        self._job_id = job_id

    def load_dataset(self):
        pass

    def load_model(self, pred_list, parse_result_fn):
        pass

    def prepare_data_loader(self):
        task = Task()
        task.run = "prepare_data_loader"
        task.party = self._client_id
        task.job_id = self._job_id
        self._queue.put(task)

    def eval(self):
        task = Task()
        task.run = "eval"
        task.party = self._client_id
        task.job_id = self._job_id
        self._queue.put(task)

    def predict(self, **kwargs):
        task = Task()
        task.run = "predict"
        task.party = self._client_id
        task.job_id = self._job_id
        if kwargs is not None and len(kwargs) > 0:
            task.params = json.dumps({
                "input_ids": kwargs.get("input_ids").tolist(),
                "attention_mask": kwargs.get("attention_mask").tolist(),
                "position_ids": kwargs.get("position_ids").tolist(),
                "past_key_values": kwargs.get("past_key_values"),
                "input_embeds": kwargs.get("input_embeds"),
                "use_cache": kwargs.get("use_cache"),
                "output_attentions": kwargs.get("output_attentions"),
                "output_hidden_states": kwargs.get("output_hidden_states"),
                "return_dict": kwargs.get("return_dict"),
            })
        self._queue.put(task)

    def train(self, *args, **kwargs):
        task = Task()
        task.run = "train"
        task.party = self._client_id
        task.job_id = self._job_id
        self._queue.put(task)

    def __call__(self, *args, **kwargs):
        self.predict(**kwargs)
