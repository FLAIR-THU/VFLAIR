from party.IParty import IParty
from framework.database.model.Task import Task


class RemotePassiveParty(IParty):
    _client_id = None
    _job_id = None
    _queue = None

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

    def predict(self):
        task = Task()
        task.run = "predict"
        task.party = self._client_id
        task.job_id = self._job_id
        self._queue.put(task)

