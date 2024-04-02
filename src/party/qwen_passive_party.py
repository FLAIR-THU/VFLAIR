from typing import Dict

from party.llm_party import Party as Party_LLM
from loguru import logger
from .LocalCommunication import LocalCommunication
from .party_utils import ProxyModel


class QW_Passive_Party(Party_LLM):
    _communication = None

    def __init__(self, args, index, **kwargs):
        super().__init__(args, index)
        self.proxy_models = {k: ProxyModel(self, k) for k in self.models}

    def prepare_data(self, args, index):
        pass



    # def predict(self, **kwargs):
    #     pass

    def init_communication(self, communication=None):
        if communication is None:
            communication = LocalCommunication(self.args.parties[self.args.k - 1])
        self._communication = communication

    # def __call__(self, *args, **kwargs):
    #     return self.predict(**kwargs)

    # @property
    # def device(self):
    #     return self.local_model.device
