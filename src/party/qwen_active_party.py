import torch
from typing import List, Tuple
from party.llm_party import Party as Party_LLM

from loguru import logger


class QW_Active_Party(Party_LLM):
    def __init__(self, args, index, **kwargs):
        super().__init__(args, index)

    def prepare_data(self, args, index):
        pass

    def predict(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, (List, Tuple)):
                kwargs.update({k: torch.tensor(v).to(self.global_model.device)})
        resp = self.global_model.forward(**kwargs)
        # logger.debug(resp.hidden_states[-1])
        # if isinstance(intermediate, dict):
        #     return {
        #         "attentions": resp.get("attentions"),
        #         "hidden_states": resp.get("hidden_states"),
        #         "logits": resp.get("logits").tolist(),
        #         "past_key_values": resp.get("past_key_values"),
        #         "loss": resp.get("loss"),
        #     }
        return resp

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    @property
    def device(self):
        return self.global_model.device