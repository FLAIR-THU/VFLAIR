import torch
from typing import List, Tuple
from party.llm_party import Party as Party_LLM
from .party_utils import ProxyModel
from transformers import AutoTokenizer
from peft import get_peft_model
from config import vfl_basic_config
from models.llm_models.qwen2 import PreTrainedModel, Qwen2ModelHead, Qwen2TailForCausalLM, Qwen2DecoderLayerParam, \
    E2EModel, \
    Qwen2ForCausalLM, Qwen2Config, PipelineVFL2Slice, PipelineVFL3Slice, E2EModelV2


class QW_Active_Party(Party_LLM):
    def __init__(self, args, index, **kwargs):
        super().__init__(args, index)
        self.proxy_models = {k: ProxyModel(self, index) for k in self.models}


    def prepare_data(self, args, index):
        pass



    def distributed_predict(self, intermediate):
        resp = self.predict(**intermediate)
        return {
            "attentions": resp.get("attentions"),
            "hidden_states": resp.get("hidden_states"),
            "logits": resp.get("logits").tolist(),
            "past_key_values": resp.get("past_key_values"),
            "loss": resp.get("loss"),
        }

    # def predict(self, **kwargs):
    #     for k, v in kwargs.items():
    #         if isinstance(v, (List, Tuple)):
    #             kwargs.update({k: torch.tensor(v, dtype=torch.bfloat16).to(self.models[2].device)})
    #     resp = self.models[2].forward(**kwargs)
    #     return resp
    #
    # def __call__(self, *args, **kwargs):
    #     return self.predict(*args, **kwargs)
    #
    # @property
    # def device(self):
    #     return self.models[2].device
