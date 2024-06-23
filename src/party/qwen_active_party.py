import torch
from typing import List, Tuple
from party.llm_party import Party as Party_LLM
from transformers import AutoTokenizer
from peft import get_peft_model
from config import vfl_basic_config
from models.llm_models.qwen2 import VFLPipelineQwen


class QW_Active_Party(Party_LLM):
    def __init__(self, args, index, **kwargs):
        super().__init__(args, index)

    def eval(self, **kwargs):
        self.global_model.eval()

    def prepare_data(self, args, index):
        print('Active Party has no data, only global model')

    def receive_pred(self, pred, giver_index):
        self.pred_received[giver_index] = pred

    def receive_attention_mask(self, attention_mask):
        self.local_batch_attention_mask = attention_mask

    def receive_token_type_ids(self, token_type_ids):
        self.local_batch_token_type_ids = token_type_ids

    def train_model(self):
        self.global_model.train()

    def train_model(self):
        self.global_model.train()

    def prepare_model(self, args, index):
        model_path = args.model_list[str(index)]['path']
        args.tokenizer = AutoTokenizer.from_pretrained(model_path)
        p = VFLPipelineQwen(vfl_basic_config.split_index, self.is_active_party)
        self.models.update(p.from_pretrained(model_path, **vfl_basic_config.kwargs_model_loading))
        # whether to train
        if _train_conf := vfl_basic_config.vfl_training_config:
            if _train_conf.peft_config:
                self._peft_model_setting()

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
