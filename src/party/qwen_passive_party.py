from typing import Dict

from party.llm_party import Party as Party_LLM
from loguru import logger
from .LocalCommunication import LocalCommunication
from transformers import AutoTokenizer
from config import vfl_basic_config
from models.llm_models.qwen2 import VFLPipelineQwen
from dataset.party_dataset import PassiveDataset, PassiveDataset_LLM


class QW_Passive_Party(Party_LLM):
    _communication = None

    def __init__(self, args, index, **kwargs):
        super().__init__(args, index)
        self.init_communication()

    def prepare_data(self, args, index):
        if not args.dataset:
            return None
        super().prepare_data(args, index)  # Party_llm's prepare_data

        self.train_dst = PassiveDataset_LLM(args, self.train_data, self.train_label)
        self.test_dst = PassiveDataset_LLM(args, self.test_data, self.test_label)

    def prepare_model(self, args, index):
        model_path = args.model_list[str(index)]['path']
        args.tokenizer = AutoTokenizer.from_pretrained(model_path)
        p = VFLPipelineQwen(vfl_basic_config.split_index, self.is_active_party)
        self.models.update(p.from_pretrained(model_path, **vfl_basic_config.kwargs_model_loading))
        # whether to train
        if _train_conf := vfl_basic_config.vfl_training_config:
            if _train_conf.peft_config:
                self._peft_model_setting()

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
