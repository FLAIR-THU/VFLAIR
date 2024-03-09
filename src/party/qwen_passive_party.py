from party.llm_party import Party as Party_LLM
from loguru import logger
from .LocalCommunication import LocalCommunication


class QW_Passive_Party(Party_LLM):
    def __init__(self, args, index):
        super().__init__(args, index)

    def prepare_data(self, args, index):
        pass

    def predict(self, kwargs):
        intermediate = self.local_model.forward(**self._format_forward_kwargs(kwargs))[0]
        logger.debug('finish passive party')
        logger.debug(intermediate.hidden_states[-1])
        return intermediate

    def init_communication(self, communication=None):
        if communication is None:
            communication = LocalCommunication(self.args.parties[self.args.k - 1])
        self._communication = communication

    def _format_forward_kwargs(self, kwargs):
        base_dict = {'input_ids': None,
                     'attention_mask': None,
                     'position_ids': None,
                     'past_key_values': None,
                     'inputs_embeds': None,
                     'use_cache': False,
                     'output_attentions': None,
                     'output_hidden_states': True,
                     'return_dict': None, }
        for k in base_dict:
            if k in kwargs:
                base_dict.update({k: kwargs.get(k)})
        return base_dict
