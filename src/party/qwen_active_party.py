import torch

from party.llm_party import Party as Party_LLM

from loguru import logger


class QW_Active_Party(Party_LLM):
    def __init__(self, args, index):
        super().__init__(args, index)

    def prepare_data(self, args, index):
        pass

    def predict(self, intermediate):
        if isinstance(intermediate, dict):
            hidden_states = torch.tensor(intermediate['hidden_states']).to(self.args.device)
            attention_mask = torch.tensor(intermediate['attention_mask']).to(self.args.device)
            if intermediate['past_key_values'] is not None:
                past_key_values = torch.tensor(intermediate['past_key_values']).to(self.args.device)
            else:
                past_key_values = None
            output_hidden_states = torch.tensor(intermediate['output_hidden_states']).to(self.args.device)
            position_ids = torch.tensor(intermediate['position_ids']).to(self.args.device)
        else:
            hidden_states = intermediate.hidden_states[0]
            attention_mask = intermediate.attention_mask[0]
            past_key_values = intermediate.past_key_values[0]
            output_hidden_states = intermediate.output_hidden_states
            position_ids = intermediate.position_ids[0]
        resp = self.global_model.forward(inputs_embeds=hidden_states,
                                         attention_mask=attention_mask,
                                         past_key_values=past_key_values,
                                         output_hidden_states=output_hidden_states,
                                         position_ids=position_ids, use_cache=False)
        logger.debug(resp.hidden_states[-1])
        return resp

    def __call__(self, *args, **kwargs):
        return self.predict(**kwargs)