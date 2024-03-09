from party.llm_party import Party as Party_LLM

from loguru import logger


class QW_Active_Party(Party_LLM):
    def __init__(self, args, index):
        super().__init__(args, index)

    def prepare_data(self, args, index):
        pass

    def predict(self, intermediate):
        resp = self.global_model.forward(inputs_embeds=intermediate.hidden_states[0],
                                         attention_mask=intermediate.attention_mask[0],
                                         past_key_values=intermediate.past_key_values[0],
                                         output_hidden_states=intermediate.output_hidden_states,
                                         position_ids=intermediate.position_ids[0], use_cache=False)
        logger.debug(resp.hidden_states[-1])
        return resp
