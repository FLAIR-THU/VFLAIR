import torch.nn as nn
import torch
import copy

from transformers.modeling_utils import PreTrainedModel
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration

from typing import Optional, Tuple, Union, List
import warnings
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    SequenceClassifierOutputWithPast,
    BaseModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)


class Llama_VFLGeneration(LlamaForCausalLM):

    def __init__(self, top_vfl):
        super().__init__(top_vfl.args.config)

        self.top_vfl = top_vfl

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        output = self.top_vfl.vfl_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            past_key_values=past_key_values,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        # output.logits = output.logits.squeeze()
        # print('in vfl outputs.logits:', output.logits.shape)

        return output


class GPT2_VFLGeneration(GPT2LMHeadModel):

    def __init__(self, top_vfl):
        super().__init__(top_vfl.args.config)

        self.top_vfl = top_vfl

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        output = self.top_vfl.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            past_key_values=past_key_values,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        # output.logits = output.logits.squeeze()
        # print('in vfl outputs.logits:', output.logits.shape)

        return output


class T5_VFLGeneration(T5ForConditionalGeneration):

    def __init__(self, top_vfl):
        super().__init__(top_vfl.args.config)

        self.top_vfl = top_vfl

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        output = self.top_vfl.vfl_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,

            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict, )

        # output.logits = output.logits.squeeze()
        # print('in vfl outputs.logits:', output.logits.shape)

        return output

# class E2EModel(Qwen2ForCausalLM):
#     def __init__(self, local_model: LocalQwen2Model, global_model: GlobalQwen2Model):
#         super().__init__(local_model.config)
#         self.layers=ModuleList()
#         self.local_model = local_model
#         self.global_model = global_model

#     def forward(
#             self,
#             input_ids: torch.LongTensor = None,
#             attention_mask: Optional[torch.Tensor] = None,
#             position_ids: Optional[torch.LongTensor] = None,
#             past_key_values: Optional[List[torch.FloatTensor]] = None,
#             inputs_embeds: Optional[torch.FloatTensor] = None,
#             labels: Optional[torch.LongTensor] = None,
#             use_cache: Optional[bool] = None,
#             output_attentions: Optional[bool] = None,
#             output_hidden_states: Optional[bool] = None,
#             return_dict: Optional[bool] = None, **kwargs
#     ) -> Union[Tuple, CausalLMOutputWithPast]:
#         intermediate = self.local_model(input_ids=input_ids,
#                                         attention_mask=attention_mask,
#                                         position_ids=position_ids,
#                                         past_key_values=past_key_values,
#                                         inputs_embeds=inputs_embeds,
#                                         # labels=labels,
#                                         use_cache=use_cache,
#                                         output_attentions=output_attentions,
#                                         output_hidden_states=output_hidden_states,
#                                         return_dict=return_dict)[0]  # type: Qwen2DecoderLayerParam

#         output = self.global_model.forward(inputs_embeds=intermediate.hidden_states[0],
#                                            attention_mask=intermediate.attention_mask[0],
#                                            past_key_values=intermediate.past_key_values[0],
#                                            output_hidden_states=intermediate.output_attentions[0],
#                                            position_ids=intermediate.position_ids[0], use_cache=False)
#         return output
