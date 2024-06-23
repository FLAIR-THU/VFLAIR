# from transformers import T5PreTrainedModel
from transformers.models.falcon.modeling_falcon import * 

from transformers import PreTrainedModel, add_start_docstrings
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,)
from transformers.utils import logging, add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

##### Evaluation with pretrained models ######
class FalconForCausalLM_pretrained(FalconForCausalLM):
    def __init__(self, global_falcon, lm_head, generation_config=None):
        super().__init__(global_falcon.config)
        self.transformer = global_falcon #FalconModel(config)
        self.head_layer = lm_head #nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.generation_config = generation_config

        # Initialize weights and apply final processing
        # self.post_init()

    def _clear_past_key_values(self):
        self.transformer._clear_past_key_values()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.head_layer(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

##################### Functional Global Models ######################
class LocalFalconModel(FalconForCausalLM,FalconModel,FalconPreTrainedModel):
    def __init__(self, full_falcon , num_encoders):
        super(FalconPreTrainedModel,self).__init__(full_falcon.config)
        self.local_num_encoders = num_encoders
        self.num_encoders_all = full_falcon.config.num_hidden_layers

        self.embed_dim = full_falcon.config.hidden_size
        self.num_heads = full_falcon.config.num_attention_heads
        self.use_alibi = full_falcon.config.alibi

        # Embedding + LN Embedding
        self.word_embeddings = full_falcon.word_embeddings #nn.Embedding(config.vocab_size, self.embed_dim)

        # Transformer blocks
        self.h = nn.ModuleList([full_falcon.h[i] for i in range(self.local_num_encoders)])
        # nn.ModuleList([FalconDecoderLayer(config) for _ in range(config.num_hidden_layers)])

        self._use_flash_attention_2 = full_falcon.config._attn_implementation == "flash_attention_2"
        self._use_sdpa = full_falcon.config._attn_implementation == "sdpa"

        # Final Layer Norm
        # self.ln_f = full_falcon.ln_f # LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        # self.post_init()

        self.past_key_values = None 
        self.embedding_output = None

    def get_input_embeddings(self):
        return self.word_embeddings

    def _clear_past_key_values(self):
        self.past_key_values = None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            if self.past_key_values == None:
                past_key_values = tuple([None] * len(self.h))
            else:
                past_key_values = self.past_key_values

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        self.embedding_output = inputs_embeds

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # Compute alibi tensor: check build_alibi_tensor documentation
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[-2]

        if self.use_alibi:
            mask = (
                torch.ones(
                    (batch_size, seq_length + past_key_values_length), device=inputs_embeds.device, dtype=torch.long
                )
                if attention_mask is None
                else attention_mask
            )
            alibi = build_alibi_tensor(mask, self.num_heads, dtype=hidden_states.dtype)
        else:
            alibi = None
            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(
                    past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
                )
                position_ids = position_ids.unsqueeze(0)

        origin_attention_mask = attention_mask
        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            if alibi is None:
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                )
            elif head_mask is None:
                alibi = alibi.reshape(batch_size, -1, *alibi.shape[1:])

                attention_mask_2d = attention_mask
                # We don't call _prepare_4d_causal_attention_mask_for_sdpa as we need to mask alibi using the 4D attention_mask untouched.
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                )

                # We take care to integrate alibi bias in the attention_mask here.
                if attention_mask_2d is None:
                    attention_mask = alibi / math.sqrt(self.config.hidden_size // self.num_heads)
                else:
                    min_dtype = torch.finfo(alibi.dtype).min
                    attention_mask = torch.masked_fill(
                        alibi / math.sqrt(self.config.hidden_size // self.num_heads),
                        attention_mask < -1,
                        min_dtype,
                    )

                    # From PyTorch 2.1 onwards, F.scaled_dot_product_attention with the memory-efficient attention backend
                    # produces nans if sequences are completely unattended in the attention mask. Details: https://github.com/pytorch/pytorch/issues/110213
                    if seq_length > 1 and attention_mask.device.type == "cuda":
                        attention_mask = AttentionMaskConverter._unmask_unattended(attention_mask, min_dtype=min_dtype)
            else:
                # PyTorch SDPA does not support head_mask, we fall back on the eager implementation in this case.
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    alibi,
                    attention_mask,
                    position_ids,
                    head_mask[i],
                    layer_past,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi=alibi,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        self.past_key_values = presents

        return {'inputs_embeds': hidden_states, 'attention_mask': origin_attention_mask}

        # # Add last hidden state
        # hidden_states = self.ln_f(hidden_states)

        # if output_hidden_states:
        #     all_hidden_states = all_hidden_states + (hidden_states,)

        # if not return_dict:
        #     return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        # return BaseModelOutputWithPastAndCrossAttentions(
        #     last_hidden_state=hidden_states,
        #     past_key_values=presents,
        #     hidden_states=all_hidden_states,
        #     attentions=all_self_attentions,
        # )

class GlobalFalconModel(FalconModel,FalconPreTrainedModel):
    def __init__(self, full_falcon , num_encoders):
        super().__init__(full_falcon.config)
        self.global_num_encoders = num_encoders
        self.num_encoders_all = full_falcon.config.num_hidden_layers
        self.local_num_encoders = self.num_encoders_all - self.global_num_encoders

        self.embed_dim = full_falcon.config.hidden_size
        self.num_heads = full_falcon.config.num_attention_heads
        self.use_alibi = full_falcon.config.alibi

        # Embedding + LN Embedding
        # self.word_embeddings = full_falcon.word_embeddings #nn.Embedding(config.vocab_size, self.embed_dim)

        # Transformer blocks
        self.h = nn.ModuleList([full_falcon.h[i] for i in range(self.local_num_encoders,self.num_encoders_all)])
        # nn.ModuleList([FalconDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        # for layer_idx in range(len(self.h)):
        #     self.h[layer_idx].attn.layer_idx = self.h[layer_idx].attn.layer_idx  -self.local_num_encoders


        self._use_flash_attention_2 = full_falcon.config._attn_implementation == "flash_attention_2"
        self._use_sdpa = full_falcon.config._attn_implementation == "sdpa"

        # Final Layer Norm
        self.ln_f = full_falcon.ln_f # LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

        self.past_key_values = None

    def _clear_past_key_values(self):
        self.past_key_values = None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        
        if past_key_values is None:
            if self.past_key_values == None:
                past_key_values = tuple([None] * len(self.h))
            else:
                past_key_values = self.past_key_values

        ### no need
        # if inputs_embeds is None:
        #     inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # Compute alibi tensor: check build_alibi_tensor documentation
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[-2]

        if self.use_alibi:
            mask = (
                torch.ones(
                    (batch_size, seq_length + past_key_values_length), device=inputs_embeds.device, dtype=torch.long
                )
                if attention_mask is None
                else attention_mask
            )
            alibi = build_alibi_tensor(mask, self.num_heads, dtype=hidden_states.dtype)
        else:
            alibi = None
            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(
                    past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
                )
                position_ids = position_ids.unsqueeze(0)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            if alibi is None:
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                )
            elif head_mask is None:
                alibi = alibi.reshape(batch_size, -1, *alibi.shape[1:])

                attention_mask_2d = attention_mask
                # We don't call _prepare_4d_causal_attention_mask_for_sdpa as we need to mask alibi using the 4D attention_mask untouched.
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                )

                # We take care to integrate alibi bias in the attention_mask here.
                if attention_mask_2d is None:
                    attention_mask = alibi / math.sqrt(self.config.hidden_size // self.num_heads)
                else:
                    min_dtype = torch.finfo(alibi.dtype).min
                    attention_mask = torch.masked_fill(
                        alibi / math.sqrt(self.config.hidden_size // self.num_heads),
                        attention_mask < -1,
                        min_dtype,
                    )

                    # From PyTorch 2.1 onwards, F.scaled_dot_product_attention with the memory-efficient attention backend
                    # produces nans if sequences are completely unattended in the attention mask. Details: https://github.com/pytorch/pytorch/issues/110213
                    if seq_length > 1 and attention_mask.device.type == "cuda":
                        attention_mask = AttentionMaskConverter._unmask_unattended(attention_mask, min_dtype=min_dtype)
            else:
                # PyTorch SDPA does not support head_mask, we fall back on the eager implementation in this case.
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    alibi,
                    attention_mask,
                    position_ids,
                    head_mask[i],
                    layer_past,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi=alibi,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        self.past_key_values = presents

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
