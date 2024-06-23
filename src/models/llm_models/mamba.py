# from transformers import T5PreTrainedModel
from transformers.models.mamba.modeling_mamba import *
#  (MambaCausalLMOutput,MambaCache,
# MambaModel, MambaForCausalLM, MambaPreTrainedModel)

from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from transformers.utils.import_utils import is_causal_conv1d_available, is_mamba_ssm_available

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


##### Evaluation with pretrained models ######
class MambaForCausalLM_pretrained(MambaForCausalLM):
    def __init__(self, global_mamba, lm_head, generation_config=None):
        super().__init__(global_mamba.config)
        self.backbone = global_mamba  # MambaModel(config)
        self.head_layer = lm_head  # nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.generation_config = generation_config
        # Initialize weights and apply final processing
        # self.post_init()

    def _clear_past_key_values(self):
        self.backbone.past_key_values = None

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            cache_params: Optional[MambaCache] = None,
            labels: Optional[torch.LongTensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            use_cache: Optional[bool] = None,
            **kwargs,  # for now we need this for generation
    ) -> Union[Tuple, MambaCausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        mamba_outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
        )
        hidden_states = mamba_outputs[0]

        logits = self.head_layer(hidden_states.to(self.head_layer.weight.dtype)).float()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + mamba_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MambaCausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=mamba_outputs.cache_params,
            hidden_states=mamba_outputs.hidden_states,
        )


##################### Functional Global Models ######################
class LocalMambaModel(MambaForCausalLM, MambaPreTrainedModel):
    def __init__(self, full_mamba, num_encoders):
        super(MambaPreTrainedModel, self).__init__(full_mamba.config)

        self.local_encoders_num = num_encoders
        self.num_encoders_all = full_mamba.config.num_hidden_layers

        self.embeddings = full_mamba.embeddings  # nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([full_mamba.layers[idx] for idx in range(self.local_encoders_num)])
        # nn.ModuleList([MambaBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        self.norm_f = full_mamba.norm_f  # MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        # Initialize weights and apply final processing
        # self.post_init()

        self.embedding_output = None

    def get_input_embeddings(self):
        return self.embeddings

    def _clear_past_key_values(self):
        self.past_key_values = None

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.LongTensor] = None,
            cache_params: Optional[MambaCache] = None,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,  # `attention_mask` is passed by the tokenizer and we don't want it
    ) -> Union[Tuple, MambaOutput]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        self.embedding_output = inputs_embeds # add

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if cache_params is None and use_cache:
            cache_params = MambaCache(
                self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        for mixer_block in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(mixer_block.__call__, hidden_states, cache_params)
            else:
                hidden_states = mixer_block(hidden_states, cache_params=cache_params)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        return {'inputs_embeds': hidden_states}

        # if use_cache:
        #     cache_params.seqlen_offset += inputs_embeds.shape[1]

        # hidden_states = self.norm_f(hidden_states)

        # if output_hidden_states:
        #     all_hidden_states = all_hidden_states + (hidden_states,)

        # if not return_dict:
        #     return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        # return MambaOutput(
        #     last_hidden_state=hidden_states,
        #     cache_params=cache_params if use_cache else None,
        #     hidden_states=all_hidden_states,
        # )


class GlobalMambaModel(MambaPreTrainedModel):
    def __init__(self, full_mamba, num_encoders):
        super().__init__(full_mamba.config)

        self.global_encoders_num = num_encoders
        self.num_encoders_all = full_mamba.config.num_hidden_layers
        self.local_encoders_num = self.num_encoders_all - self.global_encoders_num

        # self.embeddings = full_mamba.embeddings #nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [full_mamba.layers[idx] for idx in range(self.local_encoders_num, self.num_encoders_all)])
        # nn.ModuleList([MambaBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        self.norm_f = full_mamba.norm_f  # MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # Initialize weights and apply final processing
        self.post_init()

    def _clear_past_key_values(self):
        self.past_key_values = None

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.LongTensor] = None,
            cache_params: Optional[MambaCache] = None,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,  # `attention_mask` is passed by the tokenizer and we don't want it
    ) -> Union[Tuple, MambaOutput]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        # no need
        # if inputs_embeds is None:
        #     inputs_embeds = self.embeddings(input_ids)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if cache_params is None and use_cache:
            cache_params = MambaCache(
                self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        for mixer_block in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(mixer_block.__call__, hidden_states, cache_params)
            else:
                hidden_states = mixer_block(hidden_states, cache_params=cache_params)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if use_cache:
            cache_params.seqlen_offset += inputs_embeds.shape[1]

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return MambaOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )
