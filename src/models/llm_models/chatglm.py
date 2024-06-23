from .third_party_modeling.configuration_chatglm import ChatGLMConfig
from .third_party_modeling.modeling_chatglm import *
from .third_party_modeling.tokenization_chatglm import ChatGLMTokenizer

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, LayerNorm
from torch.nn import CrossEntropyLoss, LayerNorm, MSELoss, BCEWithLogitsLoss
from torch.nn.utils import skip_init
from typing import Optional, Tuple, Union, List, Callable, Dict, Any

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig, ModelOutput

import math
from typing import List, Optional, Tuple, Union


##### Evaluation with pretrained models ######
class ChatGLMForConditionalGeneration_pretrained(ChatGLMForConditionalGeneration):
    def __init__(self, global_chatglm, head_layer, generation_config=None, empty_init=True, device=None):
        super().__init__(global_chatglm.config)

        self.max_sequence_length = global_chatglm.config.max_length

        self.transformer = global_chatglm  # ChatGLMModel(config, empty_init=empty_init, device=device)
        self.head_layer = head_layer  # self.transformer.output_layer
        self.config = global_chatglm.config
        self.quantized = False
        self.generation_config = generation_config

        if self.config.quantization_bit:
            self.quantize(self.config.quantization_bit, empty_init=True)

    def _clear_past_key_values(self):
        self.transformer._clear_past_key_values()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            return_last_logit: Optional[bool] = False,
            **kwargs
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # added code: ensure dtype alignment
        model_dtype = self.transformer.encoder.layers[0].self_attention.query_key_value.weight.dtype
        inputs_embeds =inputs_embeds.to(model_dtype)

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,

            **kwargs
        )

        hidden_states = transformer_outputs[0]
        if return_last_logit:
            hidden_states = hidden_states[-1:]

        # lm_logits = self.transformer.output_layer(hidden_states)
        lm_logits = self.head_layer(hidden_states)

        lm_logits = lm_logits.transpose(0, 1).contiguous()

        loss = None
        if labels is not None:
            lm_logits = lm_logits.to(torch.float32)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


##################### Functional Global Models ######################
class LocalChatGLMModel(ChatGLMForConditionalGeneration, ChatGLMPreTrainedModel):  # ChatGLMPreTrainedModel
    def __init__(self, full_chatglm, num_encoders=1, device=None, empty_init=True):
        super(ChatGLMPreTrainedModel, self).__init__(full_chatglm.config)
        if empty_init:
            init_method = skip_init
        else:
            init_method = default_init
        init_kwargs = {}
        if device is not None:
            init_kwargs["device"] = device

        self.local_num_encoders = num_encoders  # local model hidden layers
        self.num_encoders_all = full_chatglm.config.num_layers  # all hidden layers num

        self.embedding = full_chatglm.embedding  # init_method(Embedding, config, **init_kwargs)

        self.num_layers = self.local_num_encoders  # full_chatglm.config.num_layers
        self.multi_query_group_num = full_chatglm.config.multi_query_group_num
        self.kv_channels = full_chatglm.config.kv_channels

        # Rotary positional embeddings
        self.seq_length = full_chatglm.config.seq_length
        rotary_dim = (
            full_chatglm.config.hidden_size // full_chatglm.config.num_attention_heads if full_chatglm.config.kv_channels is None else full_chatglm.config.kv_channels
        )

        self.rotary_pos_emb = full_chatglm.rotary_pos_emb
        # RotaryEmbedding(rotary_dim // 2, original_impl=config.original_rope, device=device,
        #                                       dtype=config.torch_dtype)

        self.encoder = LocalGLMTransformer(full_chatglm.encoder, config=full_chatglm.config,
                                           num_encoders=self.local_num_encoders)

        # init_method(GLMTransformer, config, **init_kwargs)

        # self.output_layer = full_chatglm.output_layer
        # init_method(nn.Linear, config.hidden_size, config.padded_vocab_size, bias=False,
        #                                 dtype=config.torch_dtype, **init_kwargs)

        self.pre_seq_len = full_chatglm.config.pre_seq_len
        self.prefix_projection = full_chatglm.config.prefix_projection
        if self.pre_seq_len is not None:
            for param in self.parameters():
                param.requires_grad = False
            self.prefix_tokens = torch.arange(self.pre_seq_len).long()
            self.prefix_encoder = PrefixEncoder(config)
            self.dropout = torch.nn.Dropout(0.1)
    
        self.embedding_output = None

    def get_input_embeddings(self):
        return self.embedding

    def _clear_past_key_values(self):
        self.past_key_values = None

    def forward(
            self,
            input_ids,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.BoolTensor] = None,
            full_attention_mask: Optional[torch.BoolTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = input_ids.shape

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)
        
        self.embedding_output = inputs_embeds # add

        if self.pre_seq_len is not None:
            if past_key_values is None:
                past_key_values = self.get_prompt(batch_size=batch_size, device=input_ids.device,
                                                  dtype=inputs_embeds.dtype)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask.new_ones((batch_size, self.pre_seq_len)),
                                            attention_mask], dim=-1)

        if full_attention_mask is None:
            if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
                full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)

        # Rotary positional embeddings
        rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

        # Run encoder.
        local_encoder_output_dict = self.encoder(
            inputs_embeds, full_attention_mask, rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values, use_cache=use_cache, output_hidden_states=output_hidden_states
        )
        # hidden_states, presents, all_hidden_states, all_self_attentions
        # print('local end hidden_states:',local_encoder_output_dict['hidden_states'])


        return {'inputs_embeds': local_encoder_output_dict['hidden_states'],#.transpose(0,1),
                'attention_mask': local_encoder_output_dict['attention_mask'],
                'position_ids': position_ids
                # 'all_hidden_states':all_hidden_states,
                # 'all_self_attentions':all_self_attentions
                }

        # if not return_dict:
        #     return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        # return BaseModelOutputWithPast(
        #     last_hidden_state=hidden_states,
        #     past_key_values=presents,
        #     hidden_states=all_hidden_states,
        #     attentions=all_self_attentions,
        # )


class GlobalChatGLMModel(ChatGLMPreTrainedModel):  # ChatGLMPreTrainedModel
    def __init__(self, full_chatglm, num_encoders=1, device=None, empty_init=True):
        super().__init__(full_chatglm.config)
        if empty_init:
            init_method = skip_init
        else:
            init_method = default_init
        init_kwargs = {}
        if device is not None:
            init_kwargs["device"] = device

        self.global_num_encoders = num_encoders  # global model hidden layers
        self.num_encoders_all = full_chatglm.config.num_layers  # all hidden layers num
        self.local_num_encoders = self.num_encoders_all - self.global_num_encoders  # local model hidden layers

        self.embedding = full_chatglm.embedding  # init_method(Embedding, config, **init_kwargs)

        self.num_layers = self.local_num_encoders  # full_chatglm.config.num_layers
        self.multi_query_group_num = full_chatglm.config.multi_query_group_num
        self.kv_channels = full_chatglm.config.kv_channels

        # Rotary positional embeddings
        self.seq_length = full_chatglm.config.seq_length
        rotary_dim = (
            full_chatglm.config.hidden_size // full_chatglm.config.num_attention_heads if full_chatglm.config.kv_channels is None else full_chatglm.config.kv_channels
        )

        self.rotary_pos_emb = full_chatglm.rotary_pos_emb
        # RotaryEmbedding(rotary_dim // 2, original_impl=config.original_rope, device=device,
        #                                       dtype=config.torch_dtype)

        self.encoder = GlobalGLMTransformer(full_chatglm.encoder, config=full_chatglm.config,
                                            num_encoders=self.global_num_encoders)
        # init_method(GLMTransformer, config, **init_kwargs)

        # self.output_layer = full_chatglm.output_layer
        # init_method(nn.Linear, config.hidden_size, config.padded_vocab_size, bias=False,
        #                                 dtype=config.torch_dtype, **init_kwargs)

        self.pre_seq_len = full_chatglm.config.pre_seq_len
        self.prefix_projection = full_chatglm.config.prefix_projection
        if self.pre_seq_len is not None:
            for param in self.parameters():
                param.requires_grad = False
            self.prefix_tokens = torch.arange(self.pre_seq_len).long()
            self.prefix_encoder = PrefixEncoder(config)
            self.dropout = torch.nn.Dropout(0.1)

    def _clear_past_key_values(self):
        self.past_key_values = None

    def forward(
            self,
            input_ids,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.BoolTensor] = None,
            full_attention_mask: Optional[torch.BoolTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        ## load batch_size, seq_length from inputs_embeds
        # batch_size, seq_length = input_ids.shape
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            # [seq_length, batch_size, embed_dim]
            seq_length, batch_size = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        ## no need
        # if inputs_embeds is None:
        #     inputs_embeds = self.embedding(input_ids)

        if self.pre_seq_len is not None:
            if past_key_values is None:
                past_key_values = self.get_prompt(batch_size=batch_size, device=input_ids.device,
                                                  dtype=inputs_embeds.dtype)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask.new_ones((batch_size, self.pre_seq_len)),
                                            attention_mask], dim=-1)

        ## load full_attention_mask directly from attention_mask
        ## already processed in local model
        # if full_attention_mask is None:
        #     if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
        #         full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)
        full_attention_mask = attention_mask

        # Rotary positional embeddings
        rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

        # Run encoder.
        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
            inputs_embeds, full_attention_mask, rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values, use_cache=use_cache, output_hidden_states=output_hidden_states
        )

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class LocalGLMTransformer(torch.nn.Module):
    """Transformer class."""

    def __init__(self, full_glm_transformer, config: ChatGLMConfig, num_encoders=1, device=None):
        super(LocalGLMTransformer, self).__init__()

        self.fp32_residual_connection = config.fp32_residual_connection
        self.post_layer_norm = config.post_layer_norm

        # Number of layers.
        self.num_layers = config.num_layers

        self.local_num_encoders = num_encoders
        self.global_num_encoders = self.num_layers - num_encoders

        # Transformer layers.
        def build_layer(layer_number):
            return GLMBlock(config, layer_number, device=device)

        self.layers = torch.nn.ModuleList([full_glm_transformer.layers[i] for i in range(self.local_num_encoders)])
        # torch.nn.ModuleList([build_layer(i + 1) for i in range(self.num_layers)])

        # if self.post_layer_norm:
        #     LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
        #     # Final layer norm before output.
        #     self.final_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device,
        #                                          dtype=config.torch_dtype)

        self.gradient_checkpointing = False

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_caches=None,
            use_cache: Optional[bool] = True,
            output_hidden_states: Optional[bool] = False,
    ):
        if not kv_caches:
            kv_caches = [None for _ in range(self.num_layers)]
        presents = () if use_cache else None
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_self_attentions = None
        all_hidden_states = () if output_hidden_states else None
        for index in range(self.local_num_encoders):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer = self._get_layer(index)
            if self.gradient_checkpointing and self.training:
                layer_ret = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_caches[index],
                    use_cache
                )
            else:
                layer_ret = layer(
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_cache=kv_caches[index],
                    use_cache=use_cache
                )
            hidden_states, kv_cache = layer_ret
            if use_cache:
                presents = presents + (kv_cache,)

        return {'hidden_states': hidden_states,
                'attention_mask': attention_mask
                }
        # if output_hidden_states:
        #     all_hidden_states = all_hidden_states + (hidden_states,)

        # # Final layer norm.
        # if self.post_layer_norm:
        #     hidden_states = self.final_layernorm(hidden_states)

        # return hidden_states, presents, all_hidden_states, all_self_attentions


class GlobalGLMTransformer(torch.nn.Module):
    """Transformer class."""

    def __init__(self, full_glm_transformer, config: ChatGLMConfig, num_encoders=1, device=None):
        super(GlobalGLMTransformer, self).__init__()

        self.fp32_residual_connection = config.fp32_residual_connection
        self.post_layer_norm = config.post_layer_norm

        # Number of layers.
        self.num_layers = config.num_layers

        self.global_num_encoders = num_encoders
        self.local_num_encoders = self.num_layers - num_encoders

        # Transformer layers.
        def build_layer(layer_number):
            return GLMBlock(config, layer_number, device=device)

        self.layers = torch.nn.ModuleList(
            [full_glm_transformer.layers[i] for i in range(self.local_num_encoders, self.num_layers)])
        # torch.nn.ModuleList([build_layer(i + 1) for i in range(self.num_layers)])

        if self.post_layer_norm:
            LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
            # Final layer norm before output.
            self.final_layernorm = full_glm_transformer.final_layernorm
            # LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device,dtype=config.torch_dtype)

        self.gradient_checkpointing = False

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_caches=None,
            use_cache: Optional[bool] = True,
            output_hidden_states: Optional[bool] = False,
    ):
        if not kv_caches:
            kv_caches = [None for _ in range(self.num_layers)]
        presents = () if use_cache else None
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_self_attentions = None
        all_hidden_states = () if output_hidden_states else None
        for index in range(self.global_num_encoders):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer = self._get_layer(index)
            if self.gradient_checkpointing and self.training:
                layer_ret = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_caches[index],
                    use_cache
                )
            else:
                layer_ret = layer(
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_cache=kv_caches[index],
                    use_cache=use_cache
                )
            hidden_states, kv_cache = layer_ret
            if use_cache:
                presents = presents + (kv_cache,)
            # if index in [0,1]:
            #     print('global next hidden_states:',hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Final layer norm.
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        # print('global end hidden_states:',hidden_states)

        return hidden_states, presents, all_hidden_states, all_self_attentions
