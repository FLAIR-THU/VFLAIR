"""
copy source codes from transformers, then modify
code based on transformers=4.37.2
"""
import gc

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2ForCausalLM, Qwen2Config, Qwen2DecoderLayer, \
    BaseModelOutputWithPast, Cache, DynamicCache, _prepare_4d_causal_attention_mask_for_sdpa, \
    _prepare_4d_causal_attention_mask, PreTrainedModel
from torch.nn import ModuleList
from typing import Iterable, Optional, Union, List, Tuple, Callable
# import logging as logger
from loguru import logger
import torch
import copy
import os


class Qwen2DecoderLayerParam(object):
    def __init__(self,
                 hidden_states: torch.Tensor,
                 attention_mask: Optional[torch.Tensor] = None,
                 position_ids: Optional[torch.LongTensor] = None,
                 past_key_values: Optional[Tuple[torch.Tensor]] = None,
                 output_attentions: Optional[bool] = False,
                 output_hidden_states: Optional[bool] = False,
                 use_cache: Optional[bool] = False):
        self.hidden_states = hidden_states
        self.attention_mask = attention_mask
        self.position_ids = position_ids
        self.past_key_values = None
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.use_cache = use_cache
        self.labels = None

    def prepare_for_forward(self):
        return {
            'inputs_embeds': self.hidden_states,
            'attention_mask': self.attention_mask,
            'past_key_values': self.past_key_values,
            'output_hidden_states': self.output_hidden_states,
            'position_ids': self.position_ids,
            'use_cache': False,
            'labels': self.labels
        }

    def to(self, device):
        for v in self.__dict__.values():
            if isinstance(v, torch.Tensor):
                v.to(device)
        return self

    def to_json(self):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__.update({k: v.tolist()})


class VFLModel:
    def vfl_split(self, idx_of_layers: Iterable[int]) -> bool:
        raise NotImplementedError('Not implemented')


class Qwen2ModelSplitter(Qwen2Model):
    def split_layers(self, idx_of_layers: Iterable[int]) -> bool:
        new_layers = ModuleList()
        for i, layer in enumerate(self.layers):
            if i in idx_of_layers:
                new_layers.append(layer)
        self.layers = new_layers
        # update config
        self.config.num_hidden_layers = len(new_layers)
        return True


class Qwen2ModelHead(Qwen2ModelSplitter, VFLModel):
    def vfl_split(self, idx_of_layers: Iterable[int]) -> bool:
        return self.split_layers(idx_of_layers)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            labels: Optional[torch.LongTensor] = None
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        改写forward方法的输入输出
        :param input_ids:
        :param attention_mask:
        :param position_ids:
        :param past_key_values:
        :param inputs_embeds:
        :param use_cache:
        :param output_attentions:
        :param output_hidden_states:
        :param return_dict:
        :return:
        """
        logger.debug("run local Qwen model forward")
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        intermediate_states = Qwen2DecoderLayerParam(hidden_states=hidden_states,
                                                     attention_mask=attention_mask,
                                                     position_ids=position_ids,
                                                     past_key_values=past_key_values,
                                                     output_attentions=output_attentions,
                                                     output_hidden_states=output_hidden_states,
                                                     use_cache=use_cache, )
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return intermediate_states
        #         , BaseModelOutputWithPast(
        #     last_hidden_state=hidden_states,
        #     past_key_values=next_cache,
        #     hidden_states=all_hidden_states,
        #     attentions=all_self_attns,
        # )


class Qwen2ModelBody(Qwen2ModelSplitter, VFLModel):

    def vfl_split(self, idx_of_layers: Iterable[int]) -> bool:
        return self.split_layers(idx_of_layers)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids:
            raise ValueError('dont support this param, pls use inputs_embeds instead')
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        # 注释掉多余的计算
        # else:
        #     # 4d mask is passed through the layers
        #     attention_mask = _prepare_4d_causal_attention_mask(
        #         attention_mask,
        #         (batch_size, seq_length),
        #         inputs_embeds,
        #         past_key_values_length,
        #         sliding_window=self.config.sliding_window,
        #     )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        intermediate_states = Qwen2DecoderLayerParam(hidden_states=hidden_states,
                                                     attention_mask=attention_mask,
                                                     position_ids=position_ids,
                                                     past_key_values=past_key_values,
                                                     output_attentions=output_attentions,
                                                     use_cache=use_cache, )
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return intermediate_states, BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Qwen2ModelTail(Qwen2ModelSplitter):
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """

        :param input_ids: 禁止传入
        :param attention_mask:
        :param position_ids:
        :param past_key_values:
        :param inputs_embeds:
        :param use_cache:
        :param output_attentions:
        :param output_hidden_states:
        :param return_dict:
        :return:
        """
        logger.debug("run global Qwen model forward")
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids:
            raise ValueError('dont support this param, pls use inputs_embeds instead')
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        # 注释掉多余的计算
        # else:
        #     # 4d mask is passed through the layers
        #     attention_mask = _prepare_4d_causal_attention_mask(
        #         attention_mask,
        #         (batch_size, seq_length),
        #         inputs_embeds,
        #         past_key_values_length,
        #         sliding_window=self.config.sliding_window,
        #     )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Qwen2TailForCausalLM(Qwen2ForCausalLM, VFLModel):
    def __init__(self, config: Qwen2Config, **kwargs):
        super().__init__(config)
        self.model = Qwen2ModelTail(config)
        # Initialize weights and apply final processing
        self.post_init()

    def vfl_split(self, idx_of_layers: Iterable[int]) -> bool:
        return self.model.split_layers(idx_of_layers)


class Qwen2TailForCausalLMExtraCopy(Qwen2ForCausalLM, VFLModel):
    def __init__(self, config: Qwen2Config, **kwargs):
        super().__init__(config)
        self.vfl_model_tail = Qwen2ModelTail(config)
        # self.model_copy=Qwen2ForCausalLM(config) #self.model 在父类初始化时加载了一份完整模型
        # Initialize weights and apply final processing
        self.post_init()

    def vfl_split(self, idx_of_layers: Iterable[int]) -> bool:
        return self.vfl_model_tail.split_layers(idx_of_layers)

    def forward(self, is_first_iter=False, *args, **kwargs):
        if is_first_iter:
            return self.vfl_model_tail(*args, **kwargs)
        else:
            return super(Qwen2ForCausalLM, self).forward(*args, **kwargs)


class E2EModel(Qwen2ForCausalLM):
    def __init__(self, model_config: Qwen2Config, local_model: [Callable, PreTrainedModel],
                 global_model: [Callable, PreTrainedModel]):
        model_config.tie_word_embeddings = False
        super().__init__(model_config)
        self.layers = None
        self.model = None
        self.lm_head = None
        self.local_model = local_model
        self.global_model = global_model
        self.post_init()

    @property
    def device(self) -> torch.device:
        return self.local_model.device

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        intermediate = self.local_model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        position_ids=position_ids,
                                        past_key_values=past_key_values,
                                        inputs_embeds=inputs_embeds,
                                        # labels=labels,
                                        use_cache=use_cache,
                                        output_attentions=output_attentions,
                                        output_hidden_states=output_hidden_states,
                                        return_dict=return_dict)

        if not isinstance(intermediate, Qwen2DecoderLayerParam):
            intermediate = Qwen2DecoderLayerParam(intermediate)
        intermediate.labels = labels
        intermediate = intermediate.prepare_for_forward()

        # logger.debug(str(type(intermediate))+'thread id : '+str(threading.currentThread().ident))
        output = self.global_model(**intermediate)

        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                output[k]=v.to(self.device)
            if isinstance(v,tuple) and isinstance(v[0],torch.Tensor):
                output[k] = tuple(t.to(self.device) for t in v)
        logger.debug(f"finish e2e model forward")
        return output

    def save_pretrained(
            self,
            save_directory: Union[str, os.PathLike],
            is_main_process: bool = True,
            state_dict: Optional[dict] = None,
            save_function: Callable = torch.save,
            push_to_hub: bool = False,
            max_shard_size: Union[int, str] = "5GB",
            safe_serialization: bool = True,
            variant: Optional[str] = None,
            token: Optional[Union[str, bool]] = None,
            save_peft_format: bool = True,
            **kwargs,
    ):
        self.local_model.save_pretrained(save_directory=save_directory,
                                         is_main_process=is_main_process,
                                         state_dict=state_dict,
                                         save_function=save_function,
                                         push_to_hub=push_to_hub,
                                         max_shard_size=max_shard_size,
                                         safe_serialization=safe_serialization,
                                         variant=variant,
                                         token=token,
                                         save_peft_format=save_peft_format,
                                         **kwargs, )
        # self.global_model.save_pretrained(save_directory=save_directory,
        #                                   is_main_process=is_main_process,
        #                                   state_dict=state_dict,
        #                                   save_function=save_function,
        #                                   push_to_hub=push_to_hub,
        #                                   max_shard_size=max_shard_size,
        #                                   safe_serialization=safe_serialization,
        #                                   variant=variant,
        #                                   token=token,
        #                                   save_peft_format=save_peft_format,
        #                                   **kwargs, )


class PipelineVFL2Slice:
    """
    2-slice VFL
    can:
        loading from raw or split model
        save VFL model
    """

    def __init__(self, is_server_end: bool):
        self.is_server_end = is_server_end

    def from_pretrained(self, path_pretrain_model, split_index: int, **kwargs):
        """

        :param path_pretrain_model:
        :param split_index: value should satisfy 0 < value < num_hidden_layers
        :param kwargs:
        :return:
        """
        if self.is_server_end:
            return self._load_model_tail(path_pretrain_model, split_index, **kwargs)
        else:
            return self._load_model_head(path_pretrain_model, split_index, **kwargs)

    def _load_model_head(self, path_model_head, split_index=None, **kwargs):
        model_head = Qwen2ModelHead.from_pretrained(path_model_head, **kwargs)
        if split_index is not None:
            model_head.vfl_split(range(0, split_index))
        return model_head

    def _load_model_tail(self, path_model_tail, split_index=None, **kwargs):
        model_tail = Qwen2TailForCausalLM.from_pretrained(path_model_tail, **kwargs)
        if split_index is not None:
            model_tail.vfl_split(range(split_index, model_tail.config.num_hidden_layers))
        return model_tail

    @staticmethod
    def save_vfl(model_path, model: [Qwen2ModelHead, Qwen2TailForCausalLM] = None):
        if model:
            model.save_pretrained(model_path)

    def from_vfl(self, model_path, **kwargs):
        if self.is_server_end:
            model_tail = self._load_model_tail(model_path, **kwargs)
            return model_tail
        else:
            model_head = self._load_model_head(model_path, **kwargs)
            return model_head
