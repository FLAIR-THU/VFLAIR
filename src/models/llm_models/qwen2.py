"""
copy source codes from transformers, then modify
code based on transformers=4.37.2
"""
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2ForCausalLM, Qwen2Config, Qwen2DecoderLayer, \
    BaseModelOutputWithPast, Cache, DynamicCache, _prepare_4d_causal_attention_mask_for_sdpa, \
    _prepare_4d_causal_attention_mask, PreTrainedModel
from torch.nn import ModuleList, Parameter
from typing import Iterable, Optional, Union, List, Tuple, Callable, Dict, Iterator
from loguru import logger
import torch
import copy
import os
from peft.peft_model import PeftModel
from .base import VFLPipeline, VFLModel


class Qwen2DecoderLayerParam(object):
    """
    deal with params between vfl models
    """

    def __init__(self,
                 hidden_states: torch.Tensor,
                 attention_mask: Optional[torch.Tensor] = None,
                 position_ids: Optional[torch.LongTensor] = None,
                 past_key_values: Optional[Tuple[torch.Tensor]] = None,
                 output_attentions: Optional[bool] = False,
                 output_hidden_states: Optional[bool] = False,
                 use_cache: Optional[bool] = False,
                 labels: Optional[torch.Tensor] = None):
        self.hidden_states = hidden_states
        self.attention_mask = attention_mask
        self.position_ids = position_ids
        self.past_key_values = None
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.use_cache = use_cache
        self.labels = None

    def prepare_for_forward(self):
        ans = {
            'inputs_embeds': self.hidden_states,
            'attention_mask': self.attention_mask,
            'past_key_values': self.past_key_values,
            'output_hidden_states': self.output_hidden_states,
            'position_ids': self.position_ids,
            'use_cache': False,
        }
        if self.labels is not None:
            ans.update({'labels': self.labels})
        return ans

    def to(self, device):
        for v in self.__dict__.values():
            if isinstance(v, torch.Tensor):
                v.to(device)
        return self

    def to_json(self):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__.update({k: v.tolist()})

    def detach(self):
        new_dict = {}
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                v_new = v.detach().clone()
                v_new.requires_grad = v.requires_grad
                new_dict.update({k: v_new})
            else:
                new_dict.update({k: copy.deepcopy(v)})
        return Qwen2DecoderLayerParam(**new_dict)

    def get(self, key, default=None):
        if key == 'inputs_embeds':
            key = 'hidden_states'
        return self.__dict__.get(key, default)

    def items(self):
        for k, v in self.__dict__.items():
            yield k, v

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value


class Qwen2ModelSplitter(Qwen2Model, VFLModel):
    def vfl_split(self, idx_of_layers: Iterable[int]) -> bool:
        return self._split_layers(idx_of_layers)

    def _split_layers(self, idx_of_layers: Iterable[int]) -> bool:
        new_layers = ModuleList()
        for i, layer in enumerate(self.layers):
            if i in idx_of_layers:
                new_layers.append(layer)
        self.layers = new_layers
        # update config
        self.config.num_hidden_layers = len(new_layers)
        return True

    def _clear_past_key_values(self):
        self.past_key_values = None


class Qwen2ModelHead(Qwen2ModelSplitter):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        # todo: del norm will cause error when load from original model weight
        # del self.norm

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
            **kwargs
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
        logger.debug(f"{self.__class__.__name__} forward")
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
        self.embedding_output = inputs_embeds # add


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
        # hidden_states = self.norm(hidden_states)

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


class Qwen2ModelBody(Qwen2ModelSplitter):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        # todo: remove self.norm is required
        # del self.norm

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
            **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        logger.debug(f"{self.__class__.__name__} forward")
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
        # hidden_states = self.norm(hidden_states)

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
        # ))


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
        logger.debug(f"{self.__class__.__name__} forward")
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
        return self.model.vfl_split(idx_of_layers)

    def _clear_past_key_values(self):
        self.model._clear_past_key_values()


class E2EModel(Qwen2ForCausalLM):
    def __init__(self, model_config: Qwen2Config, models: Dict[int, Union[PreTrainedModel
        # , ProxyModel
    ]]):
        model_config.tie_word_embeddings = False
        super().__init__(model_config)
        self.layers = None
        self.model = None
        self.lm_head = None
        self.models = models
        self.tensor_in = {}  # type: Dict[int, torch.Tensor]
        self.tensor_out = {}  # type: Dict[int, torch.Tensor]

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for i in range(len(self.models)):
            if isinstance(self.models[i], (PreTrainedModel, PeftModel)):
                yield from self.models[i].parameters()
            else:
                logger.warning(f"Parameter type {type(self.models[i])} is not supported")
                continue

    def can_generate(self) -> bool:
        return True

    @property
    def num_of_slice(self):
        return len(self.models)

    @property
    def device(self) -> torch.device:
        return self.models[0].device

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        self.__is_vfl()
        m = self.models[0]  # type:ProxyModel
        return m.optimizer

    def __is_vfl(self, raise_exception: bool = True):
        """
        some methods only support under vfl structure
        :return: bool
        """
        for model in self.models.items():
            # if not isinstance(model, ProxyModel):
            if isinstance(model, PreTrainedModel):
                if raise_exception:
                    raise ValueError(f"Model type {type(model)} is not supported")
                else:
                    return False
        return True

    def eval(self):
        for model in self.models.values():
            model.eval()

    def train(self, **kwargs):
        for model in self.models.values():
            model.train(**kwargs)

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
        intermediate = None
        for i in range(self.num_of_slice):
            model = self.models[i]
            if i == 0:
                _input = {'input_ids': input_ids,
                          'attention_mask': attention_mask,
                          'position_ids': position_ids,
                          'past_key_values': past_key_values,
                          'inputs_embeds': inputs_embeds,
                          'use_cache': use_cache,
                          'output_attentions': output_attentions,
                          'output_hidden_states': output_hidden_states,
                          'return_dict': return_dict}
            else:
                if not isinstance(intermediate, Qwen2DecoderLayerParam):
                    intermediate = Qwen2DecoderLayerParam(intermediate)
                # intermediate = intermediate.detach()
                if i == len(self.models) - 1:
                    intermediate.labels = labels
                intermediate.to(model.device)
                _input = intermediate.prepare_for_forward()
            if (t := _input.get('inputs_embeds')) is not None:
                self.tensor_in[i] = t
            intermediate = model(**_input)
            if (t := intermediate.get('hidden_states')) is not None:
                self.tensor_out[i] = t
        for k, v in intermediate.items():
            if isinstance(v, torch.Tensor):
                intermediate[k] = v.to(self.device)
            if isinstance(v, tuple) and isinstance(v[0], torch.Tensor):
                intermediate[k] = tuple(t.to(self.device) for t in v)
        logger.debug(f"finish e2e model forward")
        # model_weights = {}
        # for i in range(3):
        #     model_weights.update({i: {k: v for k, v in self.models[i].named_parameters()}})
        # dev_out[2].loss.backward()
        # dev_out[1].hidden_states.backward(gradient=dev_in[2]['inputs_embeds'].grad)
        return intermediate

    def backward(self):
        self.__is_vfl()
        # todo: 依次调用，提供终止方式
        for i in range(len(self.models) - 1, 0, -1):
            self.models[i].backward()

    def optimizer_step(self):
        self.__is_vfl()
        # todo: 可以同时调用
        for i in range(len(self.models) - 1, 0, -1):
            self.models[i].optimizer_step()

    def optimizer_zero_grad(self):
        self.__is_vfl()
        # todo: 可以同时调用
        for i in range(len(self.models) - 1, 0, -1):
            self.models[i].optimizer_zero_grad()

    def lr_scheduler_step(self):
        self.__is_vfl()
        # todo: 可以同时调用
        for i in range(len(self.models) - 1, 0, -1):
            self.models[i].lr_scheduler_step()

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
        for i, model in self.models.items():
            model.save_pretrained(save_directory=os.path.join(save_directory, f'model_{i}'),
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


class VFLPipelineQwen(VFLPipeline):

    def _load_model_head(self, model_name_or_path, do_split=False, **kwargs) -> Union[PreTrainedModel, VFLModel]:
        model_head = Qwen2ModelHead.from_pretrained(model_name_or_path, **kwargs)
        if do_split:
            model_head.vfl_split(range(0, self.split_index[0]))
        return model_head

    def _load_model_tail(self, model_name_or_path, do_split=False, **kwargs) -> Union[PreTrainedModel, VFLModel]:
        model_tail = Qwen2TailForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        if do_split:
            split_index = self.split_index[-1]
            model_tail.vfl_split(
                range(split_index if split_index > 0 else split_index + model_tail.config.num_hidden_layers,
                      model_tail.config.num_hidden_layers))
        return model_tail

    def _load_model_body(self, model_name_or_path, do_split=False, **kwargs) -> Union[PreTrainedModel, VFLModel]:
        model_body = Qwen2ModelBody.from_pretrained(model_name_or_path, **kwargs)
        if do_split:
            split_index = self.split_index
            model_body.vfl_split(range(split_index[0],
                                       split_index[1] if split_index[1] > 0 else
                                       split_index[1] + model_body.config.num_hidden_layers))
        return model_body
