# from transformers import T5PreTrainedModel
from transformers.models.xlnet.modeling_xlnet import *  # XLNetModel, XLNetPreTrainedModel, XLNetLMHeadModel

# from transformers import PreTrainedModel, add_start_docstrings
from transformers.activations import ACT2FN
from transformers.modeling_utils import PoolerAnswerClass, PoolerEndLogits, PoolerStartLogits, PreTrainedModel, \
    SequenceSummary
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


##### Evaluation with pretrained models ######
class XLNetLMHeadModel_pretrained(XLNetLMHeadModel):

    def __init__(self, global_xlnet, lm_loss, generation_config=None):
        super().__init__(global_xlnet.config)
        self.attn_type = global_xlnet.config.attn_type
        self.same_length = global_xlnet.config.same_length

        self.transformer = global_xlnet  # XLNetModel(config)
        self.head_layer = lm_loss  # nn.Linear(config.d_model, config.vocab_size, bias=True)

        self.generation_config = generation_config

        # # Initialize weights and apply final processing
        self.post_init()

    def _clear_past_key_values(self):
        self.transformer._clear_past_key_values()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            mems: Optional[torch.Tensor] = None,
            perm_mask: Optional[torch.Tensor] = None,
            target_mapping: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            input_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_mems: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,  # delete when `use_cache` is removed in XLNetModel
    ) -> Union[Tuple, XLNetLMHeadModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, num_predict)`, *optional*):
            Labels for masked language modeling. `num_predict` corresponds to `target_mapping.shape[1]`. If
            `target_mapping` is `None`, then `num_predict` corresponds to `sequence_length`.

            The labels should correspond to the masked input words that should be predicted and depends on
            `target_mapping`. Note in order to perform standard auto-regressive language modeling a *<mask>* token has
            to be added to the `input_ids` (see the `prepare_inputs_for_generation` function and examples below)

            Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100` are ignored, the loss
            is only computed for labels in `[0, ..., config.vocab_size]`

        Return:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, XLNetLMHeadModel
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("xlnet/xlnet-large-cased")
        >>> model = XLNetLMHeadModel.from_pretrained("xlnet/xlnet-large-cased")

        >>> # We show how to setup inputs to predict a next token using a bi-directional context.
        >>> input_ids = torch.tensor(
        ...     tokenizer.encode("Hello, my dog is very <mask>", add_special_tokens=False)
        ... ).unsqueeze(
        ...     0
        ... )  # We will predict the masked token
        >>> perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
        >>> perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
        >>> target_mapping = torch.zeros(
        ...     (1, 1, input_ids.shape[1]), dtype=torch.float
        ... )  # Shape [1, 1, seq_length] => let's predict one token
        >>> target_mapping[
        ...     0, 0, -1
        ... ] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)

        >>> outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
        >>> next_token_logits = outputs[
        ...     0
        ... ]  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]

        >>> # The same way can the XLNetLMHeadModel be used to be trained by standard auto-regressive language modeling.
        >>> input_ids = torch.tensor(
        ...     tokenizer.encode("Hello, my dog is very <mask>", add_special_tokens=False)
        ... ).unsqueeze(
        ...     0
        ... )  # We will predict the masked token
        >>> labels = torch.tensor(tokenizer.encode("cute", add_special_tokens=False)).unsqueeze(0)
        >>> assert labels.shape[0] == 1, "only one word will be predicted"
        >>> perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
        >>> perm_mask[
        ...     :, :, -1
        ... ] = 1.0  # Previous tokens don't see last token as is done in standard auto-regressive lm training
        >>> target_mapping = torch.zeros(
        ...     (1, 1, input_ids.shape[1]), dtype=torch.float
        ... )  # Shape [1, 1, seq_length] => let's predict one token
        >>> target_mapping[
        ...     0, 0, -1
        ... ] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)

        >>> outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping, labels=labels)
        >>> loss = outputs.loss
        >>> next_token_logits = (
        ...     outputs.logits
        ... )  # Logits have shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_mems=use_mems,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        logits = self.head_layer(transformer_outputs[0])

        loss = None
        if labels is not None:
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return XLNetLMHeadModelOutput(
            loss=loss,
            logits=logits,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


##################### Functional Global Models ######################
class LocalXLNetModel(XLNetLMHeadModel, XLNetModel, XLNetPreTrainedModel):  # XLNetPreTrainedModel XLNetModel
    def __init__(self, full_xlnet, num_encoders=1):
        super(XLNetPreTrainedModel, self).__init__(full_xlnet.config)

        self.local_num_encoders = num_encoders
        self.num_encoders_all = full_xlnet.config.n_layer

        self.mem_len = full_xlnet.config.mem_len
        self.reuse_len = full_xlnet.config.reuse_len
        self.d_model = full_xlnet.config.d_model
        self.same_length = full_xlnet.config.same_length
        self.attn_type = full_xlnet.config.attn_type
        self.bi_data = full_xlnet.config.bi_data
        self.clamp_len = full_xlnet.config.clamp_len
        self.n_layer = full_xlnet.config.n_layer

        self.word_embedding = full_xlnet.word_embedding  # nn.Embedding(config.vocab_size, config.d_model)
        self.mask_emb = full_xlnet.mask_emb  # nn.Parameter(torch.FloatTensor(1, 1, config.d_model))
        self.layer = nn.ModuleList([full_xlnet.layer[i] for i in range(self.local_num_encoders)])
        self.dropout = full_xlnet.dropout  # nn.Dropout(config.dropout)

        # Initialize weights and apply final processing
        # self.post_init()

        self.embedding_output = None
        
    def get_input_embeddings(self):
        return self.word_embedding

    def _clear_past_key_values(self):
        self.past_key_values = None

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            mems: Optional[torch.Tensor] = None,
            perm_mask: Optional[torch.Tensor] = None,
            target_mapping: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            input_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            use_mems: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,  # delete after depreciation warning is removed
    ) -> Union[Tuple, XLNetModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if "use_cache" in kwargs:
            warnings.warn(
                "The `use_cache` argument is deprecated and will be removed in a future version, use `use_mems`"
                " instead.",
                FutureWarning,
            )
            use_mems = kwargs["use_cache"]

        if self.training:
            use_mems = use_mems if use_mems is not None else self.config.use_mems_train
        else:
            use_mems = use_mems if use_mems is not None else self.config.use_mems_eval

        # the original code for XLNet uses shapes [len, bsz] with the batch dimension at the end
        # but we want a unified interface in the library with the batch size on the first dimension
        # so we move here the first dimension (batch) to the end
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_ids = input_ids.transpose(0, 1).contiguous()
            qlen, bsz = input_ids.shape[0], input_ids.shape[1]
        elif inputs_embeds is not None:
            inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()
            qlen, bsz = inputs_embeds.shape[0], inputs_embeds.shape[1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        token_type_ids = token_type_ids.transpose(0, 1).contiguous() if token_type_ids is not None else None
        input_mask = input_mask.transpose(0, 1).contiguous() if input_mask is not None else None
        attention_mask = attention_mask.transpose(0, 1).contiguous() if attention_mask is not None else None
        perm_mask = perm_mask.permute(1, 2, 0).contiguous() if perm_mask is not None else None
        target_mapping = target_mapping.permute(1, 2, 0).contiguous() if target_mapping is not None else None

        mlen = mems[0].shape[0] if mems is not None and mems[0] is not None else 0
        klen = mlen + qlen

        dtype_float = self.dtype
        device = self.device

        # Attention mask
        # causal attention mask
        if self.attn_type == "uni":
            attn_mask = self.create_mask(qlen, mlen)
            attn_mask = attn_mask[:, :, None, None]
        elif self.attn_type == "bi":
            attn_mask = None
        else:
            raise ValueError(f"Unsupported attention type: {self.attn_type}")

        # data mask: input mask & perm mask
        assert input_mask is None or attention_mask is None, "You can only use one of input_mask (uses 1 for padding) "
        "or attention_mask (uses 0 for padding, added for compatibility with BERT). Please choose one."
        if input_mask is None and attention_mask is not None:
            input_mask = 1.0 - attention_mask

        if input_mask is not None and perm_mask is not None:
            data_mask = input_mask[None] + perm_mask
        elif input_mask is not None and perm_mask is None:
            data_mask = input_mask[None]
        elif input_mask is None and perm_mask is not None:
            data_mask = perm_mask
        else:
            data_mask = None

        if data_mask is not None:
            # all mems can be attended to
            if mlen > 0:
                mems_mask = torch.zeros([data_mask.shape[0], mlen, bsz]).to(data_mask)
                data_mask = torch.cat([mems_mask, data_mask], dim=1)
            if attn_mask is None:
                attn_mask = data_mask[:, :, :, None]
            else:
                attn_mask += data_mask[:, :, :, None]

        if attn_mask is not None:
            attn_mask = (attn_mask > 0).to(dtype_float)

        if attn_mask is not None:
            non_tgt_mask = -torch.eye(qlen).to(attn_mask)
            if mlen > 0:
                non_tgt_mask = torch.cat([torch.zeros([qlen, mlen]).to(attn_mask), non_tgt_mask], dim=-1)
            non_tgt_mask = ((attn_mask + non_tgt_mask[:, :, None, None]) > 0).to(attn_mask)
        else:
            non_tgt_mask = None

        # inputs_embeds --> output_h
        # Word embeddings and prepare h & g hidden states
        if inputs_embeds is not None:
            word_emb_k = inputs_embeds
        else:
            word_emb_k = self.word_embedding(input_ids)
        self.embedding_output = word_emb_k
        output_h = self.dropout(word_emb_k)

        # target_mapping --> output_g
        if target_mapping is not None:
            word_emb_q = self.mask_emb.expand(target_mapping.shape[0], bsz, -1)
            # else:  # We removed the inp_q input which was same as target mapping
            #     inp_q_ext = inp_q[:, :, None]
            #     word_emb_q = inp_q_ext * self.mask_emb + (1 - inp_q_ext) * word_emb_k
            output_g = self.dropout(word_emb_q)
        else:
            output_g = None

        # token_type_ids --> seg_mat
        # Segment embedding
        if token_type_ids is not None:
            # Convert `token_type_ids` to one-hot `seg_mat`
            if mlen > 0:
                mem_pad = torch.zeros([mlen, bsz], dtype=torch.long, device=device)
                cat_ids = torch.cat([mem_pad, token_type_ids], dim=0)
            else:
                cat_ids = token_type_ids

            # `1` indicates not in the same segment [qlen x klen x bsz]
            seg_mat = (token_type_ids[:, None] != cat_ids[None, :]).long()
            seg_mat = nn.functional.one_hot(seg_mat, num_classes=2).to(dtype_float)
        else:
            seg_mat = None

        # Positional encoding
        pos_emb = self.relative_positional_encoding(qlen, klen, bsz=bsz)
        pos_emb = pos_emb.to(output_h.device)
        pos_emb = self.dropout(pos_emb)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads] (a head_mask for each layer)
        # and head_mask is converted to shape [num_hidden_layers x qlen x klen x bsz x n_head]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                head_mask = head_mask.expand(self.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to float if need + fp16 compatibility
        else:
            head_mask = [None] * self.n_layer

        new_mems = ()
        if mems is None:
            mems = [None] * len(self.layer)

        attentions = [] if output_attentions else None
        hidden_states = [] if output_hidden_states else None

        for i, layer_module in enumerate(self.layer):
            if use_mems:
                # cache new mems
                new_mems = new_mems + (self.cache_mem(output_h, mems[i]),)
            if output_hidden_states:
                hidden_states.append((output_h, output_g) if output_g is not None else output_h)

            outputs = layer_module(
                output_h,  # inputs_embeds
                output_g,  # target_mapping
                attn_mask_h=non_tgt_mask,  # attn_mask
                attn_mask_g=attn_mask,
                r=pos_emb,
                seg_mat=seg_mat,  # token_type_ids
                mems=mems[i],
                target_mapping=target_mapping,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
            )
            output_h, output_g = outputs[:2]
            if output_attentions:
                attentions.append(outputs[2])

        return {'inputs_embeds': output_h, 'output_g': output_g,
                'target_mapping': target_mapping,
                'input_mask': input_mask, 'perm_mask': perm_mask,
                'token_type_ids': token_type_ids}
        # # Add last hidden state
        # if output_hidden_states:
        #     hidden_states.append((output_h, output_g) if output_g is not None else output_h)

        # output = self.dropout(output_g if output_g is not None else output_h)

        # # Prepare outputs, we transpose back here to shape [bsz, len, hidden_dim] (cf. beginning of forward() method)
        # output = output.permute(1, 0, 2).contiguous()

        # if not use_mems:
        #     new_mems = None

        # if output_hidden_states:
        #     if output_g is not None:
        #         hidden_states = tuple(h.permute(1, 0, 2).contiguous() for hs in hidden_states for h in hs)
        #     else:
        #         hidden_states = tuple(hs.permute(1, 0, 2).contiguous() for hs in hidden_states)

        # if output_attentions:
        #     if target_mapping is not None:
        #         # when target_mapping is provided, there are 2-tuple of attentions
        #         attentions = tuple(
        #             tuple(att_stream.permute(2, 3, 0, 1).contiguous() for att_stream in t) for t in attentions
        #         )
        #     else:
        #         attentions = tuple(t.permute(2, 3, 0, 1).contiguous() for t in attentions)

        # if not return_dict:
        #     return tuple(v for v in [output, new_mems, hidden_states, attentions] if v is not None)

        # return XLNetModelOutput(
        #     last_hidden_state=output, mems=new_mems, hidden_states=hidden_states, attentions=attentions
        # )


class GlobalXLNetModel(XLNetModel):
    def __init__(self, full_xlnet, num_encoders=1):
        super().__init__(full_xlnet.config)

        self.global_num_encoders = num_encoders
        self.num_encoders_all = full_xlnet.config.n_layer
        self.local_num_encoders = self.num_encoders_all - self.global_num_encoders

        self.mem_len = full_xlnet.config.mem_len
        self.reuse_len = full_xlnet.config.reuse_len
        self.d_model = full_xlnet.config.d_model
        self.same_length = full_xlnet.config.same_length
        self.attn_type = full_xlnet.config.attn_type
        self.bi_data = full_xlnet.config.bi_data
        self.clamp_len = full_xlnet.config.clamp_len
        self.n_layer = full_xlnet.config.n_layer

        # self.word_embedding = full_xlnet.word_embedding #nn.Embedding(config.vocab_size, config.d_model)
        # self.mask_emb = full_xlnet.mask_emb #nn.Parameter(torch.FloatTensor(1, 1, config.d_model))

        self.layer = nn.ModuleList([full_xlnet.layer[i] for i in range(self.local_num_encoders, self.num_encoders_all)])
        self.dropout = full_xlnet.dropout  # nn.Dropout(config.dropout)

        # Initialize weights and apply final processing
        # self.post_init()

    def _clear_past_key_values(self):
        self.past_key_values = None

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            mems: Optional[torch.Tensor] = None,
            perm_mask: Optional[torch.Tensor] = None,
            target_mapping: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            input_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            use_mems: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,

            **kwargs,  # delete after depreciation warning is removed
    ) -> Union[Tuple, XLNetModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if "use_cache" in kwargs:
            warnings.warn(
                "The `use_cache` argument is deprecated and will be removed in a future version, use `use_mems`"
                " instead.",
                FutureWarning,
            )
            use_mems = kwargs["use_cache"]

        if self.training:
            use_mems = use_mems if use_mems is not None else self.config.use_mems_train
        else:
            use_mems = use_mems if use_mems is not None else self.config.use_mems_eval

        # the original code for XLNet uses shapes [len, bsz] with the batch dimension at the end
        # but we want a unified interface in the library with the batch size on the first dimension
        # so we move here the first dimension (batch) to the end
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_ids = input_ids.transpose(0, 1).contiguous()
            qlen, bsz = input_ids.shape[0], input_ids.shape[1]
        elif inputs_embeds is not None:
            # no need to transpose again
            # inputs_embeds = inputs_embeds.transpose(0, 1).contiguous() 
            qlen, bsz = inputs_embeds.shape[0], inputs_embeds.shape[1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # no need to transpose again
        # token_type_ids = token_type_ids.transpose(0, 1).contiguous() if token_type_ids is not None else None
        # input_mask = input_mask.transpose(0, 1).contiguous() if input_mask is not None else None
        # attention_mask = attention_mask.transpose(0, 1).contiguous() if attention_mask is not None else None
        # perm_mask = perm_mask.permute(1, 2, 0).contiguous() if perm_mask is not None else None
        # target_mapping = target_mapping.permute(1, 2, 0).contiguous() if target_mapping is not None else None

        mlen = mems[0].shape[0] if mems is not None and mems[0] is not None else 0
        klen = mlen + qlen

        dtype_float = self.dtype
        device = self.device

        # Attention mask
        # causal attention mask
        if self.attn_type == "uni":
            attn_mask = self.create_mask(qlen, mlen)
            attn_mask = attn_mask[:, :, None, None]
        elif self.attn_type == "bi":
            attn_mask = None
        else:
            raise ValueError(f"Unsupported attention type: {self.attn_type}")

        # data mask: input mask & perm mask
        assert input_mask is None or attention_mask is None, "You can only use one of input_mask (uses 1 for padding) "
        "or attention_mask (uses 0 for padding, added for compatibility with BERT). Please choose one."
        # if input_mask is None and attention_mask is not None:
        #     input_mask = 1.0 - attention_mask
        # input_mask already calculated no need

        if input_mask is not None and perm_mask is not None:
            data_mask = input_mask[None] + perm_mask
        elif input_mask is not None and perm_mask is None:
            data_mask = input_mask[None]
        elif input_mask is None and perm_mask is not None:
            data_mask = perm_mask
        else:
            data_mask = None

        if data_mask is not None:
            # all mems can be attended to
            if mlen > 0:
                mems_mask = torch.zeros([data_mask.shape[0], mlen, bsz]).to(data_mask)
                data_mask = torch.cat([mems_mask, data_mask], dim=1)
            if attn_mask is None:
                attn_mask = data_mask[:, :, :, None]
            else:
                attn_mask += data_mask[:, :, :, None]

        if attn_mask is not None:
            attn_mask = (attn_mask > 0).to(dtype_float)

        if attn_mask is not None:
            non_tgt_mask = -torch.eye(qlen).to(attn_mask)
            if mlen > 0:
                non_tgt_mask = torch.cat([torch.zeros([qlen, mlen]).to(attn_mask), non_tgt_mask], dim=-1)
            non_tgt_mask = ((attn_mask + non_tgt_mask[:, :, None, None]) > 0).to(attn_mask)
        else:
            non_tgt_mask = None

        #### Load output_h/output_g directly
        # Word embeddings and prepare h & g hidden states
        # if inputs_embeds is not None:
        #     word_emb_k = inputs_embeds
        # else:
        #     word_emb_k = self.word_embedding(input_ids)
        # output_h = self.dropout(word_emb_k)
        output_h = inputs_embeds

        # if target_mapping is not None:
        #     word_emb_q = self.mask_emb.expand(target_mapping.shape[0], bsz, -1)
        #     # else:  # We removed the inp_q input which was same as target mapping
        #     #     inp_q_ext = inp_q[:, :, None]
        #     #     word_emb_q = inp_q_ext * self.mask_emb + (1 - inp_q_ext) * word_emb_k
        #     output_g = self.dropout(word_emb_q)
        # else:
        #     output_g = None
        output_g = kwargs['output_g']

        # Segment embedding
        if token_type_ids is not None:
            # Convert `token_type_ids` to one-hot `seg_mat`
            if mlen > 0:
                mem_pad = torch.zeros([mlen, bsz], dtype=torch.long, device=device)
                cat_ids = torch.cat([mem_pad, token_type_ids], dim=0)
            else:
                cat_ids = token_type_ids

            # `1` indicates not in the same segment [qlen x klen x bsz]
            seg_mat = (token_type_ids[:, None] != cat_ids[None, :]).long()
            seg_mat = nn.functional.one_hot(seg_mat, num_classes=2).to(dtype_float)
        else:
            seg_mat = None

        # Positional encoding
        pos_emb = self.relative_positional_encoding(qlen, klen, bsz=bsz)
        pos_emb = pos_emb.to(output_h.device)
        pos_emb = self.dropout(pos_emb)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads] (a head_mask for each layer)
        # and head_mask is converted to shape [num_hidden_layers x qlen x klen x bsz x n_head]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                head_mask = head_mask.expand(self.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to float if need + fp16 compatibility
        else:
            head_mask = [None] * self.n_layer

        new_mems = ()
        if mems is None:
            mems = [None] * len(self.layer)

        attentions = [] if output_attentions else None
        hidden_states = [] if output_hidden_states else None

        for i, layer_module in enumerate(self.layer):
            if use_mems:
                # cache new mems
                new_mems = new_mems + (self.cache_mem(output_h, mems[i]),)
            if output_hidden_states:
                hidden_states.append((output_h, output_g) if output_g is not None else output_h)

            outputs = layer_module(
                output_h,
                output_g,
                attn_mask_h=non_tgt_mask,
                attn_mask_g=attn_mask,
                r=pos_emb,
                seg_mat=seg_mat,
                mems=mems[i],
                target_mapping=target_mapping,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
            )
            output_h, output_g = outputs[:2]
            if output_attentions:
                attentions.append(outputs[2])

        # Add last hidden state
        if output_hidden_states:
            hidden_states.append((output_h, output_g) if output_g is not None else output_h)

        output = self.dropout(output_g if output_g is not None else output_h)

        # Prepare outputs, we transpose back here to shape [bsz, len, hidden_dim] (cf. beginning of forward() method)
        output = output.permute(1, 0, 2).contiguous()

        if not use_mems:
            new_mems = None

        if output_hidden_states:
            if output_g is not None:
                hidden_states = tuple(h.permute(1, 0, 2).contiguous() for hs in hidden_states for h in hs)
            else:
                hidden_states = tuple(hs.permute(1, 0, 2).contiguous() for hs in hidden_states)

        if output_attentions:
            if target_mapping is not None:
                # when target_mapping is provided, there are 2-tuple of attentions
                attentions = tuple(
                    tuple(att_stream.permute(2, 3, 0, 1).contiguous() for att_stream in t) for t in attentions
                )
            else:
                attentions = tuple(t.permute(2, 3, 0, 1).contiguous() for t in attentions)

        if not return_dict:
            return tuple(v for v in [output, new_mems, hidden_states, attentions] if v is not None)

        return XLNetModelOutput(
            last_hidden_state=output, mems=new_mems, hidden_states=hidden_states, attentions=attentions
        )
