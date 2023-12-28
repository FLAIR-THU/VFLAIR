from transformers import BertTokenizer, BertModel,BertConfig,PretrainedConfig, BertPreTrainedModel, BertForSequenceClassification
from transformers.modeling_outputs import (
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
)
from typing import Optional, Tuple, Union, List
import torch.nn as nn
import torch
import copy
from transformers.models.bert.modeling_bert import (BertEmbeddings,BertPooler,BertLayer)

##################### Functional Global Models ######################
'''
Models with specific usage based on BERT
'''

class BertForQuestionAnswering_pretrained(BertPreTrainedModel):
    def __init__(self, global_bert, qa_outputs):
        super().__init__(global_bert.config)
        self.num_labels = global_bert.config.num_labels
        self.backbone = global_bert # BertModel(config, add_pooling_layer=False) bert
        self.qa_outputs = qa_outputs #nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None,

        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.backbone(
            input_ids, attention_mask=attention_mask,
            
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class GlobalBertClassifier_pretrained(nn.Module):
    def __init__(self, globalbert, classifier,dropout=0.5):
        super(GlobalBertClassifier_pretrained, self).__init__()
        self.backbone = globalbert #BertModel.from_pretrained('bert-base-cased')
        self.model_type = globalbert.model_type

        self.classifier = classifier

        # torch.nn.init.xavier_uniform_(self.trainable_layer[1].weight)
        # torch.nn.init.zeros_(self.trainable_layer[1].bias)

    def forward(self, input_ids, attention_mask): # 
        # print('==== global model forward ====')
        outputs = self.backbone(input_ids,attention_mask=attention_mask,return_dict=False)#attention_mask=mask,return_dict=False)
        # print('outputs:',type(outputs),len(outputs)) #([128,256,768], [128,768])

        if self.model_type == 'Bert':
            pooled_output = outputs[1]
        elif self.model_type == 'Roberta':
            pooled_output = outputs[0]
        elif self.model_type == 'Albert':
            pooled_output = outputs[0]

        logits = self.classifier(pooled_output)
        # print('final_layer:',type(final_layer), final_layer.shape,final_layer[0])
        # print('==== global model forward ====')

        # print('logits:',type(logits),logits.shape) #[128,num_classes]
 

        return logits


class GlobalBertClassifier(nn.Module):
    def __init__(self, globalbert, output_dim,dropout=0.5):
        super(GlobalBertClassifier, self).__init__()
        self.backbone = globalbert #BertModel.from_pretrained('bert-base-cased')
        self.model_type = globalbert.model_type

        self.trainable_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, output_dim)
        )

        torch.nn.init.xavier_uniform_(self.trainable_layer[1].weight)
        torch.nn.init.zeros_(self.trainable_layer[1].bias)

    def forward(self, input_id,attention_mask):
        # print('==== global model forward ====')
        outputs = self.backbone(input_id,attention_mask=attention_mask,return_dict=False)#attention_mask=mask,return_dict=False)
        # print('outputs:',type(outputs),len(outputs)) #([128,256,768], [128,768])

        if self.model_type == 'Bert':
            pooled_output = outputs[1]
        elif self.model_type == 'Roberta':
            pooled_output = outputs[0]
        elif self.model_type == 'Albert':
            pooled_output = outputs[0]

        logits = self.trainable_layer(pooled_output)
        # print('final_layer:',type(final_layer), final_layer.shape,final_layer[0])
        # print('==== global model forward ====')

        # print('logits:',type(logits),logits.shape) #[128,num_classes]
 
        return logits

        # return SequenceClassifierOutput(
        #     loss=loss,
        #     logits=logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )

##################### Functional Global Models ######################


############## Split for Basic BERT ###############
class LocalBertEncoder(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        self.config = config
        self.layer = layer #nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        next_decoder_cache = () if use_cache else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)        
        return hidden_states

class GlobalBertEncoder(nn.Module):
    def __init__(self, config, layer, num_encoders):
        super().__init__()
        self.config = config
        self.layer = layer #nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

        self.num_encoders = num_encoders

    def forward(
        self,
        hidden_states,attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        next_decoder_cache = () if use_cache else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i+self.num_encoders] if head_mask is not None else None
            past_key_value = past_key_values[i+self.num_encoders] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class LocalBertModel(BertPreTrainedModel):
    def __init__(self, full_bert, num_encoders, model_type='Bert'):
        super(LocalBertModel, self).__init__(full_bert.config)
        self.model_type = model_type
        self.bert = full_bert

        self.config = full_bert.config
        self.embeddings = full_bert.embeddings #BertEmbeddings(config)
        self.pooler = full_bert.pooler  #BertPooler(config) if add_pooling_layer else None

        self.num_encoders = num_encoders
        if self.model_type == 'Albert':
            self.num_encoders_all = len(full_bert.encoder.albert_layer_groups)
            self.encoder_layer = nn.ModuleList([copy.deepcopy(self.bert.encoder.albert_layer_groups[i]) for i in range(self.num_encoders)])
        else:
            self.num_encoders_all = len(full_bert.encoder.layer)
            self.encoder_layer = nn.ModuleList([copy.deepcopy(self.bert.encoder.layer[i]) for i in range(self.num_encoders)])
        self.encoder = LocalBertEncoder(self.config,self.encoder_layer) #full_bert.encoder #BertEncoder(config)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # print(' === Local Bert ===')
        # print('input:',type(input_ids),input_ids.shape) # [2048,1290]

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # print('input_shape:', input_shape)
        
        batch_size, seq_length = input_shape[:2]
        # print('batch_size:',batch_size,' seq_length',seq_length)

        batch_size = int(batch_size)
        seq_length = int(seq_length)

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # print('bs:',batch_size,'  seq:',seq_length)  # [2048,1029]
        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        # print('attention_mask:',attention_mask.shape,attention_mask)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                # print('self.embeddings.token_type_ids:',self.embeddings.token_type_ids.shape)
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                # print('buffered_token_type_ids:',type(buffered_token_type_ids),buffered_token_type_ids.shape)
                # [1,512]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                # print('buffered_token_type_ids_expanded:',type(buffered_token_type_ids_expanded),buffered_token_type_ids_expanded.shape)
                
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # print('token_type_ids:',token_type_ids.shape,token_type_ids)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        input_ids = input_ids.long()

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        
        intermediate =self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # print('## local output')
        # print('intermediate:',type(intermediate), intermediate.shape )
        # print('attention_mask:',type(attention_mask), attention_mask.shape )
        # print('input_shape:',input_shape )


        return intermediate , attention_mask 

class GlobalBertModel(BertPreTrainedModel):
    def __init__(self, full_bert, num_encoders, model_type = 'Bert'):
        super(GlobalBertModel, self).__init__(full_bert.config)
        self.model_type = model_type
        self.bert = full_bert

        self.config = full_bert.config
        self.embeddings = full_bert.embeddings #BertEmbeddings(config)
        self.pooler = full_bert.pooler  #BertPooler(config) if add_pooling_layer else None
        
        # 创建指定数量的encoder层
        self.num_encoders = num_encoders
        if self.model_type == 'Albert':
            self.num_encoders_all = len(full_bert.encoder.albert_layer_groups)
            self.encoder_layer = nn.ModuleList([copy.deepcopy(self.bert.encoder.albert_layer_groups[i]) for i in range(self.num_encoders,self.num_encoders_all)])
        else:
            self.num_encoders_all = len(full_bert.encoder.layer)
            self.encoder_layer = nn.ModuleList([copy.deepcopy(self.bert.encoder.layer[i]) for i in range(self.num_encoders,self.num_encoders_all)])
        self.encoder = GlobalBertEncoder(self.config,self.encoder_layer,self.num_encoders) #full_bert.encoder #BertEncoder(config)
        
    def forward(
        self,
        intermediate,attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # print(' === Global Bert ===')
        input_shape = intermediate.shape[:2]
        # print('input_shape:',type(input_shape),input_shape)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False


        batch_size, seq_length = input_shape[:2]
        device = intermediate[0].device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        # print('attention_mask:',attention_mask.shape,attention_mask)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                # print('buffered_token_type_ids:',type(buffered_token_type_ids),buffered_token_type_ids.shape)
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                # print('buffered_token_type_ids_expanded:',type(buffered_token_type_ids_expanded),buffered_token_type_ids_expanded.shape)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        # print('token_type_ids:',token_type_ids.shape,token_type_ids)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)


        encoder_outputs = self.encoder(
            intermediate,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

############## Split for Basic BERT ###############

