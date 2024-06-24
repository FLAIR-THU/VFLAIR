import os
import sys
import numpy as np
import random

sys.path.append(os.pardir)

import torch
from torch.utils.data import DataLoader
from loguru import logger

from evaluates.attacks.attack_api import AttackerLoader
from evaluates.defenses.defense_api import DefenderLoader
from load.LoadDataset import load_dataset_per_party, load_dataset_per_party_llm, load_dataset_per_party_backdoor, \
    load_dataset_per_party_noisysample
from load.LoadModels import load_models_per_party_llm

from utils import timer
from utils.noisy_label_functions import add_noise
from utils.noisy_sample_functions import noisy_sample
from utils.basic_functions import cross_entropy_for_onehot, tf_distance_cov_cor, pairwise_dist
from utils.communication_protocol_funcs import Cache

import torch
import re
import collections
from transformers import PreTrainedModel, AutoTokenizer
from peft import get_peft_model,PeftModel
from config import vfl_basic_config
from models.llm_models.qwen2 import VFLPipelineQwen
from models.llm_models.base import VFLPipeline
from load.QwenModelLoader import QwenModelLoader
from load.GPT2ModelLoader import GPT2ModelLoader
np_str_obj_array_pattern = re.compile(r'[SaUO]')


class Party(object):
    def __init__(self, args, index, need_data=True, need_model=True):
        self.name = "party#" + str(index + 1)
        self.index = index
        self.args = args
        args.need_auxiliary = 0
        args.dataset = args.dataset_split['dataset_name']
        # data for training and testing
        self.half_dim = -1
        self.train_data = None
        self.test_data = None
        self.aux_data = None
        self.train_label = None
        self.test_label = None
        self.aux_label = None
        self.train_attribute = None
        self.test_attribute = None
        self.aux_attribute = None
        self.train_dst = None
        self.test_dst = None
        self.aux_dst = None
        self.train_loader = None
        self.test_loader = None
        self.aux_loader = None
        self.attribute_loader = None
        self.attribute_iter = None
        self.local_batch_data = None
        self.local_batch_attention_mask = None
        self.local_batch_token_type_ids = None
        # backdoor poison data and label and target images list
        self.train_poison_data = None
        self.train_poison_label = None
        self.test_poison_data = None
        self.test_poison_label = None
        self.train_target_list = None
        self.test_target_list = None
        # store attributes of model slices
        self.input_tensors = {}  # type:dict[int,torch.Tensor]
        self.output_tensors = {}  # type:dict[int,torch.Tensor]
        self.models = {}  # type:dict[int,PreTrainedModel]
        self.optimizers = {}  # type:dict[int,torch.optim.Optimizer]
        self.lr_schedulers = {}  # type:dict[int,torch.optim.lr_scheduler.LinearLR]
        # local model
        self.local_model = None
        self.local_model_optimizer = None
        # global_model
        self.global_model = None
        self.global_model_optimizer = None
        if need_model:
            self.prepare_model(args, index)

        # attack and defense
        # self.attacker = None
        self.defender = None

        if need_data:
            self.prepare_data(args, index)
        # self.prepare_attacker(args, index)
        # self.prepare_defender(args, index)

        self.local_gradient = None
        self.local_pred = None
        self.local_pred_clone = None

        self.origin_pred = None  # for adversarial training

        self.cache = Cache()
        self.prev_batches = []
        self.num_local_updates = 0

        self.party_time = 0

        ####### predict results ######
        self.input_shape = None
        self.global_pred = None

        self.local_attention_mask = None  # GPT2
        self.local_sequence_lengths = None  # GPT2 Classification
        self.local_attention_mask = None  # Llama

        # for adversarial training
        self.adversary_loss = None
        self.mapping_distance = None

    @property
    def is_active_party(self):
        return self.index == self.args.k - 1

    def eval(self):
        for m in self.models.values():
            if m:
                m.eval()

    def train(self,*args,**kwargs):
        for m in self.models.values():
            if m:
                m.train(*args,**kwargs)

    def update_model_data(self, model_data):
        self.args.tokenizer = model_data['tokenizer']
        self.models = model_data['models']
        self.args.config = model_data['config']
        self.args.model_architectures = self.args.config.architectures
        self.args.model_embedded_dim = self.args.config.hidden_size
        if model_data['generation_config']:
            self.args.generation_config = model_data['generation_config']
        self._set_peft()

    def prepare_data(self, args, index):
        (
            args,
            self.half_dim,
            train_dst,
            test_dst,
        ) = load_dataset_per_party_llm(args, index)

        self.train_data, self.train_label = train_dst
        self.test_data, self.test_label = test_dst
        # self.train_data, self.train_label = train_dst[0][:20],train_dst[1][:20]
        # self.test_data, self.test_label = train_dst[0][:20],train_dst[1][:20]

    def prepare_data_loader(self, need_auxiliary=0, **kwargs):
        # self.train_loader = DataLoader(self.train_dst, batch_size=batch_size) # , 
        # self.test_loader = DataLoader(self.test_dst, batch_size=batch_size) # , shuffle=True ,collate_fn=my_collate
        # if self.args.need_auxiliary == 1 and self.aux_dst != None:
        #     self.aux_loader = DataLoader(self.aux_dst, batch_size=batch_size)

        batch_size = self.args.batch_size
        test_batch_size = self.args.test_batch_size
        self.train_loader = DataLoader(self.train_dst, batch_size=batch_size, collate_fn=lambda x: x)  # ,
        self.test_loader = DataLoader(self.test_dst, batch_size=test_batch_size,
                                      collate_fn=lambda x: x)  # , shuffle=True ,collate_fn=my_collate
        if need_auxiliary == 1 and self.aux_dst != None:
            self.aux_loader = DataLoader(self.aux_dst, batch_size=batch_size, collate_fn=lambda x: x)

    def prepare_model(self, args, index):
        if args.model_type.lower() == 'qwen2':
            model_path = args.model_list[str(index)]['path']
            loader = QwenModelLoader()
            result = loader.load(model_path, self.is_active_party)

            args.tokenizer = result['tokenizer']
            self.models.update(result['models'])
            args.config = result['config']
            args.model_architectures = args.config.architectures
            args.model_embedded_dim = args.config.hidden_size
            args.generation_config = result['generation_config']
            self._set_peft()
        else:
            (
                args,
                self.local_model,
                self.local_model_optimizer,
                self.global_model,
                self.global_model_optimizer
            ) = load_models_per_party_llm(args, index)

    def _set_peft(self):
        """
        peft training or load trained peft weights
        :return:
        """
        if peft_model_path:=self.args.model_list[str(self.index)].get('peft_model_path'):
            for i,m in self.models.items():
                _model_path=os.path.join(peft_model_path,f"model_{i}")
                if m and os.path.exists(_model_path):
                    self.models[i]=PeftModel.from_pretrained(m, _model_path).train()

        if _train_conf := vfl_basic_config.vfl_training_config:
            if _train_conf.peft_config:
                self._peft_model_setting()



    def _peft_model_setting(self):
        _train_conf = vfl_basic_config.vfl_training_config
        logger.info(f"enable peft model setting: \n{str(_train_conf.peft_config)}")

        for i in _train_conf.trainable_slice:
            model = self.models.get(i,None)
            if model:
                model.enable_input_require_grads()
                if not isinstance(model,PeftModel):
                    peft_model = get_peft_model(model, _train_conf.peft_config)
                    peft_model.print_trainable_parameters()
                    self.models.update({i: peft_model})
                else:
                    model.set_adapter('default')
                trainable_params = filter(lambda x: x.requires_grad, model.parameters())
                # 定义优化器和学习率调度器
                optimizer = torch.optim.AdamW(trainable_params, lr=_train_conf.training_args.learning_rate)
                scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, end_factor=0.01, total_iters=10)
                self.optimizers.update({i: optimizer})
                self.lr_schedulers.update({i: scheduler})

    def label_to_one_hot(self, target, num_classes=10):
        target = target.long()
        # print('label_to_one_hot:', target, type(target),type(target[0]))
        try:
            _ = target.size()[1]
            # print("use target itself", target.size())
            onehot_target = target.type(torch.float32).to(self.device)
        except:
            target = torch.unsqueeze(target, 1).to(self.device)
            # print("use unsqueezed target", target.size(),type(target))

            onehot_target = torch.zeros(target.size(0), num_classes, device=self.device)
            onehot_target.scatter_(1, target, 1)
        return onehot_target

    def receive_gradient(self, gradient):
        self.local_gradient = gradient
        return

    def give_pred_old(self, use_cache=False):
        intermediate = self.local_model(input_ids=self.local_batch_data, \
                                        attention_mask=self.local_batch_attention_mask, \
                                        token_type_ids=self.local_batch_token_type_ids, \
                                        past_key_values=self.past_key_values, \
                                        use_cache=use_cache)
        self.local_pred = intermediate['inputs_embeds']
        self.local_attention_mask = intermediate['attention_mask'] if ('attention_mask' in intermediate) else None

        self.local_pred_clone = self.local_pred.detach().clone()
        if self.local_attention_mask != None:
            self.local_attention_mask = self.local_attention_mask.detach().clone()

        # if self.args.model_type in ['Bert','Roberta']:
        #     # SequenceClassification & QuestionAnswering
        #     intermediate = self.local_model(input_ids = self.local_batch_data, \
        #                                     attention_mask = self.local_batch_attention_mask, \
        #                                     token_type_ids=self.local_batch_token_type_ids)
        #     self.local_pred = intermediate['local_pred']
        #     self.local_attention_mask = intermediate['local_attention_mask']

        #     self.local_pred_clone = self.local_pred.detach().clone()
        #     self.local_attention_mask = self.local_attention_mask.detach().clone()

        # elif self.args.model_type == 'GPT2':
        #     if self.args.task_type == 'SequenceClassification':
        #         # self.local_pred,  self.local_sequence_lengths, self.local_attention_mask, _
        #         intermediate = self.local_model(input_ids = self.local_batch_data,\
        #                                         attention_mask = self.local_batch_attention_mask,\
        #                                         token_type_ids = self.local_batch_token_type_ids)
        #         self.local_pred = intermediate['local_pred']
        #         self.local_sequence_lengths = intermediate['local_sequence_lengths']
        #         self.local_attention_mask = intermediate['local_attention_mask']

        #         self.local_pred_clone = self.local_pred.detach().clone()
        #         self.local_attention_mask = self.local_attention_mask.detach().clone()
        #     elif self.args.task_type == 'CausalLM':
        #         intermediate = self.local_model(input_ids = self.local_batch_data,\
        #                                         attention_mask = self.local_batch_attention_mask,\
        #                                         token_type_ids = self.local_batch_token_type_ids,\
        #                                         use_cache = use_cache)
        #         self.local_pred = intermediate['local_pred']
        #         self.local_sequence_lengths = intermediate['local_sequence_lengths']
        #         self.local_attention_mask = intermediate['local_attention_mask']

        #         self.local_pred_clone = self.local_pred.detach().clone()
        #         self.local_attention_mask = self.local_attention_mask.detach().clone()
        #     elif self.args.task_type == 'Generation':
        #         # # renew and transmit past_key_values
        #         # self.local_pred,  self.local_sequence_lengths, self.local_attention_mask ,self.past_key_values = \
        #         #     self.local_model(self.local_batch_data, attention_mask = self.local_batch_attention_mask, \
        #         #     token_type_ids = self.local_batch_token_type_ids, \
        #         #     past_key_values = self.past_key_values ,use_cache = use_cache)

        #         intermediate = self.local_model(input_ids = self.local_batch_data,\
        #                                         attention_mask = self.local_batch_attention_mask,\
        #                                         token_type_ids = self.local_batch_token_type_ids,\
        #                                         use_cache = use_cache)
        #         self.local_pred = intermediate['local_pred']
        #         self.local_sequence_lengths = intermediate['local_sequence_lengths']
        #         self.local_attention_mask = intermediate['local_attention_mask']
        #         self.past_key_values = intermediate['local_past_key_values']

        #         self.local_pred_clone = self.local_pred.detach().clone()
        #         self.local_attention_mask = self.local_attention_mask.detach().clone()

        #     elif self.args.task_type == 'QuestionAnswering':
        #         intermediate = self.local_model(input_ids = self.local_batch_data,\
        #                                         attention_mask = self.local_batch_attention_mask,\
        #                                         token_type_ids = self.local_batch_token_type_ids,\
        #                                         use_cache = use_cache)
        #         self.local_pred = intermediate['local_pred']
        #         self.local_sequence_lengths = intermediate['local_sequence_lengths']
        #         self.local_attention_mask = intermediate['local_attention_mask']

        #         self.local_pred_clone = self.local_pred.detach().clone()
        #         self.local_attention_mask = self.local_attention_mask.detach().clone()

        # elif self.args.model_type == 'Llama':
        #     intermediate = self.local_model(input_ids = self.local_batch_data,\
        #                                     attention_mask = self.local_batch_attention_mask)
        #     self.local_pred = intermediate['local_pred']
        #     self.local_sequence_lengths = intermediate['local_sequence_lengths']
        #     self.local_attention_mask = intermediate['local_attention_mask']
        #     self.past_key_values = intermediate['local_past_key_values']

        #     self.local_pred_clone = self.local_pred.detach().clone()
        #     if self.local_attention_mask != None:
        #         self.local_attention_mask = self.local_attention_mask.detach().clone()

        #     # if self.args.task_type == 'SequenceClassification':
        #     #     self.local_pred,  self.local_sequence_lengths, self.local_attention_mask, self.past_key_values = self.local_model(\
        #     #         self.local_batch_data, attention_mask = self.local_batch_attention_mask)
        #     #     self.local_pred_clone = self.local_pred.detach().clone()
        #     #     self.local_attention_mask = self.local_attention_mask.detach().clone()
        #     # elif self.args.task_type == 'CausalLM':
        #     #     self.local_pred,  self.local_sequence_lengths, self.local_attention_mask, self.past_key_values  = self.local_model(\
        #     #         self.local_batch_data, attention_mask = self.local_batch_attention_mask)
        #     #     self.local_pred_clone = self.local_pred.detach().clone()
        #     #     if (self.local_attention_mask!=None):
        #     #         self.local_attention_mask = self.local_attention_mask.detach().clone()
        #     # elif self.args.task_type == 'Generation':
        #     #     self.local_pred,  self.local_sequence_lengths, self.local_attention_mask , self.past_key_values = \
        #     #         self.local_model(self.local_batch_data, attention_mask = self.local_batch_attention_mask, \
        #     #         past_key_values = self.past_key_values ,use_cache = use_cache)
        #     #     self.local_pred_clone = self.local_pred.detach().clone()
        #     #     self.local_attention_mask = self.local_attention_mask.detach().clone()
        #     # elif self.args.task_type == 'QuestionAnswering':
        #     #     self.local_pred,  self.local_sequence_lengths, self.local_attention_mask, self.past_key_values  = self.local_model(self.local_batch_data, attention_mask = self.local_batch_attention_mask)
        #     #     self.local_pred_clone = self.local_pred.detach().clone()
        #     #     self.local_attention_mask = self.local_attention_mask.detach().clone()

        # elif self.args.model_type == 'T5':
        #     if self.args.task_type == 'CausalLM':
        #         self.local_pred,  self.local_attention_mask = \
        #             self.local_model(self.local_batch_data, attention_mask = self.local_batch_attention_mask, \
        #             use_cache = use_cache)
        #         self.local_pred_clone = self.local_pred.detach().clone()
        #         self.local_attention_mask = self.local_attention_mask.detach().clone()

        ######### Defense Applied on Local Model Prediction Process ###########
        if self.args.apply_mid and (self.index in self.args.defense_configs["party"]) and (self.mid_position == "out"):
            self.local_pred, self.mid_loss = self.mid_model(self.local_pred)  # , self.local_attention_mask
            self.local_pred_clone = self.local_pred.detach().clone()

        elif self.args.apply_mid and (self.index in self.args.defense_configs["party"]) and (
                self.mid_position == "inner"):
            # print('inner mid: self.mid_position=',self.mid_position)
            self.mid_loss = self.local_model.mid_loss
            # print(' self.local_model.mid_loss:', self.local_model.mid_loss)

        elif (self.args.apply_adversarial == True and (self.index in self.args.defense_configs["party"])):
            self.origin_pred = self.local_pred.clone()
            # print('self.origin_pred:',self.origin_pred.shape)
            self.local_pred = self.adversarial_model(self.origin_pred)
            self.local_pred_clone = self.local_pred.detach().clone()
        ######### Defense Applied on Local Model Prediction Process ###########

        self.transferred_past_key_values = None  # no need to transmit past_key_values
        if use_cache:  # need to transmit past_key_values
            self.transferred_past_key_values = self.past_key_values

        intermediate['inputs_embeds'] = self.local_pred_clone
        intermediate['attention_mask'] = self.local_attention_mask
        # intermediate['past_key_values'] = self.transferred_past_key_values

        return intermediate

        # if self.args.model_type in ['Bert','Roberta']:
        #     intermediate['local_pred'] = self.local_pred_clone
        #     intermediate['local_attention_mask'] = self.local_attention_mask
        #     intermediate['local_past_key_values'] = self.transferred_past_key_values
        #     return intermediate
        #     # self.local_pred, self.local_pred_clone , self.local_attention_mask, self.transferred_past_key_values
        # elif self.args.model_type == 'GPT2':
        #     if self.args.task_type == 'SequenceClassification':
        #         intermediate['local_pred'] = self.local_pred_clone
        #         intermediate['local_attention_mask'] = self.local_attention_mask
        #         intermediate['local_past_key_values'] = self.transferred_past_key_values
        #         intermediate['local_sequence_lengths'] = self.local_sequence_lengths
        #         return intermediate
        #         # self.local_pred, self.local_pred_clone, self.local_attention_mask ,self.transferred_past_key_values, \
        #         #         self.local_sequence_lengths
        #     elif self.args.task_type == 'CausalLM':
        #         intermediate['local_pred'] = self.local_pred_clone
        #         intermediate['local_attention_mask'] = self.local_attention_mask
        #         intermediate['local_past_key_values'] = self.transferred_past_key_values
        #         return intermediate
        #     elif self.args.task_type == 'Generation':
        #         intermediate['local_pred'] = self.local_pred_clone
        #         intermediate['local_attention_mask'] = self.local_attention_mask
        #         intermediate['local_past_key_values'] = self.transferred_past_key_values
        #         return intermediate
        #     elif self.args.task_type == 'QuestionAnswering':
        #         intermediate['local_pred'] = self.local_pred_clone
        #         intermediate['local_attention_mask'] = self.local_attention_mask
        #         intermediate['local_past_key_values'] = self.transferred_past_key_values
        #         return intermediate
        # elif self.args.model_type == 'Llama':
        #     intermediate['local_pred'] = self.local_pred_clone
        #     intermediate['local_attention_mask'] = self.local_attention_mask
        #     intermediate['local_past_key_values'] = self.transferred_past_key_values
        #     intermediate['local_sequence_lengths'] = self.local_sequence_lengths
        #     return intermediate

        # elif self.args.model_type == 'T5':
        #     if self.args.task_type == 'CausalLM':
        #         return self.local_pred, self.local_pred_clone,self.local_attention_mask, self.transferred_past_key_values

    def _tensor_to_device(self,dict_like:dict,device):
        for k,v in dict_like.items():
            if isinstance(v,torch.Tensor):
                dict_like[k] = v.to(device)

    @timer()
    def give_pred(self, use_cache=False):
        # print('give_pred_dev:',self.local_data_input.keys())
        self.local_data_input['use_cache'] = use_cache
        self._tensor_to_device(self.local_data_input,self.models[0].device)
        intermediate = self.forward(model_index=0,**self.local_data_input)
        if not isinstance(intermediate, dict):
            intermediate = intermediate.prepare_for_forward()
        self.local_pred = intermediate['inputs_embeds']
        self.local_attention_mask = intermediate['attention_mask'] if ('attention_mask' in intermediate) else None

        self.local_pred_clone = self.local_pred.detach().clone()
        if self.local_attention_mask != None:
            self.local_attention_mask = self.local_attention_mask.detach().clone()

        ######### Defense Applied on Local Model Prediction Process ###########
        if self.args.apply_mid and (self.index in self.args.defense_configs["party"]) and (self.mid_position == "out"):
            self.local_pred, self.mid_loss = self.mid_model(self.local_pred)  # , self.local_attention_mask
            self.local_pred_clone = self.local_pred.detach().clone()

        elif self.args.apply_mid and (self.index in self.args.defense_configs["party"]) and (
                self.mid_position == "inner"):
            # print('inner mid: self.mid_position=',self.mid_position)
            self.mid_loss = self.local_model.mid_loss
            # print(' self.local_model.mid_loss:', self.local_model.mid_loss)

        elif (self.args.apply_adversarial == True and (self.index in self.args.defense_configs["party"])):
            self.origin_pred = self.local_pred.clone()
            # if self.args.model_type == 'ChatGLM':
            #     self.origin_pred = self.origin_pred.transpose(0,1)

            self.local_pred = self.adversarial_model(self.origin_pred)

            # if self.args.model_type == 'ChatGLM':
            #     self.local_pred = self.local_pred.transpose(0,1)
                
            self.local_pred_clone = self.local_pred.detach().clone()
        ######### Defense Applied on Local Model Prediction Process ###########

        # self.transferred_past_key_values = None # no need to transmit past_key_values
        # if use_cache: # need to transmit past_key_values
        #     self.transferred_past_key_values = self.past_key_values

        intermediate['inputs_embeds'] = self.local_pred_clone
        if self.local_attention_mask != None:
            intermediate['attention_mask'] = self.local_attention_mask

        return intermediate

    def give_current_lr(self):
        return (self.local_model_optimizer.state_dict()['param_groups'][0]['lr'])

    def LR_decay(self, i_epoch):
        eta_0 = self.args.main_lr
        eta_t = eta_0 / (np.sqrt(i_epoch + 1))
        for param_group in self.local_model_optimizer.param_groups:
            param_group['lr'] = eta_t

    def obtain_local_data_old(self, input_ids=None,
                              inputs_embeds=None,
                              local_batch_attention_mask=None,
                              local_batch_token_type_ids=None,
                              past_key_values=None,
                              **kwargs):
        self.local_batch_data = input_ids  # input_ids
        self.local_batch_attention_mask = local_batch_attention_mask
        self.local_batch_token_type_ids = local_batch_token_type_ids

        self.past_key_values = past_key_values

    def obtain_local_data(self, data_input_dict, **kwargs):
        # self.local_batch_data = kwargs['input_ids'] # input_ids
        # self.local_batch_attention_mask = kwargs['attention_mask']
        # self.local_batch_token_type_ids = kwargs['token_type_ids'] if 'token_type_ids' in kwargs else None
        # self.past_key_values = past_key_values
        # print('obtain_local_data_dev:',kwargs.keys())
        if data_input_dict:
            self._tensor_to_device(data_input_dict,self.models[0].device)
            self.local_data_input = data_input_dict
        else:
            pass

    @property
    def local_model(self):
        if 0 in self.models:
            return self.models[0]
        else:
            return None

    @local_model.setter
    def local_model(self, model):
        if model is None:
            pass
        else:
            self.models.update({0: model})

    @property
    def local_model_optimizer(self):
        if 0 in self.optimizers:
            return self.optimizers[0]
        else:
            return None

    @local_model_optimizer.setter
    def local_model_optimizer(self, optimizer):
        if optimizer is None:
            pass
        else:
            self.optimizers.update({0: optimizer})

    @property
    def local_pred(self):
        return self.output_tensors[0]

    @local_pred.setter
    def local_pred(self, tensor):
        self.output_tensors[0]=tensor

    @property
    def global_model(self):
        if 1 in self.models:
            return self.models[1]
        else:
            return None

    @global_model.setter
    def global_model(self, model):
        if model is None:
            pass
        else:
            self.models.update({1: model})

    @property
    def global_model_optimizer(self):
        return self.optimizers[1]

    @global_model_optimizer.setter
    def global_model_optimizer(self, optimizer):
        self.optimizers.update({1: optimizer})


    def local_forward(self):
        # args.local_model()
        pass

    @timer()
    def forward(self, model_index, **kwargs):
        logger.debug(f"model_{model_index} forward")
        self.input_tensors[model_index] = kwargs.get('inputs_embeds')
        resp = self.models[model_index](**kwargs)
        if model_index == vfl_basic_config.num_of_slice - 1:
            self.output_tensors[model_index] = resp.get('logits')
        else:
            self.output_tensors[model_index] = resp.get('inputs_embeds')
        return self._detach_tensor(resp)

    def _detach_tensor(self, dict_like: dict):
        return dict_like  # todo: need to check whether used in local mode
        # for key, value in dict_like.items():
        #     if isinstance(value, torch.Tensor) and value.requires_grad:
        #         dict_like[key] = value.detach().clone()
        #         dict_like[key].requires_grad = True
        # return dict_like

    def backward(self, model_index, **kwargs):
        logger.debug(f"model_{model_index} backward")
        self.output_tensors[model_index].backward(**kwargs)

    def optimizer_step(self, model_index):
        logger.debug(f"model_{model_index} optimize")
        self.optimizers[model_index].step()
        self.optimizers[model_index].zero_grad()

    def get_model_folder(self):
        model_folder = os.getenv('MODEL_FOLDER')
        if model_folder is None:
            raise ValueError('MODEL_FOLDER env must not be empty and should contain /')
        return model_folder

    def save_pretrained(self,model_index,model_id,model_folder=None,**kwargs):
        if model_folder is None:
            model_folder = self.get_model_folder()
        for i,m in self.models.items():
            if m and i in model_index:
                logger.debug(f"save model {i}")
                VFLPipeline.save_pretrained(model_name_or_path=os.path.join(*filter(lambda x:x is not None,[model_folder,model_id])),
                                            models={i:m},
                                            **kwargs)