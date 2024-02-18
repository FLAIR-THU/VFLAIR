import os
import sys
import numpy as np
import random
sys.path.append(os.pardir)

import torch
from torch.utils.data import DataLoader

from evaluates.attacks.attack_api import AttackerLoader
from evaluates.defenses.defense_api import DefenderLoader
from load.LoadDataset import load_dataset_per_party,load_dataset_per_party_llm, load_dataset_per_party_backdoor,load_dataset_per_party_noisysample
from load.LoadModels import load_models_per_party

from utils.noisy_label_functions import add_noise
from utils.noisy_sample_functions import noisy_sample
from utils.basic_functions import cross_entropy_for_onehot, tf_distance_cov_cor,pairwise_dist
from utils.communication_protocol_funcs import Cache

import torch
import re
import collections
from torch._six import string_classes

np_str_obj_array_pattern = re.compile(r'[SaUO]')

def my_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(my_collate_err_msg_format.format(elem.dtype))

            return my_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: my_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(my_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        # if not all(len(elem) == elem_size for elem in it):
        #     for elem in it:
        #         print('elem_size:',elem_size,'  elem:',type(elem),len(elem) )
        #     raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [my_collate(samples) for samples in transposed]

    raise TypeError(my_collate_err_msg_format.format(elem_type))

class Party(object):
    def __init__(self, args, index):
        self.name = "party#" + str(index + 1)
        self.index = index
        self.args = args
        args.need_auxiliary = 0
        args.dataset = args.dataset['dataset_name']
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
        # local model
        self.local_model = None
        self.local_model_optimizer = None
        # global_model
        self.global_model = None
        self.global_model_optimizer = None

        # attack and defense
        # self.attacker = None
        self.defender = None

        self.prepare_model(args, index)
        self.prepare_data(args, index)
        # self.prepare_attacker(args, index)
        # self.prepare_defender(args, index)

        self.local_gradient = None
        self.local_pred = None
        self.local_pred_clone = None
        
        self.origin_pred = None # for adversarial training

        self.cache = Cache()
        self.prev_batches = []
        self.num_local_updates = 0

        ####### predict results ######
        self.input_shape = None
        self.global_pred = None

        self.local_attention_mask = None # GPT2
        self.local_sequence_lengths = None # GPT2 Classification
        self.local_attention_mask = None # Llama

        # for adversarial training
        self.adversary_loss = None
        self.mapping_distance = None


    def prepare_data(self, args, index):
        (
            args,
            self.half_dim,
            train_dst,
            test_dst,
        ) = load_dataset_per_party_llm(args, index)

        self.train_data, self.train_label = train_dst
        self.test_data, self.test_label = test_dst

    def prepare_data_loader(self, need_auxiliary=0, **kwargs):
        # self.train_loader = DataLoader(self.train_dst, batch_size=batch_size) # , 
        # self.test_loader = DataLoader(self.test_dst, batch_size=batch_size) # , shuffle=True ,collate_fn=my_collate
        # if self.args.need_auxiliary == 1 and self.aux_dst != None:
        #     self.aux_loader = DataLoader(self.aux_dst, batch_size=batch_size)

        batch_size = self.args.batch_size
        self.train_loader = DataLoader(self.train_dst, batch_size=batch_size ,collate_fn=lambda x:x ) # ,
        self.test_loader = DataLoader(self.test_dst, batch_size=batch_size ,collate_fn=lambda x:x) # , shuffle=True ,collate_fn=my_collate
        if need_auxiliary == 1 and self.aux_dst != None:
            self.aux_loader = DataLoader(self.aux_dst, batch_size=batch_size ,collate_fn=lambda x:x)

    def prepare_model(self, args, index):
        # prepare model and optimizer
        (
            args,
            self.local_model,
            self.local_model_optimizer,
            self.global_model,
            self.global_model_optimizer
        ) = load_models_per_party(args, index)

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

    def give_pred(self):

        if self.args.model_type == 'Bert':
            # SequenceClassification & QuestionAnswering
            self.local_pred, self.local_attention_mask  = self.local_model(input_ids = self.local_batch_data, attention_mask = self.local_batch_attention_mask, token_type_ids=self.local_batch_token_type_ids)
            # print('self.local_model.origin_output:',self.local_model.origin_output.shape)
            # print('self.local_model.adversarial_output:',self.local_model.adversarial_output.shape) # local_pred
            if (self.args.apply_adversarial == True and (self.index in self.args.defense_configs["party"])):
                self.origin_pred = self.local_pred
                self.local_pred = self.adversarial_model(self.origin_pred)
            self.local_pred_clone = self.local_pred.detach().clone()
            self.local_attention_mask = self.local_attention_mask.detach().clone()
            return self.local_pred, self.local_pred_clone , self.local_attention_mask
        
        elif self.args.model_type == 'GPT2':
            if self.args.task_type == 'SequenceClassification':
                self.local_pred,  self.local_sequence_lengths, self.local_attention_mask = self.local_model(self.local_batch_data, attention_mask = self.local_batch_attention_mask, token_type_ids = self.local_batch_token_type_ids)
                if (self.args.apply_adversarial == True and (self.index in self.args.defense_configs["party"])):
                    self.origin_pred = self.local_pred
                    self.local_pred = self.adversarial_model(self.origin_pred)
                self.local_pred_clone = self.local_pred.detach().clone()
                self.local_attention_mask = self.local_attention_mask.detach().clone()
                return self.local_pred, self.local_pred_clone,self.local_sequence_lengths,self.local_attention_mask
            elif self.args.task_type == 'CausalLM':
                self.local_pred,  self.local_sequence_lengths, self.local_attention_mask  = self.local_model(self.local_batch_data, attention_mask = self.local_batch_attention_mask, token_type_ids = self.local_batch_token_type_ids)
                if (self.args.apply_adversarial == True and (self.index in self.args.defense_configs["party"])):
                    self.origin_pred = self.local_pred
                    self.local_pred = self.adversarial_model(self.origin_pred)
                self.local_pred_clone = self.local_pred.detach().clone()
                self.local_attention_mask = self.local_attention_mask.detach().clone()
                return self.local_pred, self.local_pred_clone,self.local_attention_mask
            elif self.args.task_type == 'QuestionAnswering':
                self.local_pred,  self.local_sequence_lengths, self.local_attention_mask  = self.local_model(self.local_batch_data, attention_mask = self.local_batch_attention_mask, token_type_ids = self.local_batch_token_type_ids)
                if (self.args.apply_adversarial == True and (self.index in self.args.defense_configs["party"])):
                    self.origin_pred = self.local_pred
                    self.local_pred = self.adversarial_model(self.origin_pred)
                self.local_pred_clone = self.local_pred.detach().clone()
                self.local_attention_mask = self.local_attention_mask.detach().clone()
                return self.local_pred, self.local_pred_clone,self.local_attention_mask

        elif self.args.model_type == 'Llama':
            if self.args.task_type == 'SequenceClassification':
                self.local_pred,  self.local_sequence_lengths, self.local_attention_mask = self.local_model(self.local_batch_data, attention_mask = self.local_batch_attention_mask)
                if (self.args.apply_adversarial == True and (self.index in self.args.defense_configs["party"])):
                    self.origin_pred = self.local_pred
                    self.local_pred = self.adversarial_model(self.origin_pred)
                self.local_pred_clone = self.local_pred.detach().clone()
                self.local_attention_mask = self.local_attention_mask.detach().clone()
                return self.local_pred, self.local_pred_clone,self.local_sequence_lengths,self.local_attention_mask
            elif self.args.task_type == 'CausalLM':
                self.local_pred,  self.local_sequence_lengths, self.local_attention_mask  = self.local_model(self.local_batch_data, attention_mask = self.local_batch_attention_mask)
                if (self.args.apply_adversarial == True and (self.index in self.args.defense_configs["party"])):
                    self.origin_pred = self.local_pred
                    self.local_pred = self.adversarial_model(self.origin_pred)
                self.local_pred_clone = self.local_pred.detach().clone()
                self.local_attention_mask = self.local_attention_mask.detach().clone()
                return self.local_pred, self.local_pred_clone,self.local_attention_mask
            elif self.args.task_type == 'QuestionAnswering':
                self.local_pred,  self.local_sequence_lengths, self.local_attention_mask  = self.local_model(self.local_batch_data, attention_mask = self.local_batch_attention_mask)
                if (self.args.apply_adversarial == True and (self.index in self.args.defense_configs["party"])):
                    self.origin_pred = self.local_pred
                    self.local_pred = self.adversarial_model(self.origin_pred)
                self.local_pred_clone = self.local_pred.detach().clone()
                self.local_attention_mask = self.local_attention_mask.detach().clone()
                return self.local_pred, self.local_pred_clone,self.local_attention_mask
    
    def give_current_lr(self):
        return (self.local_model_optimizer.state_dict()['param_groups'][0]['lr'])

    def LR_decay(self,i_epoch):
        eta_0 = self.args.main_lr
        eta_t = eta_0/(np.sqrt(i_epoch+1))
        for param_group in self.local_model_optimizer.param_groups:
            param_group['lr'] = eta_t 
            
    def obtain_local_data(self, data, local_batch_attention_mask, local_batch_token_type_ids):
        self.local_batch_data = data # input_ids
        self.local_batch_attention_mask = local_batch_attention_mask
        self.local_batch_token_type_ids = local_batch_token_type_ids
    
    def local_forward():
        # args.local_model()
        pass

    def local_backward(self,weight=None):
        # print('local_backward self.local_pred:',self.local_pred.requires_grad)

        self.num_local_updates += 1 # another update
        
        # update local model
        if self.local_model_optimizer != None:
            # adversarial training : update adversarial model
            if (self.args.apply_adversarial == True and (self.index in self.args.defense_configs["party"])):
                self.adversarial_model_optimizer.zero_grad()

                final_adversary_loss = - self.adversarial_loss + self.args.adversary_lambda * self.mapping_distance
                final_adversary_loss.backward(retain_graph=True)

                # self.weights_grad_a = torch.autograd.grad(
                #     self.local_pred,
                #     self.local_model.adversarial_model.parameters(),
                #     # self.local_model.parameters(),
                #     grad_outputs=self.local_gradient,
                #     retain_graph=True,
                # )
                # for w, g in zip(self.local_model.local_model.parameters(), self.weights_grad_a):
                #     if w.requires_grad:
                #         if w.grad != None:
                #             w.grad += g.detach()
                #         else:
                #             w.grad = g.detach()

                self.local_model_optimizer.step()

                self.adversarial_model_optimizer.step()

