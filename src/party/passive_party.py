import sys, os
sys.path.append(os.pardir)
import torch
import json
import collections
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from utils.basic_functions import cross_entropy_for_onehot, tf_distance_cov_cor
from party.party import Party
from party.llm_party import Party as Party_LLM

from dataset.party_dataset import PassiveDataset, PassiveDataset_LLM
from dataset.party_dataset import ActiveDataset
from load.LoadModels import load_models_per_party, QuestionAnsweringModelOutput
# load_models_per_party_new
from utils.squad_utils import normalize_answer, _get_best_indexes, compute_exact, compute_f1
from utils.communication_protocol_funcs import get_size_of
from evaluates.defenses.defense_api import apply_defense
from dataset.party_dataset import ActiveDataset
from utils.communication_protocol_funcs import compress_pred

from models.imagined_adversary_models import *
from models.adversarial_model import *

from models.mid_model_rapper import *

import time
import numpy as np

class PassiveParty(Party):
    def __init__(self, args, index):
        super().__init__(args, index)

    def prepare_data(self, args, index):
        super().prepare_data(args, index)
        # self.train_dst = TensorDataset(train_inputs, train_masks) # the second label is just a place holder
        # self.test_dst = TensorDataset(test_inputs, test_masks) # the second label is just a place holder
        
        self.train_dst = PassiveDataset(self.train_data)
        self.test_dst = PassiveDataset(self.test_data)
        if self.args.need_auxiliary == 1:
            self.aux_dst = ActiveDataset(self.aux_data, self.aux_label)


class PassiveParty_LLM(Party_LLM):
    def __init__(self, args, index, need_data = True):
        print(f'==== initialize PassiveParty_LLM : party {index}======')
        super().__init__(args, index, need_data = need_data)
        if args.device == 'cuda':
            cuda_id = args.gpu
            torch.cuda.set_device(cuda_id)
            print(f'running on cuda{torch.cuda.current_device()}')

        self.init_apply_defense(args.apply_defense, args.apply_adversarial, args.defense_configs, args.main_lr, args.device)
        
        self.criterion = cross_entropy_for_onehot
        
        # self.encoder = args.encoder
        self.train_index = None #args.idx_train
        self.test_index = None #args.idx_test
        
        self.device = args.device

        self.gt_one_hot_label = None
        self.clean_one_hot_label = None

        self.pred_received = []
        for _ in range(args.k):
            self.pred_received.append([])
        
        self.global_pred = None
        self.global_loss = None
        self.communication_cost = 0
        self.num_total_comms = 0
        self.current_step = 0

        self.num_labels = args.num_classes
        self.weights_grad_a = None # no gradient for model in passive party(no model update)

        self.encoder_trainable = args.encoder_trainable[index]

    def init_apply_defense(self, need_apply_defense, apply_adversarial, defense_configs, main_lr, device):
        # some defense need model, add here
        if need_apply_defense:
            if apply_adversarial and (self.index in defense_configs["party"]):
                # add adversarial model for local model
                if not 'party' in defense_configs:
                    defense_configs['party'] = [0]
                    print('[warning] default passive party selected for applying adversarial training')

            
                self.adversarial_model_lr = defense_configs['adversarial_model_lr']
                self.adversarial_model_hidden_size = defense_configs['adversarial_model_hidden_size'] if ('adversarial_model_hidden_size' in defense_configs) else 80
                if not ('adversarial_model' in defense_configs):
                    adversarial_model_name = 'Adversarial_Mapping'
                else:
                    adversarial_model_name = defense_configs['adversarial_model']
                seq_length = defense_configs['seq_length']
                embed_dim = defense_configs['embed_dim']
                # prepare adversarial model --  for adversarial training
                self.adversarial_model = globals()[adversarial_model_name](seq_length, embed_dim, self.adversarial_model_hidden_size).to(self.args.device)
                if self.local_model_optimizer == None:
                    self.local_model_optimizer = torch.optim.Adam(self.adversarial_model.parameters(), lr=self.adversarial_model_lr)
                else:
                    self.local_model_optimizer.add_param_group({'params': self.adversarial_model.parameters(), 'lr': self.adversarial_model_lr})

                # self.adversarial_model_optimizer = torch.optim.Adam(
                #             [{'params': self.adversarial_model.parameters(), 'lr': adversarial_model_lr}])

                # prepare imagined adversary --  for adversarial training
                imagined_adversary_model_name = defense_configs['imagined_adversary']
                self.imagined_adversary_hidden_size = defense_configs['imagined_adversary_hidden_size'] if ('imagined_adversary_hidden_size' in defense_configs) else 80
                self.imagined_adversary = globals()[imagined_adversary_model_name](seq_length, embed_dim, self.imagined_adversary_hidden_size).to(device)
                self.imagined_adversary_lr = defense_configs['imagined_adversary_lr']
                self.imagined_adversary_optimizer = torch.optim.Adam(list(self.imagined_adversary.parameters()), lr=self.imagined_adversary_lr)

                self.adversary_crit = nn.CrossEntropyLoss()
                self.adversary_lambda = defense_configs['lambda']
            
            elif self.args.apply_mid and (self.index in self.args.defense_configs["party"]):
                self.mid_lambda = self.args.defense_configs['lambda'] 
                self.mid_model_name = self.args.defense_configs['mid_model_name'] 
                self.mid_lr = self.args.defense_configs['lr'] 
                self.squeeze_dim = self.args.defense_configs['squeeze_dim'] if 'squeeze_dim' in self.args.defense_configs else 0

                self.mid_position = self.args.defense_configs['mid_position'] \
                if 'mid_position' in self.args.defense_configs else "out" # "inner"

                current_bottleneck_scale = int(self.args.defense_configs['bottleneck_scale']) \
                    if 'bottleneck_scale' in self.args.defense_configs else 1
        
                if 'std_shift_hyperparameter' in self.args.defense_configs:
                    std_shift_hyperparameter = int(self.args.defense_configs['std_shift_hyperparameter'])
                else:
                    std_shift_hyperparameter = 5 
                    

                seq_length = self.args.defense_configs['seq_length']
                embed_dim = self.args.defense_configs['embed_dim']


                if self.mid_position == "inner":
                    print('init defense: inner mid')
                    if 'Squeeze' in self.mid_model_name:
                        self.local_model.inner_mid_model = globals()[self.mid_model_name](seq_length,embed_dim,\
                         mid_lambda=self.mid_lambda,squeeze_dim = self.squeeze_dim ,bottleneck_scale=current_bottleneck_scale, std_shift=std_shift_hyperparameter).to(self.args.device)
                    else:
                        self.local_model.inner_mid_model = globals()[self.mid_model_name](seq_length,embed_dim,\
                    mid_lambda=self.mid_lambda,bottleneck_scale=current_bottleneck_scale, std_shift=std_shift_hyperparameter).to(self.args.device)
                    
                    if self.local_model_optimizer == None:
                        self.local_model_optimizer = torch.optim.Adam(self.local_model.inner_mid_model.parameters(), lr=self.mid_lr)
                    else:
                        self.local_model_optimizer.add_param_group({'params': self.local_model.inner_mid_model.parameters(),\
                         'lr': self.mid_lr})

                else:
                    print('init defense: out mid')
                    print(self.mid_model_name)
                    if 'Squeeze' in self.mid_model_name:
                        print('Squeeze')
                        self.mid_model = globals()[self.mid_model_name](seq_length,embed_dim,\
                         mid_lambda=self.mid_lambda,squeeze_dim = self.squeeze_dim ,bottleneck_scale=current_bottleneck_scale, std_shift=std_shift_hyperparameter).to(self.args.device)
                    else:
                        self.mid_model = globals()[self.mid_model_name](seq_length,embed_dim,\
                        mid_lambda=self.mid_lambda,bottleneck_scale=current_bottleneck_scale, std_shift=std_shift_hyperparameter).to(self.args.device)

                    if self.local_model_optimizer == None:
                        self.local_model_optimizer = torch.optim.Adam(self.mid_model.parameters(), lr=self.mid_lr)
                    else:
                        self.local_model_optimizer.add_param_group({'params': self.mid_model.parameters(), 'lr': self.mid_lr})

                print(f'self.mid_model_name:{self.mid_model_name}')

    def eval(self, **kwargs):
        self.local_model.eval()

    def prepare_data(self, args, index):
        if not args.dataset:
            return None
        super().prepare_data(args, index) # Party_llm's prepare_data
 
        self.train_dst = PassiveDataset_LLM(args, self.train_data, self.train_label, 'train')
        self.test_dst = PassiveDataset_LLM(args, self.test_data, self.test_label, 'test')

    def update_local_pred(self, pred):
        self.pred_received[self.args.k-1] = pred
    
    def receive_pred(self, pred, giver_index):
        self.pred_received[giver_index] = pred

    def cal_global_gradient(self, global_loss, global_pred):
        # print('Passive Party cal global_gradients:')
        # print('Global Loss=',global_loss)

        if self.args.task_type == 'QuestionAnswering':
            _gradients_start = torch.autograd.grad(global_loss, global_pred.start_logits, retain_graph=True)
            _gradients_end = torch.autograd.grad(global_loss, global_pred.end_logits, retain_graph=True)
            global_gradients = _gradients_end+_gradients_start
            global_gradients_clone = global_gradients[0].detach().clone()
            global_gradients_clone = global_gradients_clone/2
            self.global_gradients = global_gradients_clone
        else:
            global_gradients = torch.autograd.grad(global_loss, global_pred.logits, retain_graph=True)
            global_gradients_clone = global_gradients[0].detach().clone()
            self.global_gradients = global_gradients_clone

        return global_gradients_clone

    def cal_loss(self, pred, test=False):
        gt_one_hot_label = self.gt_one_hot_label # label
        
        # ########### Normal Loss ###############
        if self.args.task_type == 'SequenceClassification':
            # loss = self.criterion(pred, gt_one_hot_label)
            pooled_logits = pred.logits
            labels = gt_one_hot_label

            # GPT2
            if self.num_labels == 1:
                self.problem_type = "regression"
            else:
                self.problem_type = "single_label_classification"

            if self.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                print(pooled_logits.view(-1, self.num_labels).shape)
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels) #labels.view(-1)
            # elif self.problem_type == "multi_label_classification":
            #     loss_fct = BCEWithLogitsLoss()
            #     loss = loss_fct(pooled_logits, labels)

        elif self.args.task_type == 'CausalLM':
            #  ? pred: generated text ids [bs, new_token_num]
            lm_logits = pred.logits # [bs, seq_len, vocab_size]
            next_token_logits = lm_logits[:,-1,:]

            labels = torch.tensor(gt_one_hot_label).squeeze()
            label_id = torch.tensor(labels.long()).to(self.args.device)

            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss_fct(next_token_logits, label_id)
            # print('loss:', loss)

        elif self.args.task_type == 'QuestionAnswering':
            start_logits = pred.start_logits
            end_logits = pred.end_logits

            # golden_start_positions, golden_end_positions = gt_one_hot_label[0] # bs *[start_id, end_id]  bs=1
            
            golden_start_positions = torch.tensor( [gt_one_hot_label[i][0] for i in range(gt_one_hot_label.shape[0])] )
            golden_end_positions = torch.tensor( [gt_one_hot_label[i][1] for i in range(gt_one_hot_label.shape[0])] )

            golden_start_positions = golden_start_positions.squeeze().long().to(start_logits.device) # .unsqueeze(0)
            golden_end_positions = golden_end_positions.squeeze().long().to(end_logits.device)
            # print('golden_start_positions golden_end_positions:',golden_start_positions.shape, golden_end_positions.shape)

            loss = None

            if len(golden_start_positions.size()) > 1:
                golden_start_positions = golden_start_positions.squeeze(-1).to(start_logits.device)
            if len(golden_end_positions.size()) > 1:
                golden_end_positions = golden_end_positions.squeeze(-1).to(end_logits.device)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            # print('ignored_index:',ignored_index)
            golden_start_positions = golden_start_positions.clamp(0, ignored_index)
            golden_end_positions = golden_end_positions.clamp(0, ignored_index)

            # print('start_logits end_logits:',start_logits.shape, end_logits.shape)
            # print('after clamp golden_start_positions golden_end_positions:',golden_start_positions.shape, golden_end_positions.shape)
            
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)

            start_loss = loss_fct(start_logits, golden_start_positions)
            end_loss = loss_fct(end_logits, golden_end_positions)
            loss = (start_loss + end_loss) / 2

            # print('start_loss:',start_loss,' end_loss:',end_loss,' loss:',loss)

        else:
            assert 1>2 , 'Task type not supported'
        
        self.global_loss = loss

        # ########### Defense on Loss ###############
        if self.args.apply_adversarial and (self.index in self.args.defense_configs["party"]):
            intermediate = self.local_pred # pred after adversarial model: bs, seq, embed_dim768
            adversary_recovered_embedding = self.imagined_adversary(intermediate)

            real_embedding =  self.local_model.embedding_output
            self.adversary_attack_loss = self.adversary_crit(adversary_recovered_embedding, real_embedding) / intermediate.shape[0]
            
            # avrage mapping distance on bs*seq_len   self.origin_pred: bs, seq_len, embed_dim
            self.mapping_distance = torch.norm( self.origin_pred - self.local_pred , p=2) / (self.origin_pred.shape[0]*self.origin_pred.shape[1])

            # print(f'main_loss={self.global_loss},mapping_distance={self.mapping_distance},adversary_attack_loss={self.adversary_attack_loss}')

            # renew global loss function : loss used to update adversarial model mapping
            self.adversarial_model_loss =   self.adversary_lambda * self.mapping_distance - self.adversary_attack_loss
            self.global_loss = self.global_loss + self.adversarial_model_loss 

        elif self.args.apply_mid == True and (self.index in self.args.defense_configs['party']):
            # print(f'main_loss={self.global_loss},mid_loss={self.mid_loss}')
            # print('self.mid_loss.requires_grad:',self.mid_loss.requires_grad)
            self.global_loss = self.global_loss + self.mid_loss
        # ########### Defense on Loss ###############

        return self.global_loss

    def gradient_calculation(self, pred_list, loss):
        pred_gradients_list = []
        pred_gradients_list_clone = []
        for ik in range(self.args.k):
            pred_gradients_list.append(torch.autograd.grad(loss, pred_list[ik], retain_graph=True, create_graph=True))
            # print(f"in gradient_calculation, party#{ik}, loss={loss}, pred_gradeints={pred_gradients_list[-1]}")
            pred_gradients_list_clone.append(pred_gradients_list[ik][0].detach().clone())
        # self.global_backward(pred, loss)
        return pred_gradients_list, pred_gradients_list_clone
    
    def update_local_gradient(self, gradient):
        self.local_gradient = gradient

    def global_LR_decay(self,i_epoch):
        if self.global_model_optimizer != None: 
            eta_0 = self.args.main_lr
            eta_t = eta_0/(np.sqrt(i_epoch+1))
            for param_group in self.global_model_optimizer.param_groups:
                param_group['lr'] = eta_t

    def local_backward(self): # model head 1
        # print(' === passive local backward === ')

        self.num_local_updates += 1 # another update

        # adversarial training : update adversarial model
        if (self.args.apply_adversarial == True and (self.index in self.args.defense_configs["party"])):
            # imagined_adversary update
            self.imagined_adversary_optimizer.zero_grad()
            self.adversary_attack_loss.backward(retain_graph = True)
            self.imagined_adversary_optimizer.step()

            self.local_model_optimizer.zero_grad()
            # local model trainable part
            local_model_params = list(self.adversarial_model.parameters())
            for param in self.local_model.parameters():
                if param.requires_grad:
                    local_model_params.append(param)
            weights_grad_a = torch.autograd.grad(
                self.local_pred,
                local_model_params,
                grad_outputs=self.local_gradient,
                retain_graph=True,
                #allow_unused = True
            )
            for w, g in zip(local_model_params, weights_grad_a):
                if w.requires_grad:
                    if w.grad != None:
                        w.grad += g.detach()
                    else:
                        w.grad = g.detach()

            weights_grad_a = torch.autograd.grad(
                self.adversarial_model_loss,
                self.adversarial_model.parameters(),
                retain_graph=True,
            )
            for w, g in zip(self.adversarial_model.parameters(), weights_grad_a):
                if w.requires_grad:
                    if w.grad != None:
                        w.grad += g.detach()
                    else:
                        w.grad = g.detach()

            self.local_model_optimizer.step()

        elif (self.args.apply_mid == True and (self.index in self.args.defense_configs["party"])
            and (self.index < self.args.k - 1) and self.mid_position == "out"):
            # print('before')
            # mark = 0
            # for name, param in self.mid_model.named_parameters():
            #     if mark == 0:
            #         print(name, param)
            #         mark = mark + 1

            self.local_model_optimizer.zero_grad()# self.mid_model_optimizer.zero_grad()

            # update mid+local_model with mid_loss
            self.mid_loss.backward(retain_graph=True)

            # update mid with global_loss
            weights_grad_a = torch.autograd.grad(
                self.local_pred,
                self.mid_model.parameters(),
                grad_outputs=self.local_gradient,
                retain_graph=True,
            )
            for w, g in zip(self.mid_model.parameters(), weights_grad_a):
                if w.requires_grad:
                    if w.grad != None:
                        w.grad += g.detach()
                    else:
                        w.grad = g.detach()
        
            # update local model trainable part with global_loss
            local_model_params = []
            for param in self.local_model.parameters():
                if param.requires_grad:
                    local_model_params.append(param)
            if len(local_model_params) > 0:
                self.weights_grad_a = torch.autograd.grad(
                    self.local_pred,
                    local_model_params, # self.local_model.parameters()
                    grad_outputs=self.local_gradient,
                    retain_graph=True,
                )
                for w, g in zip(local_model_params, self.weights_grad_a):
                    if w.requires_grad:
                        if w.grad != None:
                            w.grad += g.detach()
                        else:
                            w.grad = g.detach()
            
            self.local_model_optimizer.step()
     
        elif (self.args.apply_mid == True and (self.index in self.args.defense_configs["party"])
            and (self.index < self.args.k - 1) and self.mid_position == "inner"):

            self.local_model_optimizer.zero_grad()# self.mid_model_optimizer.zero_grad()
            
            ###########  update mid_model  ###########
            # with mid_loss.backward
            self.mid_loss.backward(retain_graph=True)
            # with global_loss -> local_gradient
            self.weights_grad_a = torch.autograd.grad(
                self.local_pred,
                self.local_model.inner_mid_model.parameters(),
                grad_outputs=self.local_gradient,
                retain_graph=True,
            )
            for w, g in zip(self.local_model.inner_mid_model.parameters(), self.weights_grad_a):
                if w.requires_grad:
                    if w.grad != None:
                        w.grad += g.detach()
                    else:
                        w.grad = g.detach()
            
            ###########  update local encoder  ###########
            local_model_params = []
            for param in self.local_model.parameters():
                if param.requires_grad:
                    local_model_params.append(param)
            if len(local_model_params) > 0:
                self.weights_grad_a = torch.autograd.grad(
                    self.local_pred,
                    local_model_params, # self.local_model.parameters()
                    grad_outputs=self.local_gradient,
                    retain_graph=True,
                )
                for w, g in zip(local_model_params, self.weights_grad_a):
                    if w.requires_grad:
                        if w.grad != None:
                            w.grad += g.detach()
                        else:
                            w.grad = g.detach()
            self.local_model_optimizer.step()

            # print('after')
            # mark = 0
            # for name, param in self.local_model.inner_mid_model.named_parameters():
            #     if mark == 0:
            #         print(name, param)
            #         mark = mark + 1

        else: # W/O Defense
            if self.local_model_optimizer != None:
                self.local_model_optimizer.zero_grad()

                # local model trainable part
                local_model_params = []
                for param in self.local_model.parameters():
                    if param.requires_grad:
                        local_model_params.append(param)

                if len(local_model_params) > 0:
                    self.weights_grad_a = torch.autograd.grad(
                        self.local_pred,
                        local_model_params, # self.local_model.parameters()
                        grad_outputs=self.local_gradient,
                        retain_graph=True,
                    )
                    for w, g in zip(local_model_params, self.weights_grad_a):
                        if w.requires_grad:
                            if w.grad != None:
                                w.grad += g.detach()
                            else:
                                w.grad = g.detach()

                    self.local_model_optimizer.step()


    # def calculate_gradient_each_class(self, global_pred, local_pred_list, test=False):
    #     # print(f"global_pred.shape={global_pred.size()}") # (batch_size, num_classes)
    #     self.gradient_each_class = [[] for _ in range(global_pred.size(1))]
    #     one_hot_label = torch.zeros(global_pred.size()).to(global_pred.device)
    #     for ic in range(global_pred.size(1)):
    #         one_hot_label *= 0.0
    #         one_hot_label[:,ic] += 1.0
    #         if self.train_index != None: # for graph data
    #             if test == False:
    #                 loss = self.criterion(global_pred[self.train_index], one_hot_label[self.train_index])
    #             else:
    #                 loss = self.criterion(global_pred[self.test_index], one_hot_label[self.test_index])
    #         else:
    #             loss = self.criterion(global_pred, one_hot_label)
    #         for ik in range(self.args.k):
    #             self.gradient_each_class[ic].append(torch.autograd.grad(loss, local_pred_list[ik], retain_graph=True, create_graph=True))
    #     # end of calculate_gradient_each_class, return nothing

    def launch_defense(self, gradients_list, _type):

        if _type == 'gradients':
            return apply_defense(self.args, _type, gradients_list)
        elif _type == 'pred':
            return apply_defense(self.args, _type, gradients_list)
        else:
            # further extention
            return gradients_list

    def apply_defense_on_transmission(self, pred_detach):
        # print('apply_defense_on_transmission')
        # print('pred_detach:',type(pred_detach))
        # print(pred_detach.shape)
        ########### Defense applied on pred transmit ###########
        if self.args.apply_defense == True and self.args.apply_dp == True:
            pred_detach_list = self.launch_defense(pred_detach, "pred")
            pred_detach = torch.stack(pred_detach_list)
            # print('after:',pred_detach.shape)
            
        return pred_detach

    def apply_communication_protocol_on_transmission(self, pred_detach):
        ########### communication_protocols ###########
        if self.args.communication_protocol in ['Quantization','Topk']:
            pred_detach = compress_pred( self.args ,pred_detach , self.parties[ik].local_gradient,\
                            self.current_epoch, self.current_step).to(self.args.device)
        return pred_detach

