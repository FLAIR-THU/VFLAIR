import os
import sys
import numpy as np
import random
sys.path.append(os.pardir)

import torch
from torch.utils.data import DataLoader

from evaluates.attacks.attack_api import AttackerLoader
from evaluates.defenses.defense_api import DefenderLoader
from load.LoadDataset import load_dataset_per_party, load_dataset_per_party_backdoor
from load.LoadModels import load_models_per_party

from utils.noisy_label_functions import add_noise
from utils.noisy_sample_functions import noisy_sample
from utils.basic_functions import cross_entropy_for_onehot, tf_distance_cov_cor,pairwise_dist


class Party(object):
    def __init__(self, args, index):
        self.name = "party#" + str(index + 1)
        self.index = index
        self.args = args
        # data for training and testing
        self.half_dim = -1
        self.train_data = None
        self.test_data = None
        self.aux_data = None
        self.train_label = None
        self.test_label = None
        self.aux_label = None
        self.train_dst = None
        self.test_dst = None
        self.aux_dst = None
        self.train_loader = None
        self.test_loader = None
        self.aux_loader = None
        self.local_batch_data = None
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

        self.prepare_data(args, index)
        self.prepare_model(args, index)
        # self.prepare_attacker(args, index)
        # self.prepare_defender(args, index)

        self.local_gradient = None
        self.local_pred = None
        self.local_pred_clone = None

    def receive_gradient(self, gradient):
        self.local_gradient = gradient
        return

    def give_pred(self):
        self.local_pred = self.local_model(self.local_batch_data)
        
        # ####### Noisy Sample #########
        if self.args.apply_ns == True and (self.index in self.args.attack_configs['party']):
            assert 'noise_lambda' in self.args.attack_configs, 'need parameter: noise_lambda'
            assert 'noise_rate' in self.args.attack_configs, 'need parameter: noise_rate'
            assert 'party' in self.args.attack_configs, 'need parameter: party'
            noise_rate = self.args.attack_configs['noise_rate'] if ('noise_rate' in self.args.attack_configs) else 0.1
            noisy_list = []
            noisy_list = random.sample(range(self.local_pred.size()[0]), (int(self.local_pred.size()[0]*noise_rate)))
            scale = self.args.attack_configs['noise_lambda']

            self.local_batch_data[noisy_list] = noisy_sample(self.local_batch_data[noisy_list],scale)
            self.local_pred = self.local_model(self.local_batch_data)
        # ####### Noisy Sample #########

        # ####### Missing Feature #######
        elif (self.args.apply_mf == True):
            self.local_pred = self.local_model(self.local_batch_data)
            assert 'missing_rate' in self.args.attack_configs, 'need parameter: missing_rate'
            assert 'party' in self.args.attack_configs, 'need parameter: party'
            attacker_id = self.args.attack_configs['party']
            missing_rate = self.args.attack_configs['missing_rate']
            
            if (self.index in attacker_id):
                missing_list = []
                missing_list = random.sample(range(self.local_pred.size()[0]), (int(self.local_pred.size()[0]*missing_rate)))
                # print("missing list:", missing_list)
                self.local_pred[missing_list] = torch.zeros(self.local_pred[missing_list].size()).to(self.args.device)
        # ####### Missing Feature #######
        else:
            self.local_pred = self.local_model(self.local_batch_data)
        self.local_pred_clone = self.local_pred.detach().clone()
        return self.local_pred, self.local_pred_clone

    def prepare_data(self, args, index):
        # prepare raw data for training
        if args.apply_backdoor == True:
            print("in party prepare_data, will prepare poison data for backdooring")
            (
                args,
                self.half_dim,
                (self.train_data, self.train_label),
                (self.test_data, self.test_label),
                (self.train_poison_data, self.train_poison_label),
                (self.test_poison_data, self.test_poison_label),
                self.train_target_list,
                self.test_target_list,
            ) = load_dataset_per_party_backdoor(args, index)
        elif args.need_auxiliary == 1:
            (
                args,
                self.half_dim,
                (self.train_data, self.train_label),
                (self.test_data, self.test_label),
                (self.aux_data, self.aux_label)
            ) = load_dataset_per_party(args, index)
            # print(f"in party load data, aux_data have length:{self.aux_data.shape}, train_data have length={self.train_data.shape}")
        else:
            (
                args,
                self.half_dim,
                (self.train_data, self.train_label),
                (self.test_data, self.test_label),
            ) = load_dataset_per_party(args, index)

    def prepare_data_loader(self, batch_size):
        self.train_loader = DataLoader(self.train_dst, batch_size=batch_size) # , shuffle=True
        self.test_loader = DataLoader(self.test_dst, batch_size=batch_size) # , shuffle=True
        if self.args.need_auxiliary == 1 and self.aux_dst != None:
            self.aux_loader = DataLoader(self.aux_dst, batch_size=batch_size)

    def prepare_model(self, args, index):
        # prepare model and optimizer
        (
            args,
            self.local_model,
            self.local_model_optimizer,
            self.global_model,
            self.global_model_optimizer,
        ) = load_models_per_party(args, index)


    # def prepare_attacker(self, args, index):
    #     if index in args.attack_configs['party']:
    #         self.attacker = AttackerLoader(args, index, self.local_model)

    # def prepare_defender(self, args, index):
    #     if index in args.attack_configs['party']:
    #         self.defender = DefenderLoader(args, index)
    def give_current_lr(self):
        return (self.local_model_optimizer.state_dict()['param_groups'][0]['lr'])

    def LR_decay(self,i_epoch):
        eta_0 = self.args.main_lr
        eta_t = eta_0/(np.sqrt(i_epoch+1))
        for param_group in self.local_model_optimizer.param_groups:
            param_group['lr'] = eta_t
        
            
    def obtain_local_data(self, data):
        self.local_batch_data = data


    def local_forward():
        # args.local_model()
        pass

    # def local_backward(self):
    #     # update local model
    #     self.local_model_optimizer.zero_grad()
    #     # ########## for passive local mid loss (start) ##########
    #     # if passive party in defense party, do
    #     if (
    #         self.args.apply_mid == True
    #         and (self.index in self.args.defense_configs["party"])
    #         and (self.index < self.args.k - 1)
    #         ):
    #         # get grad for local_model.mid_model.parameters()
    #         self.local_model.mid_loss.backward(retain_graph=True)
    #         self.local_model.mid_loss = torch.empty((1, 1)).to(self.args.device)
    #     # ########## for passive local mid loss (end) ##########
    #     self.weights_grad_a = torch.autograd.grad(
    #         self.local_pred,
    #         self.local_model.parameters(),
    #         grad_outputs=self.local_gradient,
    #         retain_graph=True,
    #     )
    #     for w, g in zip(self.local_model.parameters(), self.weights_grad_a):
    #         if w.requires_grad:
    #             w.grad = g.detach()
    #     self.local_model_optimizer.step()


    def local_backward(self):
        # update local model
        self.local_model_optimizer.zero_grad()
        # for w in self.local_model.parameters():
        #     if w.requires_grad:
        #         print("zero grad results in", w.grad) # None for all
        
        # ########## for passive local mid loss (start) ##########
        # if passive party in defense party, do
        if (
            self.args.apply_mid == True
            and (self.index in self.args.defense_configs["party"])
            and (self.index < self.args.k - 1)
            ):
            # get grad for local_model.mid_model.parameters()
            self.local_model.mid_loss.backward(retain_graph=True)
            self.local_model.mid_loss = torch.empty((1, 1)).to(self.args.device)
            # for w in self.local_model.parameters():
            #     if w.requires_grad:
            #         print("mid_loss grad results in", w.grad)
            # # get grad for local_model.local_model.parameters()
            # get grad for local_model.parameters()
            self.weights_grad_a = torch.autograd.grad(
                self.local_pred,
                # self.local_model.local_model.parameters(),
                self.local_model.parameters(),
                grad_outputs=self.local_gradient,
                retain_graph=True,
            )
            # for w, g in zip(self.local_model.local_model.parameters(), self.weights_grad_a):
            for w, g in zip(self.local_model.parameters(), self.weights_grad_a):
                if w.requires_grad:
                    if w.grad != None:
                        w.grad += g.detach()
                    else:
                        w.grad = g.detach()
            # for w in self.local_model.parameters():
            #     if w.requires_grad:
            #         print("total grad results in", w.grad)
        # ########## for passive local mid loss (end) ##########
        elif (
            self.args.apply_dcor == True
            and (self.index in self.args.defense_configs["party"])
            and (self.index < self.args.k - 1) # pasive defense
            ):
            self.weights_grad_a = torch.autograd.grad(
                self.local_pred,
                self.local_model.parameters(),
                grad_outputs=self.local_gradient,
                retain_graph=True,
            )
            for w, g in zip(self.local_model.parameters(), self.weights_grad_a):
                # print('w:',w.size(),'g:',g.size())
                if w.requires_grad:
                    w.grad = g.detach()

            ######### dCor Loss ############
            # print('dcor passive defense')
            self.distance_correlation_lambda = self.args.defense_configs['lambda']
            loss_dcor = self.distance_correlation_lambda * torch.log(tf_distance_cov_cor(self.local_pred, torch.flatten(self.local_batch_data, start_dim=1))) 
            dcor_gradient = torch.autograd.grad(
                loss_dcor, self.local_model.parameters(), retain_graph=True, create_graph=True
                )
            # print('dcor_gradient:',len(dcor_gradient),dcor_gradient[0].size())
            for w, g in zip(self.local_model.parameters(), dcor_gradient):
                # print('w:',w.size(),'g:',g.size())
                if w.requires_grad:
                    w.grad += g.detach()
            ######### dCor Loss ############
            

        else:
            self.weights_grad_a = torch.autograd.grad(
                self.local_pred,
                self.local_model.parameters(),
                grad_outputs=self.local_gradient,
                retain_graph=True,
            )
            for w, g in zip(self.local_model.parameters(), self.weights_grad_a):
                if w.requires_grad:
                    w.grad = g.detach()
        self.local_model_optimizer.step()
