import sys, os

sys.path.append(os.pardir)

import torch
import torch.nn.functional as F
from torch import autograd
import time
import numpy as np
import copy
import matplotlib.pyplot as plt
import random

from evaluates.attacks.attacker import Attacker
from models.global_models import *  # ClassificationModelHostHead, ClassificationModelHostTrainableHead
from utils.basic_functions import cross_entropy_for_onehot, append_exp_res
from dataset.party_dataset import PassiveDataset
from dataset.party_dataset import ActiveDataset
from torch.utils.data import DataLoader


def label_to_one_hot(target, num_classes=10):
    # print('label_to_one_hot:', target, type(target))
    try:
        _ = target.size()[1]
        # print("use target itself", target.size())
        onehot_target = target.type(torch.float32)
    except:
        target = torch.unsqueeze(target, 1)
        # print("use unsqueezed target", target.size())
        onehot_target = torch.zeros(target.size(0), num_classes)
        onehot_target.scatter_(1, target, 1)
    return onehot_target


def get_abs_acc(ABS_pred, target):
    sum = 0
    ABS_pred = torch.argmax(ABS_pred, dim=-1)
    for pred in ABS_pred:
        if pred == target:
            sum = sum + 1
    backdoor_acc = sum / len(ABS_pred)
    return backdoor_acc


def calc_loss(current_ADI, X_b, M, net_a, net_b, l_target):
    current_ADI.requires_grad_(True)

    ADI_pred_a = net_a(current_ADI)
    pred_b = net_b(X_b)
    ABS_pred = M([ADI_pred_a, pred_b])

    # print('current_ADI:', current_ADI.size())
    # print('ABS_pred:',ABS_pred.size()) # [32,10]
    _grad = torch.autograd.grad(outputs=ABS_pred, inputs=current_ADI, grad_outputs=torch.ones_like(ABS_pred), \
                                retain_graph=True)

    # print('_grad[0]:',_grad[0].size()) # 1 torch.Size([32, 1, 14, 28])
    Saliency_est = torch.norm(_grad[0], p=1).requires_grad_()  # tensor: [value]
    # print('Saliency_est:',Saliency_est,Saliency_est.size()) # 1 torch.Size([32, 1, 14, 28])

    l_target_list = []
    for _i in range(len(ABS_pred)):
        l_target_list.append(l_target)
    onehot_target_label = F.one_hot(torch.tensor(l_target_list).to(torch.int64), num_classes=ABS_pred.size(1))

    # print('torch.tensor(l_target_list):',torch.tensor(l_target_list).size()) # [32]
    # print('onehot_target_label:',onehot_target_label.size()) #[32,10]

    loss = cross_entropy_for_onehot(ABS_pred, onehot_target_label).requires_grad_()

    return Saliency_est, loss


class ASB(Attacker):
    def __init__(self, top_vfl, args):
        super().__init__(args)
        self.args = args
        # get information for launching BLI attack
        self.vfl_info = top_vfl.final_state
        # prepare parameters
        self.device = args.device
        self.num_classes = args.num_classes
        self.k = args.k
        self.label_size = args.num_classes

        # attack parameters
        self.party = args.attack_configs['party']  # parties that launch attacks
        self.lr = args.attack_configs['lr']
        self.epochs = args.attack_configs['epochs']  # maximum round
        self.inner_iterations = args.attack_configs['inner_iterations']
        self.batch_size = 1  # args.attack_configs['batch_size']

        self.alpha = 1
        self.beta = 1
        self.sigma = 0
        self.gamma = 0
        self.l_target = random.randint(0, args.num_classes - 1)  # target class
        print('Target Class set as ', self.l_target)

    def set_seed(self, seed=0):
        # random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def attack(self):
        self.set_seed(123)
        print_every = 1
        for ik in self.party:  # attacker party #ik
            index = ik
            assert index == 1, 'Only Active Party Launching ASB attack is supported'

            # aux_data S
            self.datanumber = 20
            aux_data_a = self.vfl_info["aux_data"][1][:self.datanumber]
            aux_data_b = self.vfl_info["aux_data"][0][:self.datanumber]
            aux_label = self.vfl_info["aux_label"][-1][:self.datanumber]
            aux_dst_a = ActiveDataset(aux_data_a, aux_label)
            aux_loader_a = DataLoader(aux_dst_a, batch_size=self.batch_size)
            aux_dst_b = PassiveDataset(aux_data_b)
            aux_loader_b = DataLoader(aux_dst_b, batch_size=self.batch_size)
            aux_loader_list = [aux_loader_b, aux_loader_a]
            # Test Data
            test_data_a = self.vfl_info['test_data'][1]  # Active Test Data
            test_data_b = self.vfl_info['test_data'][0]  # Passive Test Data
            test_label = self.vfl_info['test_label'][-1]

            # Model M
            global_model = self.vfl_info['global_model'].to(self.device)
            net_b = self.vfl_info['model'][0].to(self.device)  # passive bottom model
            net_a = self.vfl_info['model'][1].to(self.device)
            global_model.eval()
            net_a.eval()
            net_b.eval()

            # Freeze Net
            for name, parameter in global_model.named_parameters():
                parameter.requires_grad = False
            for name, parameter in net_a.named_parameters():
                parameter.requires_grad = False
            for name, parameter in net_b.named_parameters():
                parameter.requires_grad = False

            pred_a = net_a(test_data_a)
            pred_b = net_b(test_data_b)
            clean_pred = global_model([pred_a, pred_b])

            # mark = 0
            # for name, param in net_a.named_parameters():
            #     if mark == 0:
            #         print(name, param)
            #         mark = mark + 1

            # init V and Delta
            V = None
            Delta = None
            ####### Train ADI #######
            print('==== Train ADI ====')
            for i_epoch in range(self.epochs):

                for parties_data in zip(*aux_loader_list):
                    # self.gt_one_hot_label = label_to_one_hot(parties_data[self.k-1][1], self.num_classes)
                    # self.gt_one_hot_label = self.gt_one_hot_label.to(self.device)
                    self.parties_data = parties_data
                    batch_data_b = parties_data[0][0]  # Passive Party data
                    batch_data_a = parties_data[1][0]  # Active Party data

                    if V == None:
                        V = torch.zeros_like(batch_data_a).to(self.device)
                    if Delta == None:
                        Delta = torch.zeros_like(batch_data_a).requires_grad_(True).to(self.device)

                    # print('Delta1:',Delta.requires_grad)
                    Delta = Delta.detach().clone()
                    # print('Delta2:',Delta.requires_grad)
                    last_Delta = Delta.to(self.device)
                    Delta.requires_grad = True

                    # print('Delta3:',Delta.requires_grad)
                    # Find argmin of Delta
                    optimizer = torch.optim.Adam([Delta], self.lr)

                    for iter in range(self.inner_iterations):
                        optimizer.zero_grad()
                        Saliency_est, loss = calc_loss((batch_data_a + V + Delta), batch_data_b, global_model, net_a,
                                                       net_b, self.l_target)

                        Delta_loss = (self.alpha * Saliency_est + self.beta * loss).requires_grad_()

                        # print('iter=',iter,'  Delta',Delta.requires_grad,'  DeltaLoss=',Delta_loss)
                        Delta_loss.backward(retain_graph=True)
                        optimizer.step()
                    # Update
                    Delta = self.sigma * last_Delta + Delta
                    V = V + Delta

                ####### Test Performance #######
                if (i_epoch + 1) % print_every == 0:

                    with torch.no_grad():

                        pred_a = net_a(test_data_a)
                        pred_b = net_b(test_data_b)

                        ADI_test_data_a = test_data_a
                        for _i in range(len(ADI_test_data_a)):
                            ADI_test_data_a[_i] = ADI_test_data_a[_i] + V
                        ADI_pred_a = net_a(ADI_test_data_a)
                        ABS_pred = global_model([ADI_pred_a, pred_b])

                        print('clean_pred:', clean_pred.size(), torch.argmax(clean_pred, dim=-1))
                        print('ABS_pred:', ABS_pred.size(), torch.argmax(ABS_pred, dim=-1))
                        # print('pred_a,pred_b:',pred_a,pred_b)

                        backdoor_acc = get_abs_acc(ABS_pred, self.l_target)
                        # MSE.append(mse)
                        # PSNR.append(psnr)
                    print('Epoch {}% \t Delta_loss:{:.2f} backdoor_acc:{:.2f}'.format(
                        i_epoch, Delta_loss.item(), backdoor_acc))

            # mark = 0
            # for name, param in net_a.named_parameters():
            #     if mark == 0:
            #         print('after===')
            #         print(name, param)
            #         mark = mark + 1

        print(f"ASB, self.args.apply_defense={self.args.apply_defense}")
        print(f'batch_size=%d,class_num=%d, backdoor_acc=%lf' % (self.batch_size, self.num_classes, backdoor_acc))

        # append_exp_res(self.exp_res_path, exp_result)

        print("returning from ASB")
        return backdoor_acc
