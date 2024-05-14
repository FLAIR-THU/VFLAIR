import sys, os

sys.path.append(os.pardir)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
import numpy as np
import copy
import pickle
import matplotlib.pyplot as plt
import itertools

from evaluates.attacks.attacker import Attacker
from models.global_models import *
from utils.basic_functions import cross_entropy_for_onehot, append_exp_res
from dataset.party_dataset import PassiveDataset
from dataset.party_dataset import ActiveDataset

from evaluates.defenses.defense_functions import LaplaceDP_for_pred, GaussianDP_for_pred


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


class custom_AE(nn.Module):
    def __init__(self, latent_dim, target_dim):
        super(custom_AE, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_dim, 600),
            nn.LayerNorm(600),
            nn.ReLU(),

            nn.Linear(600, 200),
            nn.LayerNorm(200),
            nn.ReLU(),

            nn.Linear(200, 100),
            nn.LayerNorm(100),
            nn.ReLU(),

            nn.Linear(100, target_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        return self.net(x)


class ResSFL(Attacker):
    def __init__(self, top_vfl, args):
        super().__init__(args)
        self.args = args
        self.vfl_info = top_vfl.final_state
        self.vfl_first_epoch = top_vfl.first_epoch_state
        # prepare parameters
        self.device = args.device
        self.num_classes = args.num_classes
        self.label_size = args.num_classes
        self.k = args.k
        self.batch_size = args.batch_size

        # attack configs
        self.party = args.attack_configs['party']  # parties that launch attacks
        self.lr = args.attack_configs['lr']
        self.epochs = args.attack_configs['epochs']
        self.attack_batch_size = args.attack_configs['batch_size']

        self.criterion = cross_entropy_for_onehot

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
        for attacker_ik in self.party:  # attacker party #attacker_ik
            # assert attacker_ik == (self.k - 1), 'Only Active party launch feature inference attack'
            attacked_party_list = [ik for ik in range(self.k)]
            attacked_party_list.remove(attacker_ik)

            index = attacker_ik
            # collect necessary information
            net_b = [self.vfl_info['final_model'][ik].to(self.device) for ik in attacked_party_list]  # Passive
            net_a = self.vfl_info['final_model'][attacker_ik].to(self.device)  # Active
            global_model = self.vfl_info['final_global_model'].to(self.device)
            global_model.eval()
            net_b = [net.eval() for net in net_b]
            net_a.eval()

            batch_size = self.attack_batch_size

            # Test Data
            test_data_a = self.vfl_info['test_data'][attacker_ik]  # Active Test Data
            test_data_b = [torch.tensor(self.vfl_info['test_data'][ik], dtype=torch.float32) for ik in
                           attacked_party_list]  # Passive Test Data

            # Train with Aux Dataset
            aux_data_a = self.vfl_info["aux_data"][attacker_ik]
            aux_data_b = [self.vfl_info["aux_data"][ik] for ik in attacked_party_list]
            aux_label = self.vfl_info["aux_label"][-1]
            aux_dst_a = ActiveDataset(aux_data_a, aux_label)
            aux_loader_a = DataLoader(aux_dst_a, batch_size=batch_size)
            aux_dst_b = [PassiveDataset(aux_data_b[ik]) for ik in range(len(aux_data_b))]
            aux_loader_b = [DataLoader(aux_dst_b[ik], batch_size=batch_size) for ik in range(len(aux_data_b))]
            aux_loader_list = aux_loader_b + [aux_loader_a]

            # Initiate Decoder
            '''
            test_data_b = [[batchsize,1,28,14],]
            '''
            # if self.args.dataset in ['nuswide','breast_cancer_diagnose','diabetes','adult_income','criteo']:
            #     dim_a = test_data_a.size()[1]
            #     dim_b = test_data_b.size()[1]
            # else: # mnist cifar
            #     dim_a = test_data_a.size()[1]*test_data_a.size()[2]*test_data_a.size()[3]
            #     dim_b = test_data_b.size()[1]*test_data_b.size()[2]*test_data_b.size()[3]

            criterion = nn.MSELoss()

            if self.args.dataset == 'cifar10' or self.args.dataset == 'cifar100':
                decoder_list = [custom_AE(self.args.model_list[str(ik)]['output_dim'],
                                          3 * self.args.model_list[str(ik)]['input_dim']).to(self.device) for ik in
                                attacked_party_list]
            else:  # mnist
                decoder_list = [custom_AE(self.args.model_list[str(ik)]['output_dim'],
                                          self.args.model_list[str(ik)]['input_dim']).to(self.device) for ik in
                                attacked_party_list]

            # custom_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim, output_dim=32).to(self.device)
            optimizer_list = [torch.optim.Adam(decoder.parameters(), lr=self.lr) for decoder in decoder_list]

            feature_dimention_list = [self.args.model_list[str(ik)]['input_dim'] for ik in attacked_party_list]

            print('========= Feature Inference Training ========')
            for i_epoch in range(self.epochs):
                ####### Train Generator for each attacked party #######
                decoder_list = [decoder.train() for decoder in decoder_list]
                for parties_data in zip(*aux_loader_list):
                    self.gt_one_hot_label = label_to_one_hot(parties_data[self.k - 1][1], self.num_classes)
                    self.gt_one_hot_label = self.gt_one_hot_label.to(self.device)
                    self.parties_data = parties_data
                    batch_data_b = [parties_data[ik][0] for ik in range(len(parties_data) - 1)]  # Passive Party data
                    batch_data_a = parties_data[-1][0]  # Active Party data

                    # target img
                    img = batch_data_b

                    # Known Information : intermediate representation
                    with torch.no_grad():
                        # ir = net_b(batch_data_b)
                        # print('batch_data_b[ik]:',batch_data_b[0].size())

                        ir = [net_b[ik](batch_data_b[ik]) for ik in range(len(parties_data) - 1)]
                        ####### DP Defense On FR ########
                        if self.args.apply_dp == True:
                            if 'laplace' in self.args.defense_name.casefold():
                                ir = [LaplaceDP_for_pred(self.args, [ir[ik]]) for ik in range(len(ir))]
                                # ir = LaplaceDP_for_pred(self.args, ir)
                            elif 'gaussian' in self.args.defense_name.casefold():
                                ir = [GaussianDP_for_pred(self.args, [ir[ik]]) for ik in range(len(ir))]
                                # ir = GaussianDP_for_pred(self.args, ir)
                        ####### DP Defense On FR ########

                    output = []
                    for ik in range(len(batch_data_b)):  # should have k-1 parties, except the attacker
                        img[ik], ir[ik] = img[ik].type(torch.FloatTensor), ir[ik].type(torch.FloatTensor)
                        img[ik], ir[ik] = Variable(img[ik]).to(self.device), Variable(ir[ik]).to(self.device)

                        # recovered image
                        output.append(decoder_list[ik](ir[ik]))  # torch.Size([10])
                        # print('ir:',ir[ik].size()) # [32,  10]
                        # print('img:',img[ik].size()) # [32,  3,16,32]
                        # print('output:',output[ik].size()) # [32,512]

                        img[ik] = img[ik].reshape(output[ik].size())

                        train_loss = criterion(output[ik], img[ik])

                        optimizer_list[ik].zero_grad()
                        train_loss.backward()
                        optimizer_list[ik].step()

                ####### Test Performance of Generator #######
                if (i_epoch + 1) % print_every == 0:
                    mse_list = []
                    rand_mse_list = []
                    decoder_list = [decoder.eval() for decoder in decoder_list]
                    with torch.no_grad():
                        # test_data_a = self.vfl_first_epoch['data'][1][0] # active party 
                        # test_data_b = self.vfl_first_epoch['data'][0][0] # passive party 
                        # # pred with possible defense 
                        # test_pred_a = self.vfl_first_epoch['predict'][1]
                        # test_pred_b = self.vfl_first_epoch['predict'][0]
                        # test_global_pred = self.vfl_first_epoch['global_pred'].to(self.device)

                        img = test_data_b  # target img
                        # test_pred_b = net_b(test_data_b)

                        if self.args.dataset == 'cifar10' or self.args.dataset == 'cifar100':
                            test_data_b[ik] = test_data_b[ik].reshape([len(test_data_b[ik]), 3, 16, 32])

                        test_pred_b = [net_b[ik](test_data_b[ik]) for ik in range(len(test_data_b))]
                        ir = test_pred_b
                        ####### DP Defense On FR ########
                        if self.args.apply_dp == True:
                            if 'laplace' in self.args.defense_name.casefold():
                                ir = [LaplaceDP_for_pred(self.args, [ir[ik]]) for ik in range(len(ir))]
                                # ir = LaplaceDP_for_pred(self.args, ir)
                            elif 'gaussian' in self.args.defense_name.casefold():
                                ir = [GaussianDP_for_pred(self.args, [ir[ik]]) for ik in range(len(ir))]
                                # ir = GaussianDP_for_pred(self.args, ir)
                        ####### DP Defense On FR ########

                        output = []
                        for ik in range(len(test_data_b)):  # should have k-1 parties, except the attacker
                            img[ik], ir[ik] = img[ik].type(torch.FloatTensor), ir[ik].type(torch.FloatTensor)
                            img[ik], ir[ik] = Variable(img[ik]).to(self.device), Variable(ir[ik]).to(self.device)

                            output.append(decoder_list[ik](ir[ik]))  # reconstruction result

                            img[ik] = img[ik].reshape(output[ik].size())
                            rand_img = torch.randn(img[ik].size()).to(self.device)
                            _mse = criterion(output[ik], img[ik])
                            _rand_mse = criterion(rand_img, img[ik])
                            mse_list.append(_mse)
                            rand_mse_list.append(_rand_mse)
                            output[ik] = output[ik].reshape(img[ik].size())
                        mse = torch.sum(torch.tensor(mse_list) * torch.tensor(feature_dimention_list)) / torch.sum(
                            torch.tensor(feature_dimention_list))
                        rand_mse = torch.sum(
                            torch.tensor(rand_mse_list) * torch.tensor(feature_dimention_list)) / torch.sum(
                            torch.tensor(feature_dimention_list))
                    if i_epoch == self.epochs - 1:
                        output = torch.stack(output)
                        torch.save(output, f"./exp_result/ressfl/{self.args.defense_name}.pkl")
                    print('Epoch {}% \t train_loss:{:.2f} mse:{:.4}, mse_reduction:{:.2f}'.format(
                        i_epoch, train_loss.item(), mse, rand_mse - mse))

            ####### Clean ######
            for decoder_ in decoder_list:
                del (decoder_)
            del (aux_dst_a)
            del (aux_loader_a)
            del (aux_dst_b)
            del (aux_loader_b)
            del (aux_loader_list)
            del (test_data_b)
            del (test_data_a)

            print(f"ResSFL, if self.args.apply_defense={self.args.apply_defense}")
            print(f'batch_size=%d,class_num=%d,attacker_party_index=%d,mse=%lf' % (
            self.batch_size, self.label_size, index, mse))

        print("returning from ResSFL")
        return rand_mse, mse
