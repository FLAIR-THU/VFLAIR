import sys, os
sys.path.append(os.pardir)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
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

class Generator(nn.Module):
    def __init__(self, latent_dim, target_dim):
        super().__init__()
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
        return self.net(x)


class GenerativeRegressionNetwork(Attacker):
    def __init__(self, top_vfl, args):
        super().__init__(args)
        self.args = args
        self.vfl_info = top_vfl.final_state
        # prepare parameters
        self.device = args.device
        self.num_classes = args.num_classes
        self.label_size = args.num_classes
        self.k = args.k
        self.batch_size = args.batch_size

        # attack configs
        self.party = args.attack_configs['party'] # parties that launch attacks
        self.lr = args.attack_configs['lr']
        self.epochs = args.attack_configs['epochs']
        self.data_number = args.attack_configs['data_number'] #2048
        self.grn_batch_size = args.attack_configs['batch_size'] #64
        self.unknownVarLambda = 0.25 # by default
        
        self.criterion = cross_entropy_for_onehot
   
    
    def set_seed(self,seed=0):
        # random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def MSE_PSNR(self, batch_real_image, batch_dummy_image):
        '''
        compute MSE and PSNR
        :param batch_real_image:
        :param batch_dummy_image:
        :return:
        '''
        # print(batch_real_image.size(),batch_dummy_image.size())
        batch_real_image = batch_real_image.reshape(batch_dummy_image.size())
        mse = torch.mean((batch_real_image-batch_dummy_image)**2)
        psnr = 20 * torch.log10(1/torch.sqrt(mse))
        return mse.cpu().numpy(), psnr.cpu().numpy()

    def attack(self):
        self.set_seed(123)
        print_every = 1
        for ik in self.party: # attacker party #ik
            assert ik == 1, 'Only Active party launch feature inference attack'
            index = ik
            # collect necessary information
            net_b = self.vfl_info['model'][0].to(self.device) # Passive
            net_a = self.vfl_info['model'][1].to(self.device) # Active
            global_model = self.vfl_info['global_model'].to(self.device)
            global_model.eval()
            net_b.eval()
            net_a.eval()

            # Init Generator
            last_batch_data_a = self.vfl_info['data'][1][0] # active party 
            last_batch_data_b = self.vfl_info['data'][0][0] # passive party 
            if self.args.dataset == 'nuswide':
                print('dim_a:',last_batch_data_a.size())
                print('dim_b:',last_batch_data_b.size())
                dim_a = last_batch_data_a.size()[1]
                dim_b = last_batch_data_b.size()[1]
            else: # mnist cifar
                dim_a = last_batch_data_a.size()[1]*last_batch_data_a.size()[2]*last_batch_data_a.size()[3]
                dim_b = last_batch_data_b.size()[1]*last_batch_data_b.size()[2]*last_batch_data_b.size()[3]
            self.netG = Generator(dim_a+dim_b, dim_b)
            self.netG = self.netG.to(self.device)
            self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr = self.lr)
            '''
            "aux_data": [copy.deepcopy(self.parties[ik].aux_data) for ik in range(self.k)],
            "train_data": [copy.deepcopy(self.parties[ik].train_data) for ik in range(self.k)],
            "test_data": [copy.deepcopy(self.parties[ik].test_data) for ik in range(self.k)],
            "aux_label": [copy.deepcopy(self.parties[ik].aux_label) for ik in range(self.k)],
            "train_label": [copy.deepcopy(self.parties[ik].train_label) for ik in range(self.k)],
            "test_label": [copy.deepcopy(self.parties[ik].test_label) for ik in range(self.k)],
            "aux_loader": [copy.deepcopy(self.parties[ik].aux_loader) for ik in range(self.k)],
            "train_loader": [copy.deepcopy(self.parties[ik].train_loader) for ik in range(self.k)],
            "test_loader": [copy.deepcopy(self.parties[ik].test_loader) for ik in range(self.k)],
            "batchsize": self.args.batch_size,
            "num_classes": self.args.num_classes
            '''
            # Train with all dataset
            # train_loader_list = self.vfl_info['train_loader'] 
            # test_loader_list = self.vfl_info['test_loader']
            # test_data_a =  self.vfl_info['test_data'][1] # Active Test Data
            # test_data_b =  self.vfl_info['test_data'][0] # Passive Test Data
            
            # Train with partial dataset
            data_number = self.data_number
            batch_size = self.grn_batch_size

            test_data_a =  self.vfl_info['test_data'][1][:int(0.2*data_number)] # Active Test Data
            test_data_b =  self.vfl_info['test_data'][0][:int(0.2*data_number)] # Passive Test Data

            train_data_list = [_data[:data_number] for _data in self.vfl_info['train_data']]
            train_label = self.vfl_info['train_label'][1][:data_number]
            # test_data_list = [_data[:int(0.2*data_number)] for _data in self.vfl_info['test_data']]
            # test_label = self.vfl_info['test_label'][1][:int(0.2*data_number)]
            train_loader_a = DataLoader(ActiveDataset(train_data_list[1], train_label), batch_size=batch_size)
            train_loader_b = DataLoader(PassiveDataset(train_data_list[0]), batch_size=batch_size)
            # test_loader_a = DataLoader(ActiveDataset(test_data_list[1], test_label), batch_size=batch_size)
            # test_loader_b = DataLoader(PassiveDataset(test_data_list[0]), batch_size=batch_size)
            train_loader_list = [train_loader_b,train_loader_a]
            #test_loader_list = [test_loader_b,test_loader_a]

            print('========= Feature Inference Training ========')
            for i_epoch in range(self.epochs):
                self.netG.train()
                for parties_data in zip(*train_loader_list):
                    self.gt_one_hot_label = label_to_one_hot(parties_data[self.k-1][1], self.num_classes)
                    self.gt_one_hot_label = self.gt_one_hot_label.to(self.device)
                    self.parties_data = parties_data
                    batch_data_b = parties_data[0][0] # Passive Party data
                    batch_data_a = parties_data[1][0] # Active Party data

                    self.optimizerG.zero_grad()
                    # generate "fake inputs"
                    noise_data_b = torch.randn(batch_data_b.size()).to(self.device) # attack from passive side, data_b is at active side need to be generated from noise at passive side
                    # print('batch_data_b:',batch_data_b.size())
                    # print('torch.cat:',batch_data_a.size(),noise_data_b.size())
                    # print('cat:',torch.cat((batch_data_a,noise_data_b),dim=1).size())
                    if self.args.dataset == 'nuswide':
                        generated_data_b = self.netG(torch.cat((batch_data_a,noise_data_b),dim=1))
                    else:
                        generated_data_b = self.netG(torch.cat((batch_data_a,noise_data_b),dim=2))
                    generated_data_b = generated_data_b.reshape(batch_data_b.size())
                    # compute logits of generated/real data
                    pred_a = net_a(batch_data_a)
                    pred_b = net_b(batch_data_b)
                    dummy_pred_b = net_b(generated_data_b)
                    # aggregate logits of clients
                    real_pred = global_model([pred_a, pred_b])
                    dummy_pred = global_model([pred_a, dummy_pred_b])

                    unknown_var_loss = 0.0
                    for i in range(generated_data_b.size(0)):
                        unknown_var_loss = unknown_var_loss + (generated_data_b[i].var())     # var() unknown
                    # print(unknown_var_loss, ((pred.detach() - ground_truth_pred.detach())**2).sum())
                    # print((pred.detach() - ground_truth_pred.detach())[-5:])
                    loss = (((F.softmax(dummy_pred,dim=-1).detach() - F.softmax(real_pred,dim=-1).detach())**2).sum() + self.unknownVarLambda * unknown_var_loss * 1000)
                    loss.backward()
                    self.optimizerG.step() 


                ####### Test Performance of Generator #######
                if (i_epoch + 1) % print_every == 0:
                    self.netG.eval()
                    # MSE = []
                    # PSNR = []
                    with torch.no_grad():
                        noise_data_b = torch.randn(test_data_b.size()).to(self.device)
                        
                        if self.args.dataset == 'nuswide':
                            generated_data_b = self.netG(torch.cat((test_data_a,noise_data_b),dim=1))
                        else:
                            generated_data_b = self.netG(torch.cat((test_data_a,noise_data_b),dim=2))
                        
                        mse, psnr = self.MSE_PSNR(test_data_b, generated_data_b)
                        rand_mse,rand_pnsr = self.MSE_PSNR(test_data_b, noise_data_b)
                        mse_reduction = rand_mse-mse
                        # MSE.append(mse)
                        # PSNR.append(psnr)
                    print('Epoch {}% \t train_loss:{:.2f} mse_reduction:{:.2f} psnr_addition:{:.2f}'.format(
                        i_epoch, loss.item(), rand_mse-mse, psnr-rand_pnsr))
            
            print(f"GRN, if self.args.apply_defense={self.args.apply_defense}")
            print(f'batch_size=%d,class_num=%d,party_index=%d,psnr=%lf' % (self.batch_size, self.label_size, index, psnr))

        print("returning from GRN")
        return rand_mse,mse