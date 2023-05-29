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
        super(custom_AE,self).__init__()
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

# class custom_AE(nn.Module):
#     def __init__(self, input_nc=256, output_nc=3, input_dim=8, output_dim=32, activation = "sigmoid"):
#         super(custom_AE, self).__init__()
#         upsampling_num = int(np.log2(output_dim // input_dim))
#         # get [b, 3, 8, 8]
#         model = []
#         nc = input_nc
#         for num in range(upsampling_num - 1):
#             #TODO: change to Conv2d
#             model += [nn.Conv2d(nc, int(nc/2), kernel_size=3, stride=1, padding=1)]
#             model += [nn.ReLU()]
#             model += [nn.ConvTranspose2d(int(nc/2), int(nc/2), kernel_size=3, stride=2, padding=1, output_padding=1)]
#             model += [nn.ReLU()]
#             nc = int(nc/2)
#         if upsampling_num >= 1:
#             model += [nn.Conv2d(int(input_nc/(2 ** (upsampling_num - 1))), int(input_nc/(2 ** (upsampling_num - 1))), kernel_size=3, stride=1, padding=1)]
#             model += [nn.ReLU()]
#             model += [nn.ConvTranspose2d(int(input_nc/(2 ** (upsampling_num - 1))), output_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
#             if activation == "sigmoid":
#                 model += [nn.Sigmoid()]
#             elif activation == "tanh":
#                 model += [nn.Tanh()]
#         else:
#             model += [nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=1, padding=1)]
#             model += [nn.ReLU()]
#             model += [nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1)]
#             if activation == "sigmoid":
#                 model += [nn.Sigmoid()]
#             elif activation == "tanh":
#                 model += [nn.Tanh()]
#         self.m = nn.Sequential(*model)

    # def forward(self, x):
    #     print('x:',x.size())
    #     output = self.m(x)
    #     print('output:',output.size())
    #     assert 1>2
    #     return output


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
        self.party = args.attack_configs['party'] # parties that launch attacks
        self.lr = args.attack_configs['lr']
        self.epochs = args.attack_configs['epochs']
        self.attack_batch_size = args.attack_configs['batch_size']
        
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

    def attack(self):
        self.set_seed(123)
        print_every = 1
        for ik in self.party: # attacker party #ik
            assert ik == 1, 'Only Active party launch feature inference attack'
            index = ik
            # collect necessary information
            net_b = self.vfl_info['final_model'][0].to(self.device) # Passive
            net_a = self.vfl_info['final_model'][1].to(self.device) # Active
            global_model = self.vfl_info['final_global_model'].to(self.device)
            global_model.eval()
            net_b.eval()
            net_a.eval()

            batch_size = self.attack_batch_size

            # Test Data
            test_data_a =  self.vfl_info['test_data'][1] # Active Test Data
            test_data_b =  self.vfl_info['test_data'][0] # Passive Test Data

            # Train with Aux Dataset
            aux_data_a = self.vfl_info["aux_data"][1]
            aux_data_b = self.vfl_info["aux_data"][0]
            aux_label = self.vfl_info["aux_label"][-1]
            aux_dst_a = ActiveDataset(aux_data_a, aux_label)
            aux_loader_a = DataLoader(aux_dst_a, batch_size=batch_size)
            aux_dst_b = PassiveDataset(aux_data_b)
            aux_loader_b = DataLoader(aux_dst_b, batch_size=batch_size)
            aux_loader_list = [aux_loader_b,aux_loader_a]

            # Initiate Decoder
            '''
            test_data_b = [batchsize,1,28,14]
            '''
            if self.args.dataset == 'nuswide':
                dim_a = test_data_a.size()[1]
                dim_b = test_data_b.size()[1]
            else: # mnist cifar
                dim_a = test_data_a.size()[1]*test_data_a.size()[2]*test_data_a.size()[3]
                dim_b = test_data_b.size()[1]*test_data_b.size()[2]*test_data_b.size()[3]
            
            criterion = nn.MSELoss()
            latent_dim = net_b(test_data_b).size()[1]
            decoder = custom_AE(latent_dim, dim_b).to(self.device)
            #custom_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim, output_dim=32).to(self.device)
            optimizer = torch.optim.Adam(decoder.parameters(), lr=self.lr)
            
            print('========= Feature Inference Training ========')
            for i_epoch in range(self.epochs):
                for parties_data in zip(*aux_loader_list):
                    self.gt_one_hot_label = label_to_one_hot(parties_data[self.k-1][1], self.num_classes)
                    self.gt_one_hot_label = self.gt_one_hot_label.to(self.device)
                    self.parties_data = parties_data
                    batch_data_b = parties_data[0][0] # Passive Party data  
                    batch_data_a = parties_data[1][0] # Active Party data   

                    decoder.train()

                    # target img
                    img = batch_data_b 
                    
                    # Known Information : intermediate representation
                    with torch.no_grad():
                        ir = net_b(batch_data_b) 
                    img, ir = img.type(torch.FloatTensor), ir.type(torch.FloatTensor)
                    img, ir = Variable(img).to(self.device), Variable(ir).to(self.device)
                    
                    # recovered image
                    output = decoder(ir)
                    img = img.reshape(output.size())
                    # print('ir:',ir.size())
                    # print('img:',img.size())
                    # print('output:',output.size())

                    train_loss = criterion(output, img)

                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()

                ####### Test Performance of Generator #######
                if (i_epoch + 1) % print_every == 0:
                    with torch.no_grad():
                        # test_data_a = self.vfl_first_epoch['data'][1][0] # active party 
                        # test_data_b = self.vfl_first_epoch['data'][0][0] # passive party 
                        # # pred with possible defense 
                        # test_pred_a = self.vfl_first_epoch['predict'][1]
                        # test_pred_b = self.vfl_first_epoch['predict'][0]
                        # test_global_pred = self.vfl_first_epoch['global_pred'].to(self.device)

                        img = test_data_b # target img
                        test_pred_b = net_b(test_data_b)
                        ir = test_pred_b 
                        img, ir = img.type(torch.FloatTensor), ir.type(torch.FloatTensor)
                        img, ir = Variable(img).to(self.device), Variable(ir).to(self.device)
                        
                        output = decoder(ir) # reconstruction result

                        if i_epoch == self.epochs -1:
                            print()
                            data_to_be_visualized = output[0].reshape(img[0].size())
                            plt.imshow(data_to_be_visualized[0])
                            plt.savefig('exp_result/' + 'dummy.png')
                            plt.close()
                            origin_data = img[0][0]
                            plt.imshow(origin_data)
                            plt.savefig('exp_result/' + 'origin.png')
                            plt.close()

                        img = img.reshape(output.size())
                        rand_img = torch.randn(img.size()).to(self.device)

                        mse = criterion(output, img)
                        rand_mse = criterion(rand_img, img)
                
                    print('Epoch {}% \t train_loss:{:.2f} mse_reduction:{:.2f}'.format(
                        i_epoch, train_loss.item(), rand_mse-mse))
            
            print(f"ResSFL, if self.args.apply_defense={self.args.apply_defense}")
            print(f'batch_size=%d,class_num=%d,party_index=%d,mse=%lf' % (self.batch_size, self.label_size, index, mse))

        print("returning from ResSFL")
        return rand_mse,mse