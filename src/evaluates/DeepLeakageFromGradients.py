# import sys, os
# import time
# sys.path.append(os.pardir)

# import torch
# import csv
# import json
# import torch.nn.functional as F
# import torch.nn as nn
# from torch.autograd import grad
# import torchvision
# from utils.basic_functions import *
# from models.vision import *
# import matplotlib.pyplot as plt
# from torchvision import models, datasets, transforms

# torch.manual_seed(1234)

        
# def weights_init(m):
#     if hasattr(m, "weight"):
#         m.weight.data.uniform_(-0.5, 0.5)
#     if hasattr(m, "bias"):
#         m.bias.data.uniform_(-0.5, 0.5)

# class DLGAttacker(object): 
#     #The attacker object 
#     def __init__(self, args):
#         # self.data_size = args.data_size
#         self.verbose = 1
#         # self.save_path = args.save_path
#         # self.save_per_iter = args.save_per_iter
#         self.tt = transforms.ToPILImage()

#         self.dataset = args.dataset
#         self.model = args.model_list
#         self.num_exp = args.num_exp
#         self.epochs = args.epochs
#         self.lr = args.lr
#         self.early_stop = args.early_stop
#         self.early_stop_param = args.early_stop_param
#         self.device = args.device
#         self.batch_size_list = args.batch_size_list
#         self.num_class_list = args.num_class_list
#         self.dst = args.dst
#         self.exp_res_dir = args.exp_res_dir
#         self.exp_res_path = args.exp_res_path
#         self.net_a = args.net_list[0]
#         self.net_b = args.net_list[1]
#         self.gt_data_a = args.gt_data_a
#         self.gt_data_b = args.gt_data_b
#         self.gt_label = args.gt_label
#         self.gt_onehot_label = torch.stack(args.gt_onehot_label).to(self.device)
#         # defending parameters
#         self.apply_trainable_layer = args.apply_trainable_layer
#         self.apply_laplace = args.apply_laplace
#         self.apply_gaussian = args.apply_gaussian
#         self.dp_strength = args.dp_strength
#         self.apply_grad_spar = args.apply_grad_spar
#         self.grad_spars = args.grad_spars
#         self.apply_encoder = args.apply_encoder
#         self.apply_adversarial_encoder = args.apply_adversarial_encoder
#         self.ae_lambda = args.ae_lambda
#         self.encoder = args.encoder
#         self.apply_marvell = args.apply_marvell
#         self.marvell_s = args.marvell_s
#         self.show_param()
  
#     def train(self): #, model, origin_grad, criterion, num_iters=300
        
#         for num_classes in self.num_class_list:
#             for batch_size in self.batch_size_list:
#                 dummy_label_history = []
#                 real_label_history = []

#                 for i_run in range(1, self.num_exp+1):
#                     start_time = time.time()
                    
#                     criterion = cross_entropy_for_onehot
#                     origin_grad = self.get_original_grad(self.gt_data_a, self.gt_label, criterion, self.net_a)
#                     dummy_data = torch.randn(self.gt_data_a[0].shape).to(self.device).requires_grad_(True)
#                     dummy_label =  torch.randn([1,num_classes]).to(self.device).requires_grad_(True)
#                     #model = model.to(self.device)

#                     optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

#                     history = []
#                     for iters in range(self.epochs):
#                         def closure():
#                             optimizer.zero_grad()

#                             dummy_pred = self.net_a(dummy_data) 
#                             dummy_onehot_label = F.softmax(dummy_label, dim=-1)
#                             #print(dummy_label)
#                             dummy_loss = criterion(dummy_pred, dummy_onehot_label) 
#                             dummy_dy_dx = torch.autograd.grad(dummy_loss, self.net_a.parameters(), create_graph=True)
                            
#                             grad_diff = 0
#                             for gx, gy in zip(dummy_dy_dx, origin_grad): 
#                                 grad_diff += ((gx - gy) ** 2).sum()
#                             grad_diff.backward()
                            
#                             return grad_diff
                        
#                         optimizer.step(closure)
#                         if iters % 10 == 0: 
#                             current_loss = closure()
#                             print(iters, "%.4f" % current_loss.item())
#                             history.append(self.tt(dummy_data[0].cpu()))

#                             if self.verbose == 1:
#                                 self.save_dummy(dummy_data, dummy_label, iters)
                    
#                     self.dummy_data = dummy_data.detach().clone()
#                     self.dummy_label = dummy_label.detach().clone()

#                     fig = plt.figure(figsize=(12, 8))
#                     for i in range(30):
#                         plt.subplot(3, 10, i + 1)
#                         plt.imshow(history[i])
#                         plt.title("iter=%d" % (i * 10))
#                         plt.axis('off')
#                     plt.savefig('./visulization.png')

#                     end_time = time.time()
#                     print(f'batch_size=%d,class_num=%d,exp_id=%d,recovery_rate=%lf,time_used=%lf'
#                         % (batch_size, num_classes, i_run, rec_rate, end_time - start_time))

    
    
#     def get_original_grad(self, gt_data, gt_label, criterion, model):
      
#         self.save_dummy(gt_data, gt_label, 'real')
        
#         model = model.to(self.device)
#         gt_data = gt_data.to(self.device)
#         gt_label = gt_label.long().to(self.device)
#         gt_onehot_label = label_to_onehot(gt_label)
      
#         pred = net(gt_data)

#         y = criterion(pred, gt_onehot_label)

#         print(gt_onehot_label.shape)
#         dy_dx = torch.autograd.grad(y, net.parameters())

#         original_dy_dx = list((_.detach().clone() for _ in dy_dx))
      
#         return original_dy_dx
  
#     def get_original_grad_from_weights(self, parameter_new, parameter_old, num_rounds):
#         return torch.div(parameter_old - parameter_new, num_rounds)
  
  
#     def save_dummy(self, dummy_data, dummy_label, iters):
#         save_path = os.path.join(self.save_path, 'saved_dummys/iters_{}/'.format(iters))
#         if not os.path.exists(save_path):
#             os.makedirs(save_path)
        
#         torchvision.utils.save_image(dummy_data.detach().clone().cpu(), 
#                                     os.path.join(save_path, 'batch_images.png'), 
#                                     normalize=True,
#                                     nrow=10)
#         with open(os.path.join(save_path, 'batch_labels.txt'), 'w') as f:
#             f.write(str(dummy_label.detach().clone().cpu()))
      
        
# if __name__ == '__main__': 
    # pass
# #     #from utils.model_utils import create_model
# #     from torchvision import models, datasets, transforms
    
# #     img_index = 1
    
# #     transform = transforms.Compose(
# #        [transforms.ToTensor()])

# #     dataset = datasets.CIFAR10(root='./data', download=True)
    
# #     net = LeNet().to(args.device) #create_model('cnn', 'Mnist-alpha0.1-ratio0.5', 'FedGen')[0]
# #     net.apply(weights_init)
# #     #print(model)
# #     #model_path = os.path.join("models", 'Mnist-alpha0.1-ratio0.5')
# #     #model = torch.load(os.path.join(model_path, "server" + ".pt"))
    
# #     gt_data = transform(dataset[img_index][0]).to(args.device)
# #     gt_data = gt_data.view(args.batch_size, *gt_data.size())
# #     gt_label = torch.Tensor([dataset[img_index][1]]).long().to(args.device)
# #     gt_label = gt_label.view(args.batch_size, )
# #     criterion = cross_entropy_for_onehot
    
# #     gld_attacker =  DLGAttacker(args)
    
# #     origin_grad = gld_attacker.get_original_grad(gt_data, gt_label, criterion, net)
# #     gld_attacker.train(net, origin_grad, criterion)


import sys, os
sys.path.append(os.pardir)

# import logging
import pprint
import time

import tensorflow as tf
import torch

from models.vision import *
from utils.basic_functions import *
from utils.constants import *

import utils.constants as shared_var
from utils.marvell_functions import KL_gradient_perturb


tf.compat.v1.enable_eager_execution() 



class DeepLeakageFromGradients(object):
    def __init__(self, args):
        '''
        :param args:  contains all the necessary parameters
        '''
        self.attack_name = 'DeepLeakageFromGradients'
        self.tt = transforms.ToPILImage()
        self.dataset = args.dataset
        self.model = args.model_list
        self.num_exp = args.num_exp
        self.epochs = args.epochs
        self.lr = args.lr
        self.early_stop = args.early_stop
        self.early_stop_param = args.early_stop_param
        self.device = args.device
        self.batch_size_list = args.batch_size_list
        self.num_class_list = args.num_class_list
        self.dst = args.dst
        self.exp_res_dir = args.exp_res_dir
        self.exp_res_path = args.exp_res_path
        self.net = args.net_list[0]
        # self.net_b = args.net_list[1]
        self.gt_data = args.gt_data
        # self.gt_data_b = args.gt_data_b
        self.gt_label = args.gt_label
        self.gt_onehot_label = torch.stack(args.gt_onehot_label).to(self.device)
        # defending parameters
        self.apply_trainable_layer = args.apply_trainable_layer
        self.apply_laplace = args.apply_laplace
        self.apply_gaussian = args.apply_gaussian
        self.dp_strength = args.dp_strength
        self.apply_grad_spar = args.apply_grad_spar
        self.grad_spars = args.grad_spars
        self.apply_encoder = args.apply_encoder
        self.apply_adversarial_encoder = args.apply_adversarial_encoder
        self.ae_lambda = args.ae_lambda
        self.encoder = args.encoder
        self.apply_marvell = args.apply_marvell
        self.marvell_s = args.marvell_s
        self.show_param()

    def show_param(self):
        print(f'********** config dict **********')
        pprint.pprint(self.__dict__)

    def calc_label_recovery_rate(self, dummy_label, gt_label):
        success = torch.sum(torch.argmax(dummy_label, dim=-1) == torch.argmax(gt_label, dim=-1)).item()
        total = dummy_label.shape[0]
        return success / total

    def train(self):
        '''
        execute the label inference algorithm
        :return: recovery rate
        '''

        print(f"Running on %s{torch.cuda.current_device()}" % self.device)
        # if self.dataset == 'nuswide':
        #     all_nuswide_labels = []
        #     for line in os.listdir('./data/NUS_WIDE/Groundtruth/AllLabels'):
        #         all_nuswide_labels.append(line.split('_')[1][:-4])
        for batch_size in self.batch_size_list:
            for num_classes in self.num_class_list:

                # ######################################## generate original gradients ########################################
                criterion = cross_entropy_for_onehot
                print(self.gt_data.shape, torch.argmax(self.gt_label, dim=-1))
                pred_original = self.net(self.gt_data)
                ######################## defense start ############################
                ######################## defense1: trainable layer ############################
                if self.apply_trainable_layer:
                    active_aggregate_model = ActivePartyWithTrainableLayer(input_dim=num_classes * 2, output_dim=num_classes)
                    dummy_active_aggregate_model = ActivePartyWithTrainableLayer(input_dim=num_classes * 2, output_dim=num_classes)
                else:
                    active_aggregate_model = ActivePartyWithoutTrainableLayer()
                    dummy_active_aggregate_model = ActivePartyWithoutTrainableLayer()
                loss_original = criterion(pred_original, self.gt_onehot_label)
                
                ######################## for defense2~4, calculate pred_a_gradients ############################
                pred_original_gradients = torch.autograd.grad(loss_original, pred_original, retain_graph=True)
                pred_original_gradients_clone = pred_original_gradients[0].detach().clone()
                ######################## defense2: dp ############################
                if self.apply_laplace and self.dp_strength != 0 or self.apply_gaussian and self.dp_strength != 0:
                    location = 0.0
                    threshold = 0.2  # 1e9
                    if self.apply_laplace:
                        with torch.no_grad():
                            scale = self.dp_strength
                            # clip 2-norm per sample
                            norm_factor_original = torch.div(torch.max(torch.norm(pred_original_gradients_clone, dim=1)),threshold + 1e-6).clamp(min=1.0)
                            # add laplace noise
                            dist_original = torch.distributions.laplace.Laplace(location, scale)
                            pred_original_gradients_clone = torch.div(pred_original_gradients_clone, norm_factor_original) + \
                                        dist_original.sample(pred_original_gradients_clone.shape).to(self.device)
                    elif self.apply_gaussian:
                        with torch.no_grad():
                            scale = self.dp_strength
                            norm_factor_original = torch.div(torch.max(torch.norm(pred_original_gradients_clone, dim=1)),
                                                        threshold + 1e-6).clamp(min=1.0)
                            pred_original_gradients_clone = torch.div(pred_original_gradients_clone, norm_factor_original) + \
                                                    torch.normal(location, scale, pred_original_gradients_clone.shape).to(self.device)
                ######################## defense3: gradient sparsification ############################
                elif self.apply_grad_spar and self.grad_spars != 0:
                    with torch.no_grad():
                        percent = self.grad_spars / 100.0
                        up_thr = torch.quantile(torch.abs(pred_original_gradients_clone), percent)
                        active_up_gradients_res = torch.where(
                            torch.abs(pred_original_gradients_clone).double() < up_thr.item(),
                            pred_original_gradients_clone.double(), float(0.)).to(self.device)
                        pred_original_gradients_clone = pred_original_gradients_clone - active_up_gradients_res
                ######################## defense4: marvell ############################
                elif self.apply_marvell and self.marvell_s != 0 and num_classes == 2:
                    # for marvell, change label to [0,1]
                    marvell_y = []
                    for i in range(len(self.gt_label)):
                        marvell_y.append(int(self.gt_label[i][1]))
                    marvell_y = np.array(marvell_y)
                    shared_var.batch_y = np.asarray(marvell_y)
                    logdir = 'marvell_logs/dlg_task/{}_logs/{}'.format(self.dataset, time.strftime("%Y%m%d-%H%M%S"))
                    writer = tf.summary.create_file_writer(logdir)
                    shared_var.writer = writer
                    with torch.no_grad():
                        pred_a_gradients_clone = KL_gradient_perturb(pred_original_gradients_clone, self.classes, self.marvell_s)
                        pred_a_gradients_clone = pred_original_gradients_clone.to(self.device)
                ######################## defense end ############################
                original_dy_dx = torch.autograd.grad(pred_original, self.net.parameters(), grad_outputs=pred_original_gradients_clone) 

                
                
                # ######################################## steal from original gradients ########################################
                self.time_str = time.strftime("%Y%m%d-%H%M%S")
                save_path = f'exp_result/{self.attack_name}/{self.dataset}/saved_dummys/{self.time_str}/'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torchvision.utils.save_image(self.gt_data[0].detach().clone().cpu(), 
                                            os.path.join(save_path, 'original_image.png'), 
                                            normalize=True, nrow=10)
                for i_run in range(1, self.num_exp + 1):
                    start_time = time.time()
                    data_history = []
                    label_history = []
                    recognize_history = []
                    
                    dummy_data = torch.randn(self.gt_data.size()).to(self.device).requires_grad_(True)
                    dummy_label = torch.randn(self.gt_onehot_label.size()).to(self.device).requires_grad_(True)

                    if self.apply_trainable_layer:
                        optimizer = torch.optim.Adam([dummy_data, dummy_label] + list(dummy_active_aggregate_model.parameters()), lr=self.lr)
                    else:
                        optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=self.lr)

                    for iters in range(1, self.epochs + 1):
                        def closure():
                            optimizer.zero_grad()
                            dummy_pred = self.net(dummy_data)

                            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
                            dummy_loss = criterion(dummy_pred, dummy_onehot_label)
                            dummy_dy_dx = torch.autograd.grad(dummy_loss, self.net.parameters(), create_graph=True)
                            grad_diff = 0
                            for (gx, gy) in zip(dummy_dy_dx, original_dy_dx):
                                grad_diff += ((gx - gy) ** 2).sum()
                            grad_diff.backward()
                            return grad_diff

                        # if iters == 1:
                        #     append_exp_res(f'exp_result/{self.dataset}/exp_on_{self.dataset}_rec_rate_change.txt',
                        #                    f'{batch_size} 0 {rec_rate} {closure()}')
                        optimizer.step(closure)
                        
                        if iters % 100 == 0: 
                            current_loss = closure()
                            # print(iters, "%.4f" % current_loss.item())
                            data_history.append(self.tt(dummy_data[0].cpu()))
                            self.save_dummy(dummy_data, dummy_label, i_run, iters)
                        
                        if self.early_stop == True:
                            if closure().item() < self.early_stop_param:
                                break
                    # save data history
                    data_history.append(self.tt(self.gt_data[0].cpu()))
                    fig = plt.figure(figsize=(12, 8))
                    for i in range((self.epochs//100)+1):
                        plt.subplot(3, 10, i + 1)
                        plt.imshow(data_history[i])
                        plt.title("iter=%d" % (i * 100))
                        plt.axis('off')
                    plt.savefig('./visulization.png')
                    
                    rec_rate = self.calc_label_recovery_rate(dummy_label, self.gt_label)
                    label_history.append(rec_rate)
                    recognize_rate = self.calc_label_recovery_rate(self.net(dummy_data),self.gt_label)
                    recognize_history.append(recognize_rate)

                    self.dummy_data = dummy_data.detach().clone()
                    self.dummy_label = dummy_label.detach().clone()

                    end_time = time.time()
                    # save label history
                    rec_rate = np.sum(np.asarray(label_history)) / len(label_history)
                    recognize_rate = np.sum(np.asarray(recognize_history)) / len(recognize_history)

                    # output the rec_info of this exp
                    if self.apply_laplace or self.apply_gaussian:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,dp_strength=%lf,recovery_rate=%lf,recognize_rate=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.dp_strength,rec_rate,recognize_rate, end_time - start_time))
                    elif self.apply_grad_spar:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,grad_spars=%lf,recovery_rate=%lf,recognize_rate=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.grad_spars,rec_rate,recognize_rate, end_time - start_time))
                    elif self.apply_marvell:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,marvel_s=%lf,recovery_rate=%lf,recognize_rate=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.marvell_s,rec_rate,recognize_rate, end_time - start_time))
                    else:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,recovery_rate=%lf,recognize_rate=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, rec_rate,recognize_rate, end_time - start_time))
                avg_rec_rate = np.mean(label_history)
                avg_recognize_rate = np.mean(recognize_history)
                if self.apply_laplace or self.apply_gaussian:
                    exp_result = str(self.dp_strength) + ' ' + str(avg_rec_rate) + ' ' + str(label_history) + ' ' + str(np.max(label_history))
                elif self.apply_grad_spar:
                    exp_result = str(self.grad_spars) + ' ' + str(avg_rec_rate) + ' ' + str(label_history) + ' ' + str(np.max(label_history))
                elif self.apply_encoder or self.apply_adversarial_encoder:
                    exp_result = str(self.ae_lambda) + ' ' + str(avg_rec_rate) + ' ' + str(label_history) + ' ' + str(np.max(label_history))
                elif self.apply_marvell:
                    exp_result = str(self.marvell_s) + ' ' + str(avg_rec_rate) + ' ' + str(label_history) + ' ' + str(np.max(label_history))
                else:
                    exp_result = f"bs|num_class|recovery_rate|avg_recognize_rate,%d|%d|%lf|%s|%lf|%lf|%s" % (batch_size, num_classes, avg_rec_rate, str(label_history), np.max(label_history),avg_recognize_rate,str(recognize_history))

                print(self.exp_res_path, self.gt_label)
                append_exp_res(self.exp_res_path, exp_result)
                print(exp_result)
    
    def save_dummy(self, dummy_data, dummy_label, i_run, iters):
        save_path = f'exp_result/{self.attack_name}/{self.dataset}/saved_dummys/{self.time_str}/{i_run}/iters_{iters}/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
        torchvision.utils.save_image(dummy_data[0].detach().clone().cpu(), 
                                    os.path.join(save_path, 'batch_images.png'), 
                                    normalize=True,
                                    nrow=10)
        with open(os.path.join(save_path, 'batch_labels.txt'), 'w') as f:
            f.write(str(dummy_label.detach().clone().cpu()))

if __name__ == '__main__':
    pass