import sys, os
sys.path.append(os.pardir)
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from party.party import Party
from utils.basic_functions import cross_entropy_for_onehot, tf_distance_cov_cor,pairwise_dist
from utils.paillier_torch import PaillierMSELoss
from dataset.party_dataset import ActiveDataset

class PaillierActiveParty(Party):
    def __init__(self, args, index):
        super().__init__(args, index)
        self.criterion = PaillierMSELoss()
        self.encoder = args.encoder
        # print(f"in active party, encoder=None? {self.encoder==None}, {self.encoder}")
        self.train_index = args.idx_train
        self.test_index = args.idx_test
        
        self.gt_one_hot_label = None

        self.pred_received = []
        for _ in range(args.k):
            self.pred_received.append([])
        
        self.global_pred = None
        self.global_loss = None

    
    def prepare_data(self, args, index):
        super().prepare_data(args, index)
        self.train_dst = ActiveDataset(self.train_data, self.train_label)
        self.test_dst = ActiveDataset(self.test_data, self.test_label)
        if self.args.need_auxiliary == 1:
            self.aux_dst = ActiveDataset(self.aux_data, self.aux_label)
            # self.aux_loader = DataLoader(self.aux_dst, batch_size=batch_size,shuffle=True)

    def update_local_pred(self, pred):
        self.pred_received[self.args.k-1] = pred
    
    def receive_pred(self, pred, giver_index):
        self.pred_received[giver_index] = pred

    def aggregate(self, pred_list, gt_one_hot_label, test=False):
        pred = F.softmax(sum(pred_list), dim=-1)
        loss = self.criterion(pred, gt_one_hot_label)
        return pred, loss

    def calculate_exp_H(self):
        # currently support only two parties
        H_a = self.pred_received[0][0]
        H_a_square = self.pred_received[0][1]
        H_b = self.pred_received[1][0]
        H_b_square = self.pred_received[1][1]
        
        H = H_a + H_b
        H_square = H_a_square + 2 * H_a * H_b + H_b_square
        exp_H = 1 + H + 0.5 * H_square
        # exp_H = torch.exp(H)
        
        self.mask = torch.rand(exp_H.size())
        self.mask = self.mask - (self.mask.sum(dim=1) / self.mask.shape[1]).reshape(-1, 1)
        exp_H = exp_H + self.mask.to(exp_H.device)

        return exp_H

    def gradient_calculation(self, pred, ground_truth):
        pred_gradients_list = []
        for ik in range(self.args.k):
            pred_gradients_list.append((pred.to(ground_truth.device) - ground_truth))
        
        return pred_gradients_list
    
    def give_gradient(self, pred):
        pred = pred - self.mask.to(pred.device)
        if self.gt_one_hot_label == None:
            print('give gradient:self.gt_one_hot_label == None')
            assert 1>2

        gradients_list = self.gradient_calculation(pred, self.gt_one_hot_label)

        return gradients_list
    
    def update_local_gradient(self, gradient):
        self.local_gradient = gradient

    def local_backward(self):
        # update local model
        self.local_model_optimizer.zero_grad()
        torch.autograd.backward(self.local_pred, self.local_gradient)
        self.local_model_optimizer.step()
