import sys, os
sys.path.append(os.pardir)
import numpy as np
import torch
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
        pred = sum(pred_list[:1])
        loss = self.criterion(pred, gt_one_hot_label)
        return pred, loss

    def gradient_calculation(self, pred_list, loss):
        pred_gradients_list = []
        grad = self.criterion.p_backward()
        for ik in range(self.args.k):
            pred_gradients_list.append(grad)
        return pred_gradients_list
    
    def give_gradient(self):
        pred_list = self.pred_received 

        if self.gt_one_hot_label == None:
            print('give gradient:self.gt_one_hot_label == None')
            assert 1>2

        self.global_pred, self.global_loss = self.aggregate(pred_list, self.gt_one_hot_label)
        pred_gradients_list = self.gradient_calculation(pred_list, self.global_loss)

        return pred_gradients_list
    
    def update_local_gradient(self, gradient):
        self.local_gradient = gradient
