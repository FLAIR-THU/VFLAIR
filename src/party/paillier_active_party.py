import sys, os

sys.path.append(os.pardir)
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from party.party import Party
from utils.basic_functions import (
    cross_entropy_for_onehot,
    tf_distance_cov_cor,
    pairwise_dist,
)
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
        self.pred_received[self.args.k - 1] = pred

    def receive_pred(self, pred, giver_index):
        self.pred_received[giver_index] = pred

    def aggregate(self, pred_list, gt_one_hot_label, test=False):
        pred = self.global_model(pred_list)
        if self.train_index != None:  # for graph data
            if test == False:
                loss = self.criterion(
                    pred[self.train_index], gt_one_hot_label[self.train_index]
                )
            else:
                loss = self.criterion(
                    pred[self.test_index], gt_one_hot_label[self.test_index]
                )
        else:
            loss = self.criterion(pred, gt_one_hot_label)
        # ########## for active mid model loss (start) ##########
        if self.args.apply_mid == True and (
            self.index in self.args.defense_configs["party"]
        ):
            # print(f"in active party mid, label={gt_one_hot_label}, global_model.mid_loss_list={self.global_model.mid_loss_list}")
            assert len(pred_list) - 1 == len(self.global_model.mid_loss_list)
            for mid_loss in self.global_model.mid_loss_list:
                loss = loss + mid_loss
            self.global_model.mid_loss_list = [
                torch.empty((1, 1)).to(self.args.device)
                for _ in range(len(self.global_model.mid_loss_list))
            ]
        # ########## for active mid model loss (end) ##########
        elif self.args.apply_dcor == True and (
            self.index in self.args.defense_configs["party"]
        ):
            # print('dcor active defense')
            self.distance_correlation_lambda = self.args.defense_configs["lambda"]
            # loss = criterion(pred, gt_one_hot_label) + self.distance_correlation_lambda * torch.mean(torch.cdist(pred_a, gt_one_hot_label, p=2))
            for ik in range(self.args.k - 1):
                loss += self.distance_correlation_lambda * torch.log(
                    tf_distance_cov_cor(pred_list[ik], gt_one_hot_label)
                )  # passive party's loss
        return pred, loss

    def aggregate_H(self):
        # currently support only two parties
        H = sum(self.pred_received)

        return H

    def gradient_calculation(self, ground_truth):
        pred = self.aggregate_H()
        ground_truth = (ground_truth - 0.5) * 2
        grad = 0.25 * pred - 0.5 * ground_truth
        # grad = -ground_truth * torch.exp(-ground_truth * pred) / (1 + torch.exp(-ground_truth * pred))

        pred_gradients_list = []
        for ik in range(self.args.k):
            pred_gradients_list.append(grad)

        return pred_gradients_list

    def give_gradient(self):
        if self.gt_one_hot_label == None:
            print("give gradient:self.gt_one_hot_label == None")
            assert 1 > 2

        gradients_list = self.gradient_calculation(self.gt_one_hot_label)
        return gradients_list

    def update_local_gradient(self, gradient):
        self.local_gradient = [
            torch.matmul(
                gradient.T,
                self.local_batch_data.reshape(gradient.size()[0], -1),
            ),
            torch.sum(gradient, dim=0),
        ]
        self.local_batch_size = gradient.size()[0]

        self.random_masks = []
        for i in range(len(self.local_gradient)):
            mask = torch.randn(self.local_gradient[i].size()).to(
                self.local_gradient[i].device
            )
            self.local_gradient[i] = self.local_gradient[i] + mask
            self.random_masks.append(mask)

    def local_backward(self):
        # update local model
        self.local_model_optimizer.zero_grad()
        params = list(self.local_model.parameters())
        params[0].grad = (self.local_gradient[0] - self.random_masks[0]).to(
            params[0].device
        )
        params[1].grad = (self.local_gradient[1] - self.random_masks[1]).to(
            params[1].device
        )
        params[0].grad = params[0].grad / self.local_batch_size
        params[1].grad = params[1].grad / self.local_batch_size
        self.local_model_optimizer.step()
