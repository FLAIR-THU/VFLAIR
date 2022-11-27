import sys, os
sys.path.append(os.pardir)

import torch

from party.party import Party
from utils.basic_functions import cross_entropy_for_onehot
from dataset.nuswide_dataset import ActiveDataset

class ActiveParty(Party):
    def __init__(self, args, index):
        super().__init__(args, index)
        self.criterion = cross_entropy_for_onehot
    
    def prepare_data(self, args, index):
        super().prepare_data(args, index)
        self.train_dst = ActiveDataset(self.train_data, self.train_label)
        self.test_dst = ActiveDataset(self.test_data, self.test_label)
    
    def aggregate_and_gradient_calculation(self, pred_list, gt_one_hot_label):
        pred = self.global_model(pred_list)
        loss = self.criterion(pred, gt_one_hot_label)
        pred_gradients_list = []
        pred_gradients_list_clone = []
        for ik in range(self.args.k):
            _gradients = torch.autograd.grad(loss, pred_list[ik], retain_graph=True)
            _gradients_clone = _gradients[0].detach().clone()
            pred_gradients_list.append(_gradients)
            pred_gradients_list_clone.append(_gradients_clone)
        # self.global_backward(pred, loss)
        return pred, loss, pred_gradients_list, pred_gradients_list_clone
    
    def aggregate(self, pred_list, gt_one_hot_label):
        pred = self.global_model(pred_list)
        loss = self.criterion(pred, gt_one_hot_label)
        return pred, loss
    
    def gradient_calculation(self, pred, pred_list, loss):
        pred_gradients_list = []
        pred_gradients_list_clone = []
        for ik in range(self.args.k):
            pred_gradients_list.append(torch.autograd.grad(loss, pred_list[ik], retain_graph=True, create_graph=True))
            pred_gradients_list_clone.append(pred_gradients_list[ik][0].detach().clone())
        # self.global_backward(pred, loss)
        return pred_gradients_list, pred_gradients_list_clone
    
    def global_backward(self, pred, loss):
        _gradients = torch.autograd.grad(loss, pred, retain_graph=True)
        _gradients_clone = _gradients[0].detach().clone()
        # update local model
        self.global_model_optimizer.zero_grad()
        weights_grad_a = torch.autograd.grad(pred, self.global_model.parameters(), grad_outputs=_gradients_clone, retain_graph=True)
        for w, g in zip(self.global_model.parameters(), weights_grad_a):
            if w.requires_grad:
                w.grad = g.detach()
        self.global_model_optimizer.step()
