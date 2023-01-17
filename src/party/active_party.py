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
        self.encoder = args.encoder
    
    def prepare_data(self, args, index):
        super().prepare_data(args, index)
        self.train_dst = ActiveDataset(self.train_data, self.train_label)
        self.test_dst = ActiveDataset(self.test_data, self.test_label)
    
    # def aggregate_and_gradient_calculation(self, pred_list, gt_one_hot_label):
    #     pred = self.global_model(pred_list)
    #     loss = self.criterion(pred, gt_one_hot_label)
    #     pred_gradients_list = []
    #     pred_gradients_list_clone = []
    #     for ik in range(self.args.k):
    #         _gradients = torch.autograd.grad(loss, pred_list[ik], retain_graph=True)
    #         _gradients_clone = _gradients[0].detach().clone()
    #         pred_gradients_list.append(_gradients)
    #         pred_gradients_list_clone.append(_gradients_clone)
    #     # self.global_backward(pred, loss)
    #     return pred, loss, pred_gradients_list, pred_gradients_list_clone
    
    def aggregate(self, pred_list, gt_one_hot_label):
        pred = self.global_model(pred_list)
        loss = self.criterion(pred, gt_one_hot_label)
        # ########## for active mid model loss (start) ##########
        if self.args.apply_mid == True and (self.index in self.args.defense_configs['party']):
            for mid_loss in self.global_model.mid_loss_list:
                loss = loss + mid_loss
            self.global_model.mid_loss_list = [torch.empty((1,1)).to(self.args.device) for _ in range(len(self.global_model.mid_loss_list))]
        # ########## for active mid model loss (end) ##########
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
        if self.global_model_optimizer != None: 
            # active party with trainable global layer
            _gradients = torch.autograd.grad(loss, pred, retain_graph=True)
            _gradients_clone = _gradients[0].detach().clone()
            # update local model
            self.global_model_optimizer.zero_grad()
            
            if self.args.apply_trainable_layer == True:
                weights_grad_a = torch.autograd.grad(pred, self.global_model.parameters(), grad_outputs=_gradients_clone, retain_graph=True)
                for w, g in zip(self.global_model.parameters(), weights_grad_a):
                    if w.requires_grad:
                        w.grad = g.detach()
            elif self.args.apply_mid == True: # no trainable top model but mid models
                parameters = []
                for mid_model in self.global_model.mid_model_list:
                    parameters += list(mid_model.parameters())
                weights_grad_a = torch.autograd.grad(pred, parameters, grad_outputs=_gradients_clone, retain_graph=True)
                for w, g in zip(parameters, weights_grad_a):
                    if w.requires_grad:
                        w.grad = g.detach()
            self.global_model_optimizer.step()
