import sys, os
sys.path.append(os.pardir)
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from party.party import Party
from party.llm_party import Party as Party_LLM
from utils.basic_functions import cross_entropy_for_onehot, tf_distance_cov_cor,pairwise_dist
from dataset.party_dataset import ActiveDataset, ActiveDataset_LLM


class Server(Party_LLM):
    def __init__(self, args, index):
        super().__init__(args, index)
        self.name = "server#" + str(index + 1)
        self.criterion = cross_entropy_for_onehot
        self.encoder = args.encoder
        
        self.train_index = args.idx_train
        self.test_index = args.idx_test
        
        self.gt_one_hot_label = None

        self.pred_received = []
        for _ in range(args.k):
            self.pred_received.append([])
        
        self.global_pred = None
        self.global_loss = None


    def prepare_data(self, args, index):
        print('Server prepare no data')
        # super().prepare_data(args, index)
        # self.train_dst = ActiveDataset_LLM(args, self.train_data, self.train_label)
        # self.test_dst = ActiveDataset_LLM(args, self.test_data, self.test_label)

        # print('Active self.train_dst:',len(self.train_dst),  type(self.train_dst[0]), type(self.train_dst[1]) )

            
    # def update_local_pred(self, pred):
    #     self.pred_received[self.args.k-1] = pred
    
    def receive_pred(self, pred, giver_index):
        self.pred_received[giver_index] = pred

    def aggregate(self, pred_list, test=False):
        
        pred = self.global_model(pred_list[0], self.input_shape)
        self.global_pred = pred

        return pred


    def global_LR_decay(self,i_epoch):
        if self.global_model_optimizer != None: 
            eta_0 = self.args.main_lr
            eta_t = eta_0/(np.sqrt(i_epoch+1))
            for param_group in self.global_model_optimizer.param_groups:
                param_group['lr'] = eta_t
        
                
    def global_backward(self):

        if self.global_model_optimizer != None: 
            # server with trainable global layer
            _gradients = torch.autograd.grad(self.global_loss, self.global_pred, retain_graph=True)
            _gradients_clone = _gradients[0].detach().clone()

            # print('global_gradients_clone:',_gradients_clone)
            
            # update global model
            self.global_model_optimizer.zero_grad()
            parameters = []          
            if (self.args.apply_mid == True) and (self.index in self.args.defense_configs['party']): 
                # mid parameters
                for mid_model in self.global_model.mid_model_list:
                    parameters += list(mid_model.parameters())
                parameters += list(self.global_model.trainable_layer.parameters())
                # load grads into parameters
                weights_grad_a = torch.autograd.grad(self.global_pred, parameters, grad_outputs=_gradients_clone, retain_graph=True)
                for w, g in zip(parameters, weights_grad_a):
                    if w.requires_grad:
                        w.grad = g.detach()
            else:
                # trainable layer parameters
                # load grads into parameters
                weights_grad_a = torch.autograd.grad(self.global_pred, self.global_model.trainable_layer.parameters(), grad_outputs=_gradients_clone, retain_graph=True)
                for w, g in zip(self.global_model.trainable_layer.parameters(), weights_grad_a):
                    if w.requires_grad:
                        w.grad = g.detach()
                # print('weights_grad_a:',weights_grad_a)
                # non-trainabel layer: no need to update

            # for p in self.global_model.trainable_layer.parameters():#打印出每一层的参数的大小
            #     print(p)
            #     continue
            self.global_model_optimizer.step()
            # print('after==========')
            # for p in self.global_model.trainable_layer.parameters():#打印出每一层的参数的大小
            #     print(p)
            #     continue
            # assert 1>2
