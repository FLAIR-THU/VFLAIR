import sys, os
sys.path.append(os.pardir)

import torch
import torch.nn.functional as F
import time
import numpy as np
import copy
import pickle 
import matplotlib.pyplot as plt
import itertools 

from evaluates.attacks.attacker import Attacker
from models.global_models import * #ClassificationModelHostHead, ClassificationModelHostTrainableHead
from utils.basic_functions import cross_entropy_for_onehot, append_exp_res

class InferenceHead(nn.Module):
    def __init__(self, size_bottom_out, num_classes, num_layer=1, activation_func_type='ReLU', use_bn=True):
        super(self).__init__()
        self.bottom_model = BottomModel(dataset_name=None)

        dict_activation_func_type = {'ReLU': F.relu, 'Sigmoid': F.sigmoid, 'None': None}
        self.activation_func = dict_activation_func_type[activation_func_type]
        self.num_layer = num_layer
        self.use_bn = use_bn

        self.fc_1 = nn.Linear(size_bottom_out, size_bottom_out, bias=True)
        self.bn_1 = nn.BatchNorm1d(size_bottom_out)
        self.fc_1.apply(weights_init_ones)

        self.fc_2 = nn.Linear(size_bottom_out, size_bottom_out, bias=True)
        self.bn_2 = nn.BatchNorm1d(size_bottom_out)
        self.fc_2.apply(weights_init_ones)

        self.fc_3 = nn.Linear(size_bottom_out, size_bottom_out, bias=True)
        self.bn_3 = nn.BatchNorm1d(size_bottom_out)
        self.fc_3.apply(weights_init_ones)

        self.fc_4 = nn.Linear(size_bottom_out, size_bottom_out, bias=True)
        self.bn_4 = nn.BatchNorm1d(size_bottom_out)
        self.fc_4.apply(weights_init_ones)

        self.fc_final = nn.Linear(size_bottom_out, num_classes, bias=True)
        self.bn_final = nn.BatchNorm1d(size_bottom_out)
        self.fc_final.apply(weights_init_ones)

    def forward(self, x):
        x = self.bottom_model(x)

        if self.num_layer >= 2:
            if self.use_bn:
                x = self.bn_1(x)
            if self.activation_func:
                x = self.activation_func(x)
            x = self.fc_1(x)

        if self.num_layer >= 3:
            if self.use_bn:
                x = self.bn_2(x)
            if self.activation_func:
                x = self.activation_func(x)
            x = self.fc_2(x)

        if self.num_layer >= 4:
            if self.use_bn:
                x = self.bn_3(x)
            if self.activation_func:
                x = self.activation_func(x)
            x = self.fc_3(x)

        if self.num_layer >= 5:
            if self.use_bn:
                x = self.bn_4(x)
            if self.activation_func:
                x = self.activation_func(x)
            x = self.fc_4(x)

        if self.use_bn:
            x = self.bn_final(x)
        if self.activation_func:
            x = self.activation_func(x)
        x = self.fc_final(x)

        return x

class PassiveModelCompletion(Attacker):
    def __init__(self, top_vfl, args):
        super().__init__(args)
        self.args = args
        # get information for launching BLI attack
        self.vfl_info = top_vfl.first_epoch_state
        # prepare parameters
        self.device = args.device
        self.num_classes = args.num_classes
        self.k = args.k
        self.party = args.attack_configs['party'] # parties that launch attacks
        self.lr = args.attack_configs['lr']
        self.epochs = args.attack_configs['epochs']
        self.label_size = args.num_classes

        self.hidden_size = args.attack_configs['hidden_size']

        self.dummy_active_top_trainable_model = None
        self.optimizer_trainable = None # construct later
        self.dummy_active_top_non_trainable_model = None
        self.optimizer_non_trainable = None # construct later
        self.criterion = cross_entropy_for_onehot
        self.file_name = 'attack_result.txt'
        self.exp_res_dir = f'exp_result/main/{args.dataset}/attack/BLR/'
        self.exp_res_path = ''
    
    def set_seed(self,seed=0):
        # random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def calc_label_recovery_rate(self, dummy_label, gt_label):
        success = torch.sum(torch.argmax(dummy_label, dim=-1) == torch.argmax(gt_label, dim=-1)).item()
        total = dummy_label.shape[0]
        return success / total


    def attack(self):
        self.set_seed(123)
        for ik in self.party: # attacker party #ik
            index = ik
            
            # collect necessary information
            self_data = self.vfl_info['data'][ik][0]#copy.deepcopy(self.parties_data)
            active_data = self.vfl_info['data'][1][0] # Active party data

            bottom_model = self.vfl_info['model'][0].to(self.device)
            bottom_model.eval()
            pred_a = self.vfl_info['predict'][ik]
            true_label = self.vfl_info['label'].to(self.device) # copy.deepcopy(self.gt_one_hot_label)

            sample_count = pred_a.size()[0]

            
            # Load Inference Head (fake top model)
            assert self.label_size == pred_a.size[0]
            inferencehead = InferenceHead(self.label_size, self.hidden_size, self.label_size, num_layer=1, activation_func_type='ReLU', use_bn=True)
            optimizer = torch.optim.Adam(lr=self.lr,betas=(0.9, 0.999),eps=1e-08,weight_decay=0,amsgrad=False)
            inferencehead = inferencehead.to(self.device) # dummy top model

            # === Begin Attack ===
            print(f"PMC Attack, self.device={self.device}")
            start_time = time.time()

            for iters in range(1, self.epochs + 1):
                # s_time = time.time()
                
                def closure():
                    optimizer.zero_grad()

                    # fake pred/loss using fake top model/fake label
                    dummy_pred = dummy_active_aggregate_model(pred_a)
                    dummy_onehot_label = F.softmax(dummy_label, dim=-1)
                    dummy_loss = self.criterion(dummy_pred, dummy_onehot_label)
                    # dummy L
                    dummy_dy_dx_a = torch.autograd.grad(dummy_loss, net_a.parameters(), create_graph=True)
                    
                    # loss: L-L'
                    grad_diff = 0
                    for (gx, gy) in zip(dummy_dy_dx_a, original_dy_dx):
                        grad_diff += ((gx - gy) ** 2).sum()
                    grad_diff.backward(retain_graph=True)

                    # if iters%200==0:
                    #     print('Iters',iters,' grad_diff:',grad_diff.item())
                    return grad_diff
                
                
                # rec_rate = self.calc_label_recovery_rate(dummy_label, true_label)
                # print(f"iter={iters}::rec_rate={rec_rate}")
                optimizer.step(closure)
                e_time = time.time()
                # print(f"in BLR, i={i}, iter={iters}, time={s_time-e_time}")
                
                if self.early_stop == 1:
                    if closure().item() < self.early_stop_threshold:
                        break
                
                rec_rate = self.calc_label_recovery_rate(dummy_label, true_label)
                # if iters%200==0:
                #     print('Iters',iters,' rec_rate:',rec_rate)
                recovery_rate_history[i].append(rec_rate)
                end_time = time.time()

            print(f'batch_size=%d,class_num=%d,party_index=%d,recovery_rate=%lf,time_used=%lf' % (sample_count, self.label_size, index, rec_rate, end_time - start_time))
        
            final_rec_rate_trainable = recovery_rate_history[0][-1] #sum(recovery_rate_history[0])/len(recovery_rate_history[0])
            final_rec_rate_non_trainable = recovery_rate_history[1][-1] #sum(recovery_rate_history[1])/len(recovery_rate_history[1])
            best_rec_rate = max(final_rec_rate_trainable,final_rec_rate_non_trainable)

            print(f"BLI, if self.args.apply_defense={self.args.apply_defense}")
        
        print("returning from BLI")
        return best_rec_rate
        # return recovery_history