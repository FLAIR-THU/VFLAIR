import sys, os
sys.path.append(os.pardir)

import torch
import torch.nn.functional as F
import time
import numpy as np
import copy
import pickle 

from evaluates.attacks.attacker import Attacker
from models.model_templates import ClassificationModelHostHead, ClassificationModelHostTrainableHead
from utils.basic_functions import cross_entropy_for_onehot, append_exp_res

# class BatchLabelReconstruction(Attacker):
#     def __init__(self, args, index, local_model):
#         super().__init__(args, index, local_model)
#         self.device = args.device
#         self.index = index
#         self.lr = args.attack_configs['lr']
#         self.epochs = args.attack_configs['epochs']
#         self.label_size = args.num_classes
#         self.dummy_active_top_trainable_model = ClassificationModelHostTrainableHead(args.k*args.num_classes, args.num_classes).to(args.device)
#         self.optimizer_trainable = None # construct later
#         self.dummy_active_top_non_trainable_model = ClassificationModelHostHead().to(args.device)
#         self.optimizer_non_trainable = None # construct later
#         self.criterion = cross_entropy_for_onehot
#         self.exp_res_dir = args.exp_res_dir + f'attack/BLR/{self.index}/'
#         if not os.path.exists(self.exp_res_dir):
#             os.makedirs(self.exp_res_dir)
#         self.file_name = 'attack_result.txt'
#         self.exp_res_path = self.exp_res_dir + self.file_name
    
#     def attack(self, self_data, original_dy_dx, *params):
#         pred_a, _ = params
        
#         print(pred_a.size())
#         sample_count = pred_a.size()[0]
#         dummy_pred_b = torch.randn(pred_a.size()).to(self.device).requires_grad_(True)
#         dummy_label = torch.randn((sample_count,self.label_size)).to(self.device).requires_grad_(True)

#         self.optimizer_trainable = torch.optim.Adam([dummy_pred_b, dummy_label] + list(self.dummy_active_top_trainable_model.parameters()), lr=self.lr)
#         self.optimizer_non_trainable = torch.optim.Adam([dummy_pred_b, dummy_label], lr=self.lr)

#         # original_dy_dx = torch.autograd.grad(pred_a, self.net_a.parameters(), grad_outputs=pred_a_gradients_clone)

#         recovery_history = []
#         # passive party does not whether
#         for i, (dummy_model, optimizer) in enumerate(zip([self.dummy_active_top_trainable_model,self.dummy_active_top_non_trainable_model],[self.optimizer_trainable,self.optimizer_non_trainable])):
#             start_time = time.time()
#             for iters in range(1, self.epochs + 1):
#                 # print(f"in BLR, i={i}, iter={iters}")
#                 self.attacker_party_local_model.eval()
#                 may_converge = True
#                 def closure():
#                     optimizer.zero_grad()
#                     dummy_pred = dummy_model([self.attacker_party_local_model(self_data), dummy_pred_b])
#                     dummy_onehot_label = F.softmax(dummy_label, dim=-1)
#                     dummy_loss = self.criterion(dummy_pred, dummy_onehot_label)
#                     dummy_dy_dx_a = torch.autograd.grad(dummy_loss, self.attacker_party_local_model.parameters(), create_graph=True)
#                     grad_diff = 0
#                     # for e in dummy_dy_dx_a:
#                     #     print(e.size(),"**")
#                     # for e in original_dy_dx:
#                     #     print(e.size(),"***")
#                     for (gx, gy) in zip(dummy_dy_dx_a, original_dy_dx):
#                         grad_diff += ((gx - gy) ** 2).sum()
#                         if grad_diff > 1e8:
#                             # may_converge = False
#                             break
#                     grad_diff.backward()
#                     return grad_diff
#                 optimizer.step(closure)
#                 if may_converge == False:
#                     print("may_not converge, break")
#                     break
#                 self.attacker_party_local_model.train()
                
#                 # if self.early_stop == True:
#                 #     if closure().item() < self.early_stop_param:
#                 #         break
#             print("appending dummy_label")
#             recovery_history.append(dummy_label)

#         #     rec_rate = self.calc_label_recovery_rate(dummy_label, self.gt_label)
#         #     recovery_history[i].append(dummy_label)
#         #     end_time = time.time()
#         #     print(f'batch_size=%d,class_num=%d,party_index=%d,recovery_rate=%lf,time_used=%lf' % (sample_count, self.label_size, self.index, rec_rate, end_time - start_time))
        
#         # avg_rec_rate_trainable = sum(recovery_rate_history[0])/len(recovery_rate_history[0])
#         # avg_rec_rate_non_trainable = sum(recovery_rate_history[1])/len(recovery_rate_history[1])
#         # best_rec_rate = min(avg_rec_rate_trainable,avg_rec_rate_non_trainable)
#         # exp_result = f"bs|num_class|attack_party_index|recovery_rate,%d|%d|%lf|%s" % (sample_count, self.label_size, self.index, best_rec_rate, str(recovery_rate_history))
#         # append_exp_res(self.exp_res_path, exp_result)
        
#         # return best_rec_rate
#         print("returning from BLI")
#         return recovery_history


class BatchLabelReconstruction(Attacker):
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
        self.dummy_active_top_trainable_model = None
        self.optimizer_trainable = None # construct later
        self.dummy_active_top_non_trainable_model = None
        self.optimizer_non_trainable = None # construct later
        self.criterion = cross_entropy_for_onehot
        self.file_name = 'attack_result.txt'
        self.exp_res_dir = f'exp_result/main/{args.dataset}/attack/BLR/'
        self.exp_res_path = ''
        # self.exp_res_dir = args.exp_res_dir + f'attack/BLR/{self.index}/'
        # if not os.path.exists(self.exp_res_dir):
        #     os.makedirs(self.exp_res_dir)
        # self.exp_res_path = self.exp_res_dir + self.file_name
    
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
        self.set_seed(100)
        for ik in self.party:
            index = ik
            self.exp_res_dir = self.exp_res_dir + f'{index}/'
            if not os.path.exists(self.exp_res_dir):
                os.makedirs(self.exp_res_dir)
            self.exp_res_path = self.exp_res_dir + self.file_name
            
            # collect necessary information
            pred_a = self.vfl_info['predict'][ik]        
            # print(pred_a.size())
            self_data = self.vfl_info['data'][ik][0]
            # print(self_data.size())
            # original_dy = self.vfl_info['gradient'][ik]
            original_dy_dx = self.vfl_info['local_model_gradient'][ik]
            local_model = self.vfl_info['model'][ik]

            # pickle.dump(self.vfl_info, open('./vfl_info.pkl','wb'))
            
            true_label = self.vfl_info['label']
            # print(true_label.size())
            
            local_model_copy = copy.deepcopy(local_model)
            local_model = local_model.to(self.device)
            local_model_copy.eval()
            # new_pred_a = local_model_copy(self_data)
            # original_dy_dx = torch.autograd.grad(new_pred_a, local_model_copy.parameters(), grad_outputs=original_dy, retain_graph=True)

            sample_count = pred_a.size()[0]
            # dummy_pred_b_top_trainabel_model = torch.randn(pred_a.size()).to(self.device).requires_grad_(True)
            # dummy_label_top_trainabel_model = torch.randn((sample_count,self.label_size)).to(self.device).requires_grad_(True)
            # dummy_pred_b_no_top_trainabel_model = torch.randn(pred_a.size()).to(self.device).requires_grad_(True)
            # dummy_label_no_top_trainabel_model = torch.randn((sample_count,self.label_size)).to(self.device).requires_grad_(True)

            # self.dummy_active_top_trainable_model = ClassificationModelHostTrainableHead(self.k*self.num_classes, self.num_classes).to(self.device)
            # self.optimizer_trainable = torch.optim.Adam([dummy_pred_b_top_trainabel_model, dummy_label_top_trainabel_model] + list(self.dummy_active_top_trainable_model.parameters()), lr=self.lr)
            # # self.optimizer_trainable = torch.optim.SGD([dummy_pred_b_top_trainabel_model, dummy_label_top_trainabel_model] + list(self.dummy_active_top_trainable_model.parameters()), lr=self.lr)
            # self.dummy_active_top_non_trainable_model = ClassificationModelHostHead().to(self.device)
            # self.optimizer_non_trainable = torch.optim.Adam([dummy_pred_b_no_top_trainabel_model, dummy_label_no_top_trainabel_model], lr=self.lr)
            # # self.optimizer_non_trainable = torch.optim.SGD([dummy_pred_b_no_top_trainabel_model, dummy_label_no_top_trainabel_model], lr=self.lr)


            recovery_history = []
            recovery_rate_history = [[], []]
            # passive party does not whether
            # for i, (dummy_model, optimizer, dummy_pred_b, dummy_label) in \
            #             enumerate(zip([self.dummy_active_top_non_trainable_model,self.dummy_active_top_trainable_model],\
            #                         [self.optimizer_non_trainable,self.optimizer_trainable],\
            #                         [dummy_pred_b_no_top_trainabel_model,dummy_pred_b_top_trainabel_model],\
            #                         [dummy_label_no_top_trainabel_model,dummy_label_top_trainabel_model])):
            for i in range(2):
                if i == 0: #non_trainable_top_model
                    dummy_pred_b = torch.randn(pred_a.size()).to(self.device).requires_grad_(True)
                    dummy_label = torch.randn((sample_count,self.label_size)).to(self.device).requires_grad_(True)
                    dummy_model = ClassificationModelHostHead().to(self.device)
                    optimizer = torch.optim.Adam([dummy_pred_b, dummy_label], lr=self.lr)
                else:
                    assert i == 1 #trainable_top_model
                    dummy_pred_b = torch.randn(pred_a.size()).to(self.device).requires_grad_(True)
                    dummy_label = torch.randn((sample_count,self.label_size)).to(self.device).requires_grad_(True)
                    dummy_model = ClassificationModelHostTrainableHead(self.k*self.num_classes, self.num_classes).to(self.device)
                    optimizer = torch.optim.Adam([dummy_pred_b, dummy_label] + list(dummy_model.parameters()), lr=self.lr)

                print(f"BLI iteration for type{i}")
                start_time = time.time()
                for iters in range(1, self.epochs + 1):
                    # print(f"in BLR, i={i}, iter={iters}")
                    def closure():
                        optimizer.zero_grad()
                        dummy_pred = dummy_model([local_model_copy(self_data), dummy_pred_b])
                        
                        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
                        dummy_loss = self.criterion(dummy_pred, dummy_onehot_label)
                        # print(f"dummy_predict={dummy_pred},dummy_onehot_label={dummy_onehot_label},dummy_loss={dummy_loss}")
                        dummy_dy_dx_a = torch.autograd.grad(dummy_loss, local_model_copy.parameters(), create_graph=True)
                        grad_diff = 0
                        # for e in dummy_dy_dx_a:
                        #     print(e.size(),"**")
                        # for e in original_dy_dx:
                        #     print(e.size(),"***")
                        for (gx, gy) in zip(dummy_dy_dx_a, original_dy_dx):
                            grad_diff += ((gx - gy) ** 2).sum()
                            # if grad_diff > 1e8:
                            #     # may_converge = False
                            #     break
                        # print(f"grad_diff={grad_diff}")
                        grad_diff.backward()
                        return grad_diff
                    
                    # rec_rate = self.calc_label_recovery_rate(dummy_label, true_label)
                    # print(f"iter={iter}::rec_rate={rec_rate}")
                    optimizer.step(closure)
                    
                    # if self.early_stop == True:
                    #     if closure().item() < self.early_stop_param:
                    #         break
                # print("appending dummy_label")
                recovery_history.append(dummy_label)

                # print(dummy_label, true_label)

                rec_rate = self.calc_label_recovery_rate(dummy_label, true_label)
                recovery_rate_history[i].append(rec_rate)
                end_time = time.time()
                print(f'batch_size=%d,class_num=%d,party_index=%d,recovery_rate=%lf,time_used=%lf' % (sample_count, self.label_size, index, rec_rate, end_time - start_time))
            
            avg_rec_rate_trainable = sum(recovery_rate_history[0])/len(recovery_rate_history[0])
            avg_rec_rate_non_trainable = sum(recovery_rate_history[1])/len(recovery_rate_history[1])
            best_rec_rate = max(avg_rec_rate_trainable,avg_rec_rate_non_trainable)
            print(f"BLI, if self.args.apply_defense={self.args.apply_defense}")
            if self.args.apply_defense == True:
                exp_result = f"bs|num_class|attack_party_index|recovery_rate,%d|%d|%d|%lf|%s (AttackConfig: %s) (Defense: %s %s)" % (sample_count, self.label_size, index, best_rec_rate, str(recovery_rate_history), str(self.args.attack_configs), self.args.defense_name, str(self.args.defense_configs))
            else:
                exp_result = f"bs|num_class|attack_party_index|recovery_rate,%d|%d|%d|%lf|%s" % (sample_count, self.label_size, index, best_rec_rate, str(recovery_rate_history))
            append_exp_res(self.exp_res_path, exp_result)
        
        # return best_rec_rate
        print("returning from BLI")
        # return recovery_history