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
        self.early_stop = args.attack_configs['early_stop'] if 'early_stop' in args.attack_configs else 0
        self.early_stop_threshold = args.attack_configs['early_stop_threshold'] if 'early_stop_threshold' in args.attack_configs else 1e-7
        self.label_size = args.num_classes
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
            self.exp_res_dir = self.exp_res_dir + f'{index}/'
            if not os.path.exists(self.exp_res_dir):
                os.makedirs(self.exp_res_dir)
            self.exp_res_path = self.exp_res_dir + self.file_name
            
            # collect necessary information
            pred_a = self.vfl_info['predict'][ik]        
            self_data = self.vfl_info['data'][ik][0]
            original_dy_dx = self.vfl_info['local_model_gradient'][ik]
            local_model = self.vfl_info['model'][ik]
            true_label = self.vfl_info['label'].to(self.device)
            
            local_model_copy = copy.deepcopy(local_model)
            local_model = local_model.to(self.device)
            local_model_copy.eval()

            # ################## debug: for checking if saved results are right (start) ##################
            print(f"sample_count = {pred_a.size()[0]}, number of classes = {pred_a.size()[1]}, {self.label_size}")
            # pickle.dump(self.vfl_info, open('./vfl_info.pkl','wb'))
            # original_dy = self.vfl_info['gradient'][ik]
            # new_pred_a = local_model_copy(self_data)
            # new_original_dy_dx = torch.autograd.grad(new_pred_a, local_model_copy.parameters(), grad_outputs=original_dy, retain_graph=True)
            # print(f"predict_error:{torch.nonzero(new_pred_a-pred_a)}")
            # for new_w, w in zip(new_original_dy_dx, original_dy_dx):
            #     print(f"model_weight_error:{torch.nonzero(new_w-w)}")
            # ################## debug: for checking if saved results are right (end) ##################

            sample_count = pred_a.size()[0]
            recovery_history = []
            recovery_rate_history = [[], []]
            for i in range(2):
                if i == 0: #non_trainable_top_model
                    dummy_pred_b = torch.randn(pred_a.size()).to(self.device).requires_grad_(True)
                    dummy_label = torch.randn((sample_count,self.label_size)).to(self.device).requires_grad_(True)
                    dummy_model = ClassificationModelHostHead().to(self.device)
                    optimizer = torch.optim.Adam([dummy_pred_b, dummy_label], lr=self.lr, weight_decay=0.0)
                else:
                    assert i == 1 #trainable_top_model,  can use user define models instead
                    dummy_pred_b = torch.randn(pred_a.size()).to(self.device).requires_grad_(True)
                    dummy_label = torch.randn((sample_count,self.label_size)).to(self.device).requires_grad_(True)
                    dummy_model = ClassificationModelHostTrainableHead(self.k*self.num_classes, self.num_classes).to(self.device)
                    optimizer = torch.optim.Adam([dummy_pred_b, dummy_label] + list(dummy_model.parameters()), lr=self.lr)

                print(f"BLI iteration for type{i}, self.device={self.device}, {dummy_pred_b.device}, {dummy_label.device}")
                start_time = time.time()
                for iters in range(1, self.epochs + 1):
                    # s_time = time.time()
                    def closure():
                        optimizer.zero_grad()
                        dummy_pred = dummy_model([local_model_copy(self_data), dummy_pred_b])
                        
                        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
                        dummy_loss = self.criterion(dummy_pred, dummy_onehot_label)
                        dummy_dy_dx_a = torch.autograd.grad(dummy_loss, local_model_copy.parameters(), create_graph=True)
                        grad_diff = 0
                        for (gx, gy) in zip(dummy_dy_dx_a, original_dy_dx):
                            grad_diff += ((gx - gy) ** 2).sum()
                        grad_diff.backward()
                        return grad_diff
                    
                    rec_rate = self.calc_label_recovery_rate(dummy_label, true_label)
                    # print(f"iter={iters}::rec_rate={rec_rate}")
                    optimizer.step(closure)
                    e_time = time.time()
                    # print(f"in BLR, i={i}, iter={iters}, time={s_time-e_time}")
                    
                    if self.early_stop == 1:
                        if closure().item() < self.early_stop_threshold:
                            break
                
                # print("appending dummy_label")
                # recovery_history.append(dummy_label)
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