import sys, os
sys.path.append(os.pardir)

import torch
import torch.nn.functional as F
import time
import numpy as np

from evaluates.attacks.attacker import Attacker
from models.model_templates import ClassificationModelHostHead, ClassificationModelHostTrainableHead
from utils.basic_functions import cross_entropy_for_onehot, append_exp_res

class BatchLabelReconstruction(Attacker):
    def __init__(self, args, index, local_model):
        super().__init__(args, index, local_model)
        self.device = args.device
        self.index = index
        self.lr = args.attack_configs['lr']
        self.epochs = args.attack_configs['epochs']
        self.label_size = args.num_classes
        self.dummy_active_top_trainable_model = ClassificationModelHostTrainableHead(args.k*args.num_classes, args.num_classes).to(args.device)
        self.optimizer_trainable = None # construct later
        self.dummy_active_top_non_trainable_model = ClassificationModelHostHead().to(args.device)
        self.optimizer_non_trainable = None # construct later
        self.criterion = cross_entropy_for_onehot
        self.exp_res_dir = args.exp_res_dir + f'attack/BLR/{self.index}/'
        if not os.path.exists(self.exp_res_dir):
            os.makedirs(self.exp_res_dir)
        self.file_name = 'attack_result.txt'
        self.exp_res_path = self.exp_res_dir + self.file_name
    
    def attack(self, self_data, original_dy_dx, *params):
        pred_a, _ = params
        
        print(pred_a.size())
        sample_count = pred_a.size()[0]
        dummy_pred_b = torch.randn(pred_a.size()).to(self.device).requires_grad_(True)
        dummy_label = torch.randn((sample_count,self.label_size)).to(self.device).requires_grad_(True)

        self.optimizer_trainable = torch.optim.Adam([dummy_pred_b, dummy_label] + list(self.dummy_active_top_trainable_model.parameters()), lr=self.lr)
        self.optimizer_non_trainable = torch.optim.Adam([dummy_pred_b, dummy_label], lr=self.lr)

        # original_dy_dx = torch.autograd.grad(pred_a, self.net_a.parameters(), grad_outputs=pred_a_gradients_clone)

        recovery_history = []
        # passive party does not whether
        for i, (dummy_model, optimizer) in enumerate(zip([self.dummy_active_top_trainable_model,self.dummy_active_top_non_trainable_model],[self.optimizer_trainable,self.optimizer_non_trainable])):
            start_time = time.time()
            for iters in range(1, self.epochs + 1):
                # print(f"in BLR, i={i}, iter={iters}")
                self.attacker_party_local_model.eval()
                may_converge = True
                def closure():
                    optimizer.zero_grad()
                    dummy_pred = dummy_model([self.attacker_party_local_model(self_data), dummy_pred_b])
                    dummy_onehot_label = F.softmax(dummy_label, dim=-1)
                    dummy_loss = self.criterion(dummy_pred, dummy_onehot_label)
                    dummy_dy_dx_a = torch.autograd.grad(dummy_loss, self.attacker_party_local_model.parameters(), create_graph=True)
                    grad_diff = 0
                    # for e in dummy_dy_dx_a:
                    #     print(e.size(),"**")
                    # for e in original_dy_dx:
                    #     print(e.size(),"***")
                    for (gx, gy) in zip(dummy_dy_dx_a, original_dy_dx):
                        grad_diff += ((gx - gy) ** 2).sum()
                        if grad_diff > 1e8:
                            # may_converge = False
                            break
                    grad_diff.backward()
                    return grad_diff
                optimizer.step(closure)
                if may_converge == False:
                    print("may_not converge, break")
                    break
                self.attacker_party_local_model.train()
                
                # if self.early_stop == True:
                #     if closure().item() < self.early_stop_param:
                #         break
            print("appending dummy_label")
            recovery_history.append(dummy_label)

        #     rec_rate = self.calc_label_recovery_rate(dummy_label, self.gt_label)
        #     recovery_history[i].append(dummy_label)
        #     end_time = time.time()
        #     print(f'batch_size=%d,class_num=%d,party_index=%d,recovery_rate=%lf,time_used=%lf' % (sample_count, self.label_size, self.index, rec_rate, end_time - start_time))
        
        # avg_rec_rate_trainable = sum(recovery_rate_history[0])/len(recovery_rate_history[0])
        # avg_rec_rate_non_trainable = sum(recovery_rate_history[1])/len(recovery_rate_history[1])
        # best_rec_rate = min(avg_rec_rate_trainable,avg_rec_rate_non_trainable)
        # exp_result = f"bs|num_class|attack_party_index|recovery_rate,%d|%d|%lf|%s" % (sample_count, self.label_size, self.index, best_rec_rate, str(recovery_rate_history))
        # append_exp_res(self.exp_res_path, exp_result)
        
        # return best_rec_rate
        print("returning from BLI")
        return recovery_history

