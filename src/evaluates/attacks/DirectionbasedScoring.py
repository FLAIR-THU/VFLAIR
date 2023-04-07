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
import tensorflow as tf

from evaluates.attacks.attacker import Attacker
from models.global_models import * #ClassificationModelHostHead, ClassificationModelHostTrainableHead
from utils.basic_functions import cross_entropy_for_onehot, append_exp_res
from utils.scoring_attack_functions import update_acc,update_auc,compute_auc,cosine_similarity

def update_all_cosine_leak_auc(cosine_leak_auc_dict, grad_list, pos_grad_list, y):
    for (key, grad, pos_grad) in zip(cosine_leak_auc_dict.keys(), grad_list, pos_grad_list):
        # print(f"in cosine leak, [key, grad, pos_grad] = [{key}, {grad}, {pos_grad}]")
        # flatten each example's grad to one-dimensional
        grad = tf.reshape(grad, shape=(grad.shape[0], -1))
        # there should only be one positive example's gradient in pos_grad
        pos_grad = tf.reshape(pos_grad, shape=(pos_grad.shape[0], -1))

        # auc = update_auc(
        #             y=y,
        #             predicted_value=cosine_similarity(grad, pos_grad),
        #             m_auc=cosine_leak_auc_dict[key])
        predicted_value = cosine_similarity(grad, pos_grad).numpy()
        predicted_label = np.where(predicted_value>0,1,0).reshape(-1)
        _y = y.numpy()
        acc = ((predicted_label==_y).sum()/len(_y))
        # not only update the epoch average above
        # also log this current batch value on the tensorboard
        # if auc:
        #     with shared_var.writer.as_default():
        #         tf.summary.scalar(name=key+'_batch',
        #                           data=auc,
        #                           step=shared_var.counter)
        return acc


class DirectionbasedScoring(Attacker):
    def __init__(self, top_vfl, args):
        super().__init__(args)
        self.args = args
        # get information 
        self.vfl_info = top_vfl.first_epoch_state
        # prepare parameters
        self.device = args.device
        self.num_classes = args.num_classes
        self.k = args.k
        self.party = args.attack_configs['party'] # parties that launch attacks
        #self.num_run = args.attack_configs['num_run'] 
        # self.lr = args.attack_configs['lr']
        # self.epochs = args.attack_configs['epochs']
        # self.early_stop = args.attack_configs['early_stop'] if 'early_stop' in args.attack_configs else 0
        # self.early_stop_threshold = args.attack_configs['early_stop_threshold'] if 'early_stop_threshold' in args.attack_configs else 1e-7
        
        self.label_size = args.num_classes
        self.criterion = cross_entropy_for_onehot
        
        
        self.file_name = 'attack_result.txt'
        self.exp_res_dir = f'exp_result/main/{args.dataset}/attack/DS/'
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
            # self.exp_res_dir = self.exp_res_dir + f'{index}/'
            # if not os.path.exists(self.exp_res_dir):
            #     os.makedirs(self.exp_res_dir)
            # self.exp_res_path = self.exp_res_dir + self.file_name
            
            # collect necessary information
            pred_list = self.vfl_info['predict']
            pred_a = self.vfl_info['predict'][ik] # [copy.deepcopy(self.parties[ik].local_pred_clone) for ik in range(self.k)]
            pred_b = self.vfl_info['predict'][1] # active party 
            
            self_data = self.vfl_info['data'][ik][0]#copy.deepcopy(self.parties_data)
            active_data = self.vfl_info['data'][1][0] # Active party data

            local_gradient = self.vfl_info['gradient'][ik] 
            original_dy_dx = self.vfl_info['local_model_gradient'][ik] # gradient calculated for local model update
            #[copy.deepcopy(self.parties[ik].weights_grad_a) for ik in range(self.k)]
            
            net_a = self.vfl_info['model'][0].to(self.device)
            net_b = self.vfl_info['model'][1].to(self.device)
            global_model = self.vfl_info['global_model'].to(self.device)
            global_model.eval()
            net_a.eval()
            net_b.eval()
            
            global_pred = self.vfl_info['global_pred'].to(self.device)
            pred_a = net_a(self_data).to(self.device).requires_grad_(True)
            pred_b = net_b(active_data).to(self.device).requires_grad_(True) # real pred_b   fake:dummy pred_b

            true_label = self.vfl_info['label'].to(self.device) # copy.deepcopy(self.gt_one_hot_label)

            sample_count = pred_a.size()[0]
            recovery_history = []
            recovery_rate_history = [[], []]
            
            
            #for i in range(2)
                ## Load real global_model
                # if i==0:
                #     active_aggregate_model = ClassificationModelHostHead()
                # else:
                #     assert i ==1
                #     active_aggregate_model = ClassificationModelHostTrainableHead(self.k*self.num_classes, self.num_classes)
                
                    
            active_aggregate_model = global_model
            active_aggregate_model = active_aggregate_model.to(self.device)
            pred = active_aggregate_model([pred_a, pred_b]) # real loss
            loss = self.criterion(pred, true_label)

            ################ scoring attack ################
            start_time = time.time()
            ################ find a positive gradient ################
            pos_idx = np.random.randint(len(true_label))
            while torch.argmax(true_label[pos_idx]) != torch.tensor(1):
                pos_idx += 1
                if pos_idx >= len(true_label):
                    pos_idx -= len(true_label)
            ################ found positive gradient ################

            pred_a_gradients = torch.autograd.grad(loss, pred_a, retain_graph=True)
            pred_a_gradients_clone = pred_a_gradients[0].detach().clone()
            # original_dy_dx = torch.autograd.grad(pred_a, net_a.parameters(), grad_outputs=pred_a_gradients_clone)
            # for kk in range(len(original_dy_dx)):
            #     if original_dy_dx[kk].equal(original_dy_dx_old[kk]):
            #         print('OK')
            #     else:
            #         print('Unmatch')
          
            tf_pred_a_gradients_clone = tf.convert_to_tensor(pred_a_gradients_clone.cpu().numpy())
            tf_true_label = tf.convert_to_tensor([tf.convert_to_tensor(torch.argmax(true_label[i]).cpu().numpy()) for i in range(len(true_label))])

            cosine_leak_acc = update_all_cosine_leak_auc(
                cosine_leak_auc_dict={'only':''},
                grad_list=[tf_pred_a_gradients_clone],
                pos_grad_list=[tf_pred_a_gradients_clone[pos_idx:pos_idx+1]],
                y=tf_true_label)

            end_time = time.time()
                    
            print(f'batch_size=%d,class_num=%d,acc=%lf,time_used=%lf'
                    % (sample_count, self.label_size, cosine_leak_acc,  end_time - start_time))

            #print(f"DS, if self.args.apply_defense={self.args.apply_defense}")
            # if self.args.apply_defense == True:
            #     exp_result = f"bs|num_class|attack_party_index|Q|top_trainable|acc,%d|%d|%d|%d|%d|%lf|%s (AttackConfig: %s) (Defense: %s %s)" % (sample_count, self.label_size, index, self.args.Q, self.args.apply_trainable_layer, cosine_leak_acc, str(self.args.attack_configs), self.args.defense_name, str(self.args.defense_configs))
                
            # else:
            #     exp_result = f"bs|num_class|attack_party_index|Q|top_trainable|acc,%d|%d|%d|%d|%d|%lf" % (sample_count, self.label_size, index, self.args.Q, self.args.apply_trainable_layer, cosine_leak_acc)# str(recovery_rate_history)
            # append_exp_res(self.exp_res_path, exp_result)
        
        # xx = [i for i in range(len(recovery_rate_history[0]))]
        # plt.figure()
        # plt.plot(xx,recovery_rate_history[0],'o-', color='b', alpha=0.8, linewidth=1, label='trainable')
        # plt.plot(xx,recovery_rate_history[1],'o-', color='r', alpha=0.8, linewidth=1, label='non-trainable')
        # plt.legend()
        # plt.savefig('./exp_result/BLI_Recovery_history.png')

        print("returning from DirectionbasedScoring")
        return cosine_leak_acc
        