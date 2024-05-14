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
from models.global_models import *  # ClassificationModelHostHead, ClassificationModelHostTrainableHead
from utils.basic_functions import cross_entropy_for_onehot, append_exp_res
from utils.scoring_attack_functions import cosine_similarity
from sklearn.metrics import roc_auc_score


def update_all_cosine_leak_auc(cosine_leak_auc_dict, grad_list, pos_grad_list, y):
    for (key, grad, pos_grad) in zip(cosine_leak_auc_dict.keys(), grad_list, pos_grad_list):
        # print(f"in cosine leak, [key, grad, pos_grad] = [{key}, {grad}, {pos_grad}]")
        # flatten each example's grad to one-dimensional
        grad = tf.reshape(grad, shape=(grad.shape[0], -1))
        # there should only be one positive example's gradient in pos_grad
        pos_grad = tf.reshape(pos_grad, shape=(pos_grad.shape[0], -1))
        # print('====== update_all_cosine_leak_auc ==========')
        # print('grad:',grad.shape)
        # print('pos_grad:',pos_grad.shape)
        # auc = update_auc(
        #             y=y,
        #             predicted_value=cosine_similarity(grad, pos_grad),
        #             m_auc=cosine_leak_auc_dict[key])

        # print(f"[debug] in update_all_cosine_leak_auc, grad.shape={grad.shape}, pos_grad.shape={pos_grad.shape}, y.shape={y.shape}")
        # print(f'[debug] pos_grad={pos_grad}')
        # print(f'[deubg] possitive sample is of ratio {sum(y.numpy())/grad.shape[0]}')
        predicted_value = cosine_similarity(grad, pos_grad).numpy()

        predicted_label = np.where(predicted_value > 0, 1, 0).reshape(-1)
        _y = y.numpy()
        acc = ((predicted_label == _y).sum() / len(_y))
        # print(f'[debug] grad=[')
        # for _grad, _lable, _pred in zip(grad,y, predicted_label):
        #     print(_grad, _lable, _pred)
        # print("]")

        predicted_value = tf.reshape(predicted_value, shape=(-1))
        if tf.reduce_sum(y) == 0:  # no positive examples in this batch
            return None
        val_max = tf.math.reduce_max(predicted_value)
        val_min = tf.math.reduce_min(predicted_value)
        pred = (predicted_value - val_min + 1e-16) / (val_max - val_min + 1e-16)
        auc = roc_auc_score(y_true=y.numpy(), y_score=pred.numpy())

        return acc, auc


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
        self.party = args.attack_configs['party']  # parties that launch attacks
        # self.num_run = args.attack_configs['num_run']
        # self.lr = args.attack_configs['lr']
        # self.epochs = args.attack_configs['epochs']
        # self.early_stop = args.attack_configs['early_stop'] if 'early_stop' in args.attack_configs else 0
        # self.early_stop_threshold = args.attack_configs['early_stop_threshold'] if 'early_stop_threshold' in args.attack_configs else 1e-7

        self.label_size = args.num_classes
        self.criterion = cross_entropy_for_onehot

        self.file_name = 'attack_result.txt'
        self.exp_res_dir = f'exp_result/main/{args.dataset}/attack/DS/'
        self.exp_res_path = ''

    def set_seed(self, seed=0):
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
        for ik in self.party:  # attacker party #ik
            index = ik

            # collect necessary information

            true_label = self.vfl_info['label'].to(self.device)  # copy.deepcopy(self.gt_one_hot_label)

            print('true_label:', true_label.size())
            ################ scoring attack ################
            start_time = time.time()
            ################ find a positive gradient ################
            pos_idx = np.random.randint(len(true_label))
            print('pos_idx init:', pos_idx)
            while torch.argmax(true_label[pos_idx]) != torch.tensor(1):
                pos_idx += 1
                if pos_idx >= len(true_label):
                    pos_idx -= len(true_label)
            print('pos_idx after:', pos_idx)
            ################ found positive gradient ################

            pred_a_gradients_clone = self.vfl_info['gradient'][ik]
            # original_dy_dx = torch.autograd.grad(pred_a, net_a.parameters(), grad_outputs=pred_a_gradients_clone)
            # for kk in range(len(original_dy_dx)):
            #     if original_dy_dx[kk].equal(original_dy_dx_old[kk]):
            #         print('OK')
            #     else:
            #         print('Unmatch')
            print('pred_a_gradients_clone:', pred_a_gradients_clone.size())

            tf_pred_a_gradients_clone = tf.convert_to_tensor(pred_a_gradients_clone.cpu().numpy())
            tf_true_label = tf.convert_to_tensor(
                [tf.convert_to_tensor(torch.argmax(true_label[i]).cpu().numpy()) for i in range(len(true_label))])

            print('tf_true_label:', tf_true_label.shape)

            cosine_leak_acc, cosine_leak_auc = update_all_cosine_leak_auc(
                cosine_leak_auc_dict={'only': ''},
                grad_list=[tf_pred_a_gradients_clone],
                pos_grad_list=[tf_pred_a_gradients_clone[pos_idx:pos_idx + 1]],  #
                y=tf_true_label)

            end_time = time.time()

            print(f'batch_size=%d,class_num=%d,acc=%lf,time_used=%lf'
                  % (self.args.batch_size, self.label_size, cosine_leak_acc, end_time - start_time))

        print("returning from DirectionbasedScoring")
        return cosine_leak_acc, cosine_leak_auc
