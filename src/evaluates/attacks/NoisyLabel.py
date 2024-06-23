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
from models.global_models import *  # ClassificationModelHostHead, ClassificationModelHostTrainableHead
from utils.basic_functions import cross_entropy_for_onehot, append_exp_res


class NoisyLabel(Attacker):
    def __init__(self, top_vfl, args):
        super().__init__(args)
        self.args = args
        # get information for launching BLI attack
        self.vfl = top_vfl
        # prepare parameters
        self.device = args.device
        self.num_classes = args.num_classes
        self.k = args.k

        self.file_name = 'attack_result.txt'
        self.exp_res_dir = f'exp_result/main/{args.dataset}/attack/NL/'
        self.exp_res_path = ''

        if not os.path.exists(self.exp_res_dir):
            os.makedirs(self.exp_res_dir)
        self.exp_res_path = self.exp_res_dir + self.file_name

    def set_seed(self, seed=0):
        # random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def attack(self):
        self.set_seed(123)
        sample_count = self.vfl.batch_size
        noise_rate = self.args.attack_configs['noise_rate']
        noise_type = self.args.attack_configs['noise_type']

        print(f"NoisyLabel, self.args.apply_defense={self.args.apply_defense}")
        if self.args.apply_defense == True:
            exp_result = f"bs|num_class|Q|top_trainable|final_epoch|lr|acc,%d|%d|%d|%d|%d|%lf|%s|%lf|%lf|%s (AttackConfig: %s) (Defense: %s %s)" % (
            sample_count, self.num_classes, self.args.Q, self.args.apply_trainable_layer, self.vfl.epochs, self.vfl.lr,
            noise_type, noise_rate, self.vfl.test_acc, str(self.args.attack_configs), self.args.defense_name,
            str(self.args.defense_configs))

        else:
            exp_result = f"bs|num_class|Q|top_trainable|final_epoch|lr|noise_type|noise_rate|acc,%d|%d|%d|%d|%d|%lf|%s|%lf %lf" % (
            sample_count, self.num_classes, self.args.Q, self.args.apply_trainable_layer, self.vfl.epochs, self.vfl.lr,
            noise_type, noise_rate, self.vfl.test_acc)
        print(f'batch_size=%d,class_num=%d,noise_type=%s, noise_rate=%lf, main_acc=%lf' % (
        sample_count, self.num_classes, noise_type, noise_rate, self.vfl.test_acc))

        append_exp_res(self.exp_res_path, exp_result)

        print("returning from NL")
