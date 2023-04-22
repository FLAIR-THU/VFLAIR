import os
import sys
import numpy as np
import time

import random
import logging
import argparse
import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from torchvision import datasets
# import torch.utils
# import torch.backends.cudnn as cudnn
# from tensorboardX import SummaryWriter

from load.LoadConfigs import * #load_configs
from load.LoadParty import load_parties
from evaluates.MainTaskVFL import *
from evaluates.MainTaskVFLwithBackdoor import *
from utils.basic_functions import append_exp_res
import warnings
warnings.filterwarnings("ignore")

TARGETED_BACKDOOR = ['ReplacementBackdoor'] # main_acc  backdoor_acc
UNTARGETED_BACKDOOR = ['NoisyLabel','MissingFeature'] # main_acc
LABEL_INFERENCE = ['BatchLabelReconstruction','DataLabelReconstruction','NormbasedScoring','DirectionbasedScoring','PassiveModelCompletion'] # label_recovery

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def evaluate_no_attack(args):
    # No Attack
    args = load_attack_configs(args.configs, args, -1)
    args = load_parties(args)

    vfl = MainTaskVFL(args)
    if args.dataset not in ['cora']:
        main_acc = vfl.train()
    else:
        main_acc = vfl.train_graph()

    main_acc_noattack = main_acc
    attack_metric = main_acc_noattack - main_acc
    attack_metric_name = 'acc_loss'
    # Save record 
    exp_result = f"K|bs|LR|num_class|Q|top_trainable|epoch|attack_name|{args.attack_param_name}|main_task_acc|{attack_metric_name},%d|%d|%lf|%d|%d|%d|%d|{args.attack_name}|{args.attack_param}|{main_acc}|{attack_metric}" %\
        (args.k,args.batch_size, args.main_lr, args.num_classes, args.Q, args.apply_trainable_layer,args.main_epochs)
    print(exp_result)
    append_exp_res(args.exp_res_path, exp_result)
    return vfl, main_acc_noattack


def evaluate_label_inference(args):
    # Basic VFL Training Pipeline
    i=0
    for index in args.label_inference_index:
        args = load_attack_configs(args.configs, args, index)
        # args = load_parties(args)
        print('======= Test Attack',index,': ',args.attack_name,' =======')
        print('attack configs:',args.attack_configs)
        if args.attack_name == 'PassiveModelCompletion':
            args.need_auxiliary = 1
            args = load_parties(args) # include load dataset with auxiliary data
            vfl = MainTaskVFL(args)
            if args.dataset not in ['cora']:
                main_acc = vfl.train()
            else:
                main_acc = vfl.train_graph()
        else:  
            args.need_auxiliary = 0
            args = load_parties(args)
            # if i == 0: # Only train once for all label_inference_attack
            #     vfl = MainTaskVFL(args)
            #     if args.dataset not in ['cora']:
            #         main_acc = vfl.train()
            #     else:
            #         main_acc = vfl.train_graph()
            #     i = i + 1
            vfl = args.basic_vfl
            main_acc = args.main_acc_noattack
        
        attack_metric = vfl.evaluate_attack()
        attack_metric_name = 'label_recovery_rate'
    
        # Save record for different defense method
        exp_result = f"K|bs|LR|num_class|Q|top_trainable|epoch|attack_name|{args.attack_param_name}|main_task_acc|{attack_metric_name},%d|%d|%lf|%d|%d|%d|%d|{args.attack_name}|{args.attack_param}|{main_acc}|{attack_metric}" %\
            (args.k,args.batch_size, args.main_lr, args.num_classes, args.Q, args.apply_trainable_layer,args.main_epochs)
        print(exp_result)
        append_exp_res(args.exp_res_path, exp_result)


def evaluate_untargeted_backdoor(args):
    
    for index in args.untargeted_backdoor_index:
        args = load_attack_configs(args.configs, args, index)
        args = load_parties(args)
        
        print('======= Test Attack',index,': ',args.attack_name,' =======')
        print('attack configs:',args.attack_configs)

        vfl = MainTaskVFL(args)
        if args.dataset not in ['cora']:
            main_acc = vfl.train()
        else:
            main_acc = vfl.train_graph()

        attack_metric = args.main_acc_noattack - main_acc
        attack_metric_name = 'acc_loss'
        # Save record for different defense method
        exp_result = f"K|bs|LR|num_class|Q|top_trainable|epoch|attack_name|{args.attack_param_name}|main_task_acc|{attack_metric_name},%d|%d|%lf|%d|%d|%d|%d|{args.attack_name}|{args.attack_param}|{main_acc}|{attack_metric}" %\
            (args.k,args.batch_size, args.main_lr, args.num_classes, args.Q, args.apply_trainable_layer,args.main_epochs)
        print(exp_result)
        append_exp_res(args.exp_res_path, exp_result)


def evaluate_targeted_backdoor(args):
    # mark that backdoor data is never prepared
    args.target_label = None
    args.train_poison_list = None
    args.train_target_list = None
    args.test_poison_list = None
    args.test_target_list = None
    for index in args.targeted_backdoor_index:
        args = load_attack_configs(args.configs, args, index)
        args = load_parties(args)
        print('======= Test Attack',index,': ',args.attack_name,' =======')
        print('attack configs:',args.attack_configs)

        # Targeted Backdoor VFL Training pipeline
        if args.apply_backdoor == True:
            vfl = MainTaskVFLwithBackdoor(args)
            main_acc, backdoor_acc = vfl.train()
        else:
            vfl = MainTaskVFL(args)
            if args.dataset not in ['cora']:
                main_acc = vfl.train()
            else:
                main_acc = vfl.train_graph()
        
        attack_metric = backdoor_acc
        attack_metric_name = 'backdoor_acc'
       
        # Save record for different defense method
        exp_result = f"K|bs|LR|num_class|Q|top_trainable|epoch|attack_name|{args.attack_param_name}|main_task_acc|{attack_metric_name},%d|%d|%lf|%d|%d|%d|%d|{args.attack_name}|{args.attack_param}|{main_acc}|{attack_metric}" %\
            (args.k,args.batch_size, args.main_lr, args.num_classes, args.Q, args.apply_trainable_layer,args.main_epochs)
        print(exp_result)
        append_exp_res(args.exp_res_path, exp_result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("backdoor")
    parser.add_argument('--device', type=str, default='cpu', help='use gpu or cpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--seed', type=int, default=97, help='random seed')
    parser.add_argument('--configs', type=str, default='test_attack', help='configure json file path')
    parser.add_argument('--save_model', type=bool, default=False, help='whether to save the trained model')
    args = parser.parse_args()

    for seed in range(97,98):
        set_seed(seed)

        if args.device == 'cuda':
            cuda_id = args.gpu
            torch.cuda.set_device(cuda_id)
            print(f'running on cuda{torch.cuda.current_device()}')
        else:
            print('running on cpu')

        
        ####### load configs from *.json files #######
        ############ Basic Configs ############
        args = load_basic_configs(args.configs, args)
        args.need_auxiliary = 0 # no auxiliary dataset for attacker
        for mode in [0,1]:
            if mode == 0:
                args.global_model = 'ClassificationModelHostHead'
            else:
                args.global_model = 'ClassificationModelHostTrainableHead'
            args.apply_trainable_layer = mode

            print('============ apply_trainable_layer=',args.apply_trainable_layer,'============')
            print('================================\n')
        
            assert args.dataset_split != None, "dataset_split attribute not found config json file"
            assert 'dataset_name' in args.dataset_split, 'dataset not specified, please add the name of the dataset in config json file'
            args.dataset = args.dataset_split['dataset_name']
            print(args.dataset)

            print('======= Defense ========')
            print('Defense_Name:',args.defense_name)
            print('Defense_Config:',str(args.defense_configs))

            print('===== Total Attack Tested:',args.attack_num,' ======')
            print('targeted_backdoor:',args.targeted_backdoor_list,args.targeted_backdoor_index)
            print('untargeted_backdoor:',args.untargeted_backdoor_list,args.untargeted_backdoor_index)
            print('label_inference:',args.label_inference_list,args.label_inference_index)
            print('=================================')

            # Save record for different defense method
            args.exp_res_dir = f'exp_result/{args.dataset}/'
            if not os.path.exists(args.exp_res_dir):
                os.makedirs(args.exp_res_dir)
            filename = f'{args.defense_name}_{args.defense_param},model={args.model_list[str(0)]["type"]}.txt'
            args.exp_res_path = args.exp_res_dir + filename

            args.basic_vfl,args.main_acc_noattack = evaluate_no_attack(args)
            if args.label_inference_list != []:
                evaluate_label_inference(args)

            if args.untargeted_backdoor_list != []:
                evaluate_untargeted_backdoor(args)

            if args.targeted_backdoor_list != []:
                evaluate_targeted_backdoor(args)

    


    
    
    
