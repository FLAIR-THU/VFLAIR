import os
import sys
import numpy as np
import time

import random
import logging
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
# from torch.types import Device
import torch.utils
import torch.backends.cudnn as cudnn
# from tensorboardX import SummaryWriter

from load.LoadConfigs import load_configs, load_attack_configs, load_defense_configs
from load.LoadDataset import load_dataset
from load.LoadModels import load_models
from load.LoadParty import load_parties
from models.vision import *
from utils.basic_functions import *
from utils.constants import *
from utils.dataset.SimpleImageDataset import SimpleDataset
from utils.dataset.NuswideDataset import NUSWIDEDataset
# from evaluates.BatchLabelReconstruction import *
# from evaluates.DeepLeakageFromGradients import *
# from evaluates.ReplacementBackdoor import *
from evaluates.MainTaskVFL import *
from evaluates.MainTaskVFLwithBackdoor import *

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

tp = transforms.ToTensor()
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
     ])
transform_fn = transforms.Compose([
    transforms.ToTensor()
])

if __name__ == '__main__':
    parser = argparse.ArgumentParser("backdoor")
    parser.add_argument('--device', type=str, default='cuda', help='use gpu or cpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--seed', type=int, default=97, help='random seed')
    parser.add_argument('--configs', type=str, default='basic_configs', help='configure json file path')
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == 'cuda':
        cuda_id = args.gpu
        torch.cuda.set_device(cuda_id)
    print(f'running on cuda{torch.cuda.current_device()}')

    # load configs from *_config.json files
    args = load_configs(args.configs, args)
    assert args.dataset_split != None, "dataset_split attribute not found config json file"
    assert 'dataset_name' in args.dataset_split, 'dataset not specified, please add the name of the dataset in config json file'
    args.dataset = args.dataset_split['dataset_name']
    print(args.dataset)
    # print(args.attack_methods)

    # mark that backdoor data is never prepared
    args.target_label = None
    args.train_poison_list = None
    args.train_target_list = None
    args.test_poison_list = None
    args.test_target_list = None
    
    args.exp_res_dir = f'exp_result/main/{args.dataset}/'
    args = load_parties(args)
    if not os.path.exists(args.exp_res_dir):
        os.makedirs(args.exp_res_dir)
    filename = f'partyNum={args.k},model={args.model_list[str(0)]["type"]},lr={args.main_lr},num_exp={args.num_exp},' \
        f'epochs={args.main_epochs}.txt'
    # filename = f'dataset={args.dataset},model={args.model},lr={args.lr},num_exp={args.num_exp},' \
    #        f'epochs={args.epochs},early_stop={args.early_stop}.txt'
    args.exp_res_path = args.exp_res_dir + filename
    
    # if have inference time attack, use another VFL pipeline
    if args.apply_backdoor == True:
        vfl = MainTaskVFLwithBackdoor(args)
        # no other attacks, only backdoor attack, may change later
        args.apply_attack = False
    else:
        vfl = MainTaskVFL(args)
    vfl.train()

    if args.apply_attack == True:
        vfl.evaluate_attack()
    
    
    # # put in all the attacks
    # attack_list = []
    # for attack in args.attack_methods:
    #     # load attack configs
    #     attack_index = args.attack_methods.index(attack)
    #     attack_config_file_path = args.attack_config_list[attack_index]
    #     args = load_attack_configs(attack_config_file_path, attack, args)

    #     args.num_class_list = [(args.dataset_split['num_classes'] if('num_classes' in args.dataset_split) else 2)]
    #     args.batch_size_list = [args.batch_size]

    #     num_classes = args.num_class_list[0] # for main task evaluation
    #     args.num_classes = args.num_class_list[0]
        

    #     for defense in args.defense_methods:
    #         # load defense configs
    #         print("use defense", defense)
    #         defense_index = args.defense_methods.index(defense)
    #         defense_config_file_path = args.defense_config_list[defense_index]
    #         args = load_defense_configs(defense_config_file_path, defense, args)
    #         print("everything loaded")


            # args.models_dict = {"mnist": MLP2,
            #            "cifar100": resnet18,
            #            "cifar10": resnet18,
            #         #    "cifar10": resnet20,
            #            "nuswide": MLP2,
            #            "classifier": None}
            
            # if attack != 'ReplacementBackdoor':
            #     path = args.exp_res_dir+'no_defense_main_task.txt'
            #     test_acc_list = []
            #     for _ in range(args.num_exp):
            #         vfl_defence_image = VFLDefenceExperimentBase(args)
            #         test_acc, parameter = vfl_defence_image.train()
            #         test_acc_list.append(test_acc)
            #     append_exp_res(path, str(parameter) + ' ' + str(np.mean(test_acc_list))+ ' ' + str(test_acc_list) + ' ' + str(np.max(test_acc_list)))





