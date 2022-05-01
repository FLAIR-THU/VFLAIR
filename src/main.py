import os
import sys
import numpy as np
import time

import random
import logging
import argparse
import torch.nn as nn
from torch.types import Device
import torch.utils
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import utils
import copy

import torch

from load.LoadConfigs import load_configs, load_attack_configs, load_defense_configs
from load.LoadDataset import load_dataset
from load.LoadModels import load_models
from models.vision import *
from utils.basic_functions import *
from utils.constants import *
from evaluates.BatchLabelReconstruction import *
from evaluates.DeepLeakageFromGradients import *

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



if __name__ == '__main__':
    parser = argparse.ArgumentParser("backdoor")
    parser.add_argument('--device', type=str, default='cuda', help='use gpu or cpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--seed', type=int, default=97, help='random seed')
    parser.add_argument('--configs', type=str, default='basic_configs', help='configure json file path')
    # ####### add to config file #######
    # parser.add_argument('--defense_up', type=int, default=0)
    # parser.add_argument('--epochs', type=int, default=20, help='num of training epochs')
    # parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    # parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
    # parser.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate')
    # parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
    # parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
    # parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
    # parser.add_argument('--k', type=int, default=2, help='num of participants')
    # parser.add_argument('--model', default='mlp2', help='resnet')
    # parser.add_argument('--input_size', type=int, default=28, help='resnet')
    # parser.add_argument('--use_project_head', type=int, default=0)
    # parser.add_argument('--dp_type', type=str, default='none', help='[laplace, gaussian]')
    # parser.add_argument('--dp_strength', type=float, default=0, help='[0.1, 0.075, 0.05, 0.025,...]')
    # parser.add_argument('--gradient_sparsification', type=float, default=0)
    # parser.add_argument("--certify", type=int, default=0, help="CertifyFLBaseline")
    # parser.add_argument("--sigma", type=float, default=0, help='sigma for certify')
    # parser.add_argument("--M", type=int, default=1000, help="voting party count in CertifyFL")
    # parser.add_argument("--certify_start_epoch", type=int, default=1, help="number of epoch when the cerfity ClipAndPerturb start")
    # parser.add_argument('--autoencoder', type=int, default=0)
    # parser.add_argument('--lba', type=float, default=0)

    # parser.add_argument('--backdoor_type', type=str, default='adjust', help="type of obtaining backdoor triger")
    # parser.add_argument('--backdoor_scale', type=float, default=1.0, help="the color of backdoor triger")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == 'cuda':
        cuda_id = args.gpu
        torch.cuda.set_device(cuda_id)
    print(f'running on cuda{torch.cuda.current_device()}')

    # args.model = 'MLP2'
    # args.dataset = 'mnist'
    # args.num_classes = 10
    # args.batch_size = 64
    # args.epochs = 4000
    # args.lr = 0.01
    # args.num_exp = 10
    # args.early_stop = False
    # args.early_stop_param = 0.0001

    # args.apply_trainable_layer = 0
    # args.apply_laplace = 0
    # args.apply_gaussian = 0
    # args.dp_strength = 0.0
    # args.apply_grad_spar = 0
    # args.grad_spars = 0.0
    # args.apply_encoder = 0
    # args.apply_adversarial_encoder = 0
    # args.ae_lambda = 0.1
    # args.encoder = None
    # args.apply_marvell = 0
    # args.marvell_s = 0

    args = load_configs(args.configs, args)
    assert args.dataset_split != None, "dataset_split attribute not found config json file"
    assert 'dataset_name' in args.dataset_split, 'dataset not specified, please add the name of the dataset in config json file'
    args.dataset = args.dataset_split['dataset_name']
    print(args.dataset)
    print(args.attack_methods)
    # put in all the attacks
    attack_list = []
    for attack in args.attack_methods:
        # load attack configs
        attack_index = args.attack_methods.index(attack)
        attack_config_file_path = args.attack_config_list[attack_index]
        args = load_attack_configs(attack_config_file_path, attack, args)

        args.num_class_list = [(args.dataset_split['num_classes'] if('num_classes' in args.dataset_split) else 2)]
        args.batch_size_list = [args.batch_size]

        if args.dataset == 'cifar10':
            args.dst = datasets.CIFAR10("./data/", download=True)
            # args.batch_size_list = [2048] #[2, 32, 128, 512, 2048]
            # args.num_class_list = [2] #[5, 10, 15, 20, 40, 60, 80, 100]
        elif args.dataset == 'cifar100':
            args.dst = datasets.CIFAR100("./data/", download=True)
            # args.batch_size_list = [2048] #[2, 32, 128, 512, 2048]
            # args.num_class_list = [2] #[5, 10, 15, 20, 40, 60, 80, 100]
        elif args.dataset == 'mnist':
            args.dst = datasets.MNIST("~/.torch", download=True)
            # args.batch_size_list = [32, 128, 512, 1024, 2048]
            # args.num_class_list = [2] #[2, 3, 4, 5, 6, 7, 8, 9, 10]
        elif args.dataset == 'nuswide':
            args.dst = None
            # args.batch_size_list = [32, 128, 512, 1024, 2048]
            # args.num_class_list = [2] #[2, 4, 8, 16, 20, 40, 60, 81]

        args = load_dataset(args)

        args = load_models(args)
        # if args.model == 'MLP2':
        #     args.net_a = MLP2(np.prod(list(args.gt_data_a.size())[1:]), args.num_classes).to(args.device)
        #     args.net_b = MLP2(np.prod(list(args.gt_data_b.size())[1:]), args.num_classes).to(args.device)
        # elif args.model == 'resnet18':
        #     args.net_a = resnet18(args.num_classes).to(args.device)
        #     args.net_b = resnet18(args.num_classes).to(args.device)
        for defense in args.defense_methods:
            # load attack configs
            print("use defense", defense)
            defense_index = args.defense_methods.index(defense)
            defense_config_file_path = args.defense_config_list[defense_index]
            args = load_defense_configs(defense_config_file_path, defense, args)
            print("everything loaded")

            args.exp_res_dir = f'exp_result/{attack}/{args.dataset}/{defense}/'
            if not os.path.exists(args.exp_res_dir):
                os.makedirs(args.exp_res_dir)
            filename = f'dataset={args.dataset},model={args.model_list[str(0)]["type"]},lr={args.lr},num_exp={args.num_exp},' \
                f'epochs={args.epochs},early_stop={args.early_stop}.txt'
            # filename = f'dataset={args.dataset},model={args.model},lr={args.lr},num_exp={args.num_exp},' \
            #        f'epochs={args.epochs},early_stop={args.early_stop}.txt'
            args.exp_res_path = args.exp_res_dir + filename


    # if args.dataset_name == "cifar100":
    #     half_dim = 16
    #     # num_classes = 20
    #     # num_classes = 2
    #     num_classes = 100
    #     train_dst = datasets.CIFAR100("./dataset/", download=True, train=True, transform=transform)
    #     data, label = fetch_data_and_label(train_dst, num_classes)
    #     train_dst = SimpleDataset(data, label)
    #     test_dst = datasets.CIFAR100("./dataset/", download=True, train=False, transform=transform)
    #     data, label = fetch_data_and_label(test_dst, num_classes)
    #     test_dst = SimpleDataset(data, label)
    # elif args.dataset_name == "cifar10":
    #     half_dim = 16
    #     num_classes = 10
    #     train_dst = datasets.CIFAR10("./dataset/", download=True, train=True, transform=transform)
    #     data, label = fetch_data_and_label(train_dst, num_classes)
    #     train_dst = SimpleDataset(data, label)
    #     test_dst = datasets.CIFAR10("./dataset/", download=True, train=False, transform=transform)
    #     data, label = fetch_data_and_label(test_dst, num_classes)
    #     test_dst = SimpleDataset(data, label)
    # elif args.dataset_name == "mnist":
    #     half_dim = 14
    #     num_classes = 10
    #     # num_classes = 2
    #     train_dst = datasets.MNIST("~/.torch", download=True, train=True, transform=transform_fn)
    #     data, label = fetch_data_and_label(train_dst, num_classes)
    #     train_dst = SimpleDataset(data, label)
    #     test_dst = datasets.MNIST("~/.torch", download=True, train=False, transform=transform_fn)
    #     data, label = fetch_data_and_label(test_dst, num_classes)
    #     test_dst = SimpleDataset(data, label)
    # elif args.dataset_name == 'nuswide':
    #     half_dim = [634, 1000]
    #     num_classes = 5
    #     # num_classes = 2
    #     train_dst = NUSWIDEDataset('./data/NUS_WIDE', 'train')
    #     test_dst = NUSWIDEDataset('./data/NUS_WIDE', 'test')
    # args.train_dataset = train_dst
    # args.val_dataset = test_dst
    # args.half_dim = half_dim
    # args.num_classes = num_classes

    # args.encoder = None
    # args.models_dict = {"mnist": MLP2,
    #            "cifar100": resnet18,
    #            "cifar10": resnet18,
    #         #    "cifar10": resnet20,
    #            "nuswide": MLP2,
    #            "classifier": None}

    # path = f'./exp_result/{args.dataset_name}/no_defense_main_task.txt'
    # num_exp = 10

    # test_acc_list = []
    # for _ in range(num_exp):
    #     vfl_defence_image = VFLDefenceExperimentBase(args)
    #     test_acc = vfl_defence_image.train()
    #     test_acc_list.append(test_acc)
    # append_exp_res(path, str(np.mean(test_acc_list))+ ' ' + str(test_acc_list) + ' ' + str(np.max(test_acc_list)))


            attacker = globals()[attack](args)
            attack_list.append(attacker)
            attacker.train()




