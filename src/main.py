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

from models.vision import *
from utils.basic_functions import *
from utils.constants import *
from evaluates.BatchLabelReconstruction import *
from evaluates.SampleLabelReconstruction import *


def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def prepare_dataset(args):
    args.classes = [None] * args.num_classes
    if args.dataset == 'cifar100':
        args.classes = random.sample(list(range(100)), args.num_classes)
        all_data, all_label = get_class_i(args.dst, args.classes)
    elif args.dataset == 'mnist':
        args.classes = random.sample(list(range(10)), args.num_classes)
        all_data, all_label = get_class_i(args.dst, args.classes)
    elif args.dataset == 'nuswide':
        all_nuswide_labels = []
        for line in os.listdir('./data/NUS_WIDE/Groundtruth/AllLabels'):
            all_nuswide_labels.append(line.split('_')[1][:-4])
        args.classes = random.sample(all_nuswide_labels, args.num_classes)
        x_image, x_text, Y = get_labeled_data('./data/NUS_WIDE', args.classes, None, 'Train')
    elif args.dataset == 'cifar10':
        args.classes = random.sample(list(range(10)), args.num_classes)
        all_data, all_label = get_class_i(args.dst, args.classes)
    
    # randomly sample
    if args.dataset == 'mnist' or args.dataset == 'cifar100' or args.dataset == 'cifar10':
        gt_data = []
        gt_label = []
        for i in range(0, args.batch_size):
            sample_idx = torch.randint(len(all_data), size=(1,)).item()
            gt_data.append(all_data[sample_idx])
            gt_label.append(all_label[sample_idx])
        gt_data = torch.stack(gt_data).to(args.device)
        half_size = list(gt_data.size())[-1] // 2
        args.gt_data_a = gt_data[:, :, :half_size, :]
        args.gt_data_b = gt_data[:, :, half_size:, :]
        args.gt_label = torch.stack(gt_label).to(args.device)
        args.gt_onehot_label = gt_label  # label_to_onehot(gt_label)
    elif args.dataset == 'nuswide':
        gt_data_a, gt_data_b, gt_label = [], [], []
        for i in range(0, args.batch_size):
            sample_idx = torch.randint(len(x_image), size=(1,)).item()
            gt_data_a.append(torch.tensor(x_text[sample_idx], dtype=torch.float32))
            gt_data_b.append(torch.tensor(x_image[sample_idx], dtype=torch.float32))
            gt_label.append(torch.tensor(Y[sample_idx], dtype=torch.float32))
        args.gt_data_a = torch.stack(gt_data_a).to(args.device)
        args.gt_data_b = torch.stack(gt_data_b).to(args.device)
        args.gt_label = torch.stack(gt_label).to(args.device)
        args.gt_onehot_label = gt_label  # label_to_onehot(gt_label)
    else:
        gt_data_a, gt_data_b, gt_label = [], [], []
        args.gt_data_a = torch.stack(gt_data_a).to(args.device)
        args.gt_data_b = torch.stack(gt_data_b).to(args.device)
        args.gt_label = torch.stack(gt_label).to(args.device)
        args.gt_onehot_label = gt_label  # label_to_onehot(gt_label)

    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser("backdoor")
    parser.add_argument('--device', type=str, default='cuda', help='use gpu or cpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--seed', type=int, default=97, help='random seed')
    # ####### add to config file #######
    # parser.add_argument('--defense_up', type=int, default=0)
    # parser.add_argument('--epochs', type=int, default=20, help='num of training epochs')
    # parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    # parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
    # parser.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate')
    # parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
    # parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
    # parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
    # parser.add_argument('--k', type=int, default=3, help='num of client')
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

    args.model = 'MLP2'
    args.dataset = 'mnist'
    args.num_classes = 10
    args.batch_size = 64
    args.epochs = 4000
    args.lr = 0.01
    args.num_exp = 10
    args.early_stop = False
    args.early_stop_param = 0.0001

    args.apply_trainable_layer = 0
    args.apply_laplace = 0
    args.apply_gaussian = 0
    args.dp_strength = 0.0
    args.apply_grad_spar = 0
    args.grad_spars = 0.0
    args.apply_encoder = 0
    args.apply_adversarial_encoder = 0
    args.ae_lambda = 0.1
    args.encoder = None
    args.apply_marvell = 0
    args.marvell_s = 0


    if args.dataset == 'cifar100':
        # args.dst = datasets.CIFAR100("./dataset/", download=True)
        args.dst = datasets.CIFAR10("./dataset/", download=True)
        args.batch_size_list = [2048] #[2, 32, 128, 512, 2048]
        args.num_class_list = [2] #[5, 10, 15, 20, 40, 60, 80, 100]
    elif args.dataset == 'mnist':
        args.dst = datasets.MNIST("~/.torch", download=True)
        args.batch_size_list = [32, 128, 512, 1024, 2048]
        args.num_class_list = [2] #[2, 3, 4, 5, 6, 7, 8, 9, 10]
    elif args.dataset == 'nuswide':
        args.dst = None
        args.batch_size_list = [32, 128, 512, 1024, 2048]
        args.num_class_list = [2] #[2, 4, 8, 16, 20, 40, 60, 81]

    args = prepare_dataset(args)

    if args.model == 'MLP2':
        args.net_a = MLP2(np.prod(list(args.gt_data_a.size())[1:]), args.num_classes).to(args.device)
        args.net_b = MLP2(np.prod(list(args.gt_data_b.size())[1:]), args.num_classes).to(args.device)
    elif args.model == 'resnet18':
        args.net_a = resnet18(args.num_classes).to(args.device)
        args.net_b = resnet18(args.num_classes).to(args.device)


    args.exp_res_dir = f'exp_result/{args.dataset}/'
    if not os.path.exists(args.exp_res_dir):
        os.makedirs(args.exp_res_dir)
    filename = f'dataset={args.dataset},model={args.model},lr={args.lr},num_exp={args.num_exp},' \
           f'epochs={args.epochs},early_stop={args.early_stop}.txt'
    args.exp_res_path = args.exp_res_dir + filename
    
    attack_list = [BatchLabelReconstruction(args)]
    for attack in attack_list:
        attacker = attack
        attacker.train()




