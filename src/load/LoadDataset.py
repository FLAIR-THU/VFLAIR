import sys, os
sys.path.append(os.pardir)

import random
import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
tp = transforms.ToTensor()
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
     ])
transform_fn = transforms.Compose([
    transforms.ToTensor()
])

from utils.basic_functions import get_class_i, get_labeled_data, fetch_data_and_label

def horizontal_half(args, all_dataset):
    if args.dataset == 'mnist' or args.dataset == 'cifar100' or args.dataset == 'cifar10':
        all_data, all_label = all_dataset
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
        x_image, x_text, Y = all_dataset
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
        assert args.dataset == 'nuswide', 'dataset not supported yet'
    return args

def HFL(args, all_dataset):
    if args.dataset == 'mnist' or args.dataset == 'cifar100' or args.dataset == 'cifar10':
        all_data, all_label = all_dataset
        gt_data = []
        gt_label = []
        for i in range(0, args.batch_size):
            sample_idx = torch.randint(len(all_data), size=(1,)).item()
            gt_data.append(all_data[sample_idx])
            gt_label.append(all_label[sample_idx])
        args.gt_data = torch.stack(gt_data).to(args.device)
        args.gt_label = torch.stack(gt_label).to(args.device)
        args.gt_onehot_label = gt_label  # label_to_onehot(gt_label)
    elif args.dataset == 'nuswide':
        x_image, x_text, Y = all_dataset
        gt_data_a, gt_data_b, gt_label = [], [], []
        for i in range(0, args.batch_size):
            sample_idx = torch.randint(len(x_image), size=(1,)).item()
            gt_data_a.append(torch.tensor(x_text[sample_idx], dtype=torch.float32))
            gt_data_b.append(torch.tensor(x_image[sample_idx], dtype=torch.float32))
            gt_label.append(torch.tensor(Y[sample_idx], dtype=torch.float32))
        # args.gt_data_a = torch.stack(gt_data_a).to(args.device)
        # args.gt_data_b = torch.stack(gt_data_b).to(args.device)
        args.gt_data = [torch.stack(gt_data_a).to(args.device), torch.stack(gt_data_b).to(args.device)]
        args.gt_label = torch.stack(gt_label).to(args.device)
        args.gt_onehot_label = gt_label  # label_to_onehot(gt_label)
    else:
        gt_data_a, gt_data_b, gt_label = [], [], []
        # args.gt_data_a = torch.stack(gt_data_a).to(args.device)
        # args.gt_data_b = torch.stack(gt_data_b).to(args.device)
        args.gt_data = torch.stack([]).to(args.device)
        args.gt_label = torch.stack(gt_label).to(args.device)
        args.gt_onehot_label = gt_label  # label_to_onehot(gt_label)
        assert args.dataset == 'nuswide', 'dataset not supported yet'
    return args


def load_dataset(args):
    args.num_classes = args.num_classes
    args.classes = [None] * args.num_classes
    all_dataset = []
    if args.dataset == 'cifar100':
        args.classes = random.sample(list(range(100)), args.num_classes)
        all_data, all_label = get_class_i(args.dst, args.classes)
        all_dataset = [all_data, all_label]
    elif args.dataset == 'cifar10':
        args.classes = random.sample(list(range(10)), args.num_classes)
        all_data, all_label = get_class_i(args.dst, args.classes)
        all_dataset = [all_data, all_label]
    elif args.dataset == 'mnist':
        args.classes = random.sample(list(range(10)), args.num_classes)
        all_data, all_label = get_class_i(args.dst, args.classes)
        all_dataset = [all_data, all_label]
    elif args.dataset == 'nuswide':
        all_nuswide_labels = []
        for line in os.listdir('./data/NUS_WIDE/Groundtruth/AllLabels'):
            all_nuswide_labels.append(line.split('_')[1][:-4])
        args.classes = random.sample(all_nuswide_labels, args.num_classes)
        x_image, x_text, Y = get_labeled_data('./data/NUS_WIDE', args.classes, None, 'Train')
        all_dataset = [x_image, x_text, Y]
    
    # randomly sample
    split_type = args.dataset_split['partition_function'] if ('partition_function' in args.dataset_split) else 'horizontal_half'
    if split_type == 'horizontal_half':
        args = horizontal_half(args, all_dataset)
    elif split_type == 'HFL':
        args = HFL(args, all_dataset)
    else:
        assert split_type == 'horizontal_half', 'dataset splition type not supported yet'
    
    # important
    return args


def load_dataset_per_party(args, index):
    args.num_classes = args.num_classes
    args.classes = [None] * args.num_classes

    half_dim = -1
    if args.dataset == "cifar100":
        half_dim = 16
        train_dst = datasets.CIFAR100("../../../share_dataset/", download=True, train=True, transform=transform)
        data, label = fetch_data_and_label(train_dst, args.num_classes)
        # train_dst = SimpleDataset(data, label)
        train_dst = (torch.tensor(data), label)
        test_dst = datasets.CIFAR100("../../../share_dataset/", download=True, train=False, transform=transform)
        data, label = fetch_data_and_label(test_dst, args.num_classes)
        # test_dst = SimpleDataset(data, label)
        test_dst = (torch.tensor(data), label)
    elif args.dataset == "cifar20":
        half_dim = 16
        train_dst = datasets.CIFAR100("../../../share_dataset/", download=True, train=True, transform=transform)
        data, label = fetch_data_and_label(train_dst, args.num_classes)
        # train_dst = SimpleDataset(data, label)
        train_dst = (torch.tensor(data), label)
        test_dst = datasets.CIFAR100("../../../share_dataset/", download=True, train=False, transform=transform)
        data, label = fetch_data_and_label(test_dst, args.num_classes)
        # test_dst = SimpleDataset(data, label)
        test_dst = (torch.tensor(data), label)
    elif args.dataset == "cifar10":
        half_dim = 16
        train_dst = datasets.CIFAR10("../../../share_dataset/", download=True, train=True, transform=transform)
        data, label = fetch_data_and_label(train_dst, args.num_classes)
        # train_dst = SimpleDataset(data, label)
        train_dst = (torch.tensor(data), label)
        test_dst = datasets.CIFAR10("../../../share_dataset/", download=True, train=False, transform=transform)
        data, label = fetch_data_and_label(test_dst, args.num_classes)
        # test_dst = SimpleDataset(data, label)
        test_dst = (torch.tensor(data), label)
    elif args.dataset == "mnist":
        half_dim = 14
        train_dst = datasets.MNIST("~/.torch", download=True, train=True, transform=transform_fn)
        data, label = fetch_data_and_label(train_dst, args.num_classes)
        # train_dst = SimpleDataset(data, label)
        print(type(data),type(data[0]),type(label))
        train_dst = (data, label)
        test_dst = datasets.MNIST("~/.torch", download=True, train=False, transform=transform_fn)
        data, label = fetch_data_and_label(test_dst, args.num_classes)
        # test_dst = SimpleDataset(data, label)
        test_dst = (data, label)
    # elif args.dataset_name == 'nuswide':
    #     half_dim = [634, 1000]
    #     train_dst = NUSWIDEDataset('../../../share_dataset/NUS_WIDE', 'train')
    #     test_dst = NUSWIDEDataset('../../../share_dataset/NUS_WIDE', 'test')
    # args.train_dataset = train_dst
    # args.val_dataset = test_dst
    else:
        assert args.dataset == 'mnist', "dataset not supported yet"
    
    train_dst = (train_dst[0].to(args.device),train_dst[1].to(args.device))
    test_dst = (test_dst[0].to(args.device),test_dst[1].to(args.device))

    if args.k == 2:
        if index == 0:
            # train_dst[0].shape = (samplecount,channels,height,width) for MNIST and CIFAR
            # passive party does not have label
            train_dst = (train_dst[0][:, :, :half_dim, :], None)
            test_dst = (test_dst[0][:, :, :half_dim, :], None)
        elif index == 1:
            train_dst = (train_dst[0][:, :, half_dim:, :], train_dst[1])
            test_dst = (test_dst[0][:, :, half_dim:, :], test_dst[1])
        else:
            assert index <= 1, "invalide party index"
    elif args.k == 4:
        if index == 3:
            train_dst = (train_dst[0][:, :, half_dim:, half_dim:], train_dst[1])
            test_dst = (test_dst[0][:, :, half_dim:, half_dim:], test_dst[1])         
        else:
            # passive party does not have label
            if index == 0:
                train_dst = (train_dst[0][:, :, :half_dim, :half_dim], None)
                test_dst = (test_dst[0][:, :, :half_dim, :half_dim], None)
            elif index == 1:
                train_dst = (train_dst[0][:, :, :half_dim, half_dim:], None)
                test_dst = (test_dst[0][:, :, :half_dim, half_dim:], None)
            elif index == 2:
                train_dst = (train_dst[0][:, :, half_dim:, :half_dim], None)
                test_dst = (test_dst[0][:, :, half_dim:, :half_dim], None)
            else:
                assert index <= 3, "invalide party index"
    else:
        assert (args.k == 2 or args.k == 4), "total number of parties not supported for data partitioning"
    
    # important
    return args, half_dim, train_dst, test_dst