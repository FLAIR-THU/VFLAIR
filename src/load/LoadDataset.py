import sys, os
sys.path.append(os.pardir)

import random
import torch
from utils.basic_functions import get_class_i, get_labeled_data


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


def load_dataset(args):
    args.num_classes = args.num_class_list[0]
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
    else:
        assert split_type == 'horizontal_half', 'dataset splition type not supported yet'
    
    # important
    return args
