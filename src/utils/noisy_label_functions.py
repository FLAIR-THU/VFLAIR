import numpy as np
import torch
import random


def label_to_one_hot(target, num_classes=10):
    # print('label_to_one_hot:', target, type(target))
    try:
        _ = target.size()[1]
        onehot_target = target.type(torch.float32)
    except:
        target = torch.unsqueeze(target, 1)
        # print("use unsqueezed target", target.size())
        onehot_target = torch.zeros(target.size(0), num_classes)
        onehot_target.scatter_(1, target, 1)
    return onehot_target


def old_add_noise(args, true_label):
    assert args.num_classes == 10 or args.num_classes == 2, 'Noisy label only availabel for 10/2 classification'
    assert ('noise_rate' in args.attack_configs), 'noise rate not specified'
    assert ('noise_type' in args.attack_configs), 'noise type not specified'
    r = args.attack_configs['noise_rate']
    noise_type = args.attack_configs['noise_type']
    assert (
                noise_type == "asymmetric" or noise_type == "symmetric"), 'invalid noise_type, it should be symmetric/asymmetric'

    if len(true_label.size()) > 1:
        true_label = torch.argmax(true_label, dim=-1)

    if noise_type == "asymmetric":
        if args.num_classes == 10:
            for i in range(len(true_label)):
                if np.random.random() < r:
                    # print('Add Noise')
                    if true_label[i] == 9:
                        true_label[i] = 1
                    elif true_label[i] == 2:
                        true_label[i] = 0
                    elif true_label[i] == 4:
                        true_label[i] = 7
                    elif true_label[i] == 3:
                        true_label[i] = 5
                    elif true_label[i] == 5:
                        true_label[i] = 3
        elif args.num_classes == 2:
            for i in range(len(true_label)):
                if np.random.random() < r:
                    # print('Add Noise')
                    if true_label[i] == 0:
                        true_label[i] = 1
                    elif true_label[i] == 1:
                        true_label[i] = 0
    elif noise_type == "symmetric":
        for i in range(len(true_label)):
            if np.random.random() < r:
                # print('Add Noise')
                true_label[i] = np.random.randint(0, args.num_classes)

    noisy_label = label_to_one_hot(true_label, args.num_classes)
    # print(true_label.size())
    return noisy_label


def add_noise(args, true_label):
    assert args.num_classes == 10 or args.num_classes == 2, 'Noisy label only availabel for 10/2 classification'
    assert ('noise_rate' in args.attack_configs), 'noise rate not specified'
    assert ('noise_type' in args.attack_configs), 'noise type not specified'
    num_classes = args.num_classes
    r = args.attack_configs['noise_rate']
    noise_type = args.attack_configs['noise_type']

    if len(true_label.size()) > 1:
        true_label = torch.argmax(true_label, dim=-1)

    if noise_type == "asymmetric":
        for i in range(len(true_label)):
            if np.random.random() < r:
                label_lst = list(range(num_classes))
                label_lst.remove(true_label[i])
                true_label[i] == random.sample(label_lst, k=1)[0]
    elif noise_type == "pairflip":
        for i in range(len(true_label)):
            if np.random.random() < r:
                # print('Add Noise')
                true_label[i] = (true_label[i] - 1) % num_classes  # np.random.randint(0,args.num_classes)

    # noisy_label = label_to_one_hot(true_label, args.num_classes)
    noisy_label = label_to_one_hot(true_label.cpu(), args.num_classes).to(args.device)
    # print(true_label.size())
    return noisy_label
