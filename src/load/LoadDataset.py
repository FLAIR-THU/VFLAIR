import sys, os
sys.path.append(os.pardir)

import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score,accuracy_score,recall_score,f1_score,precision_score,roc_curve,auc,average_precision_score,log_loss
from copy import deepcopy, copy

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

from utils.basic_functions import get_class_i, get_labeled_data, fetch_data_and_label, generate_poison_data
from utils.cora_utils import *
from utils.graph_functions import load_data1, split_graph

TABULAR_DATA = ['breast_cancer_diagnose','diabetes','adult_income']
GRAPH_DATA = ['cora']

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


def dataset_partition(args, index, dst, half_dim):
    if args.dataset in ['mnist', 'cifar10', 'cifar100', 'cifar20']:
        if args.k == 2:
            if index == 0:
                # return (dst[0][:, :, :half_dim, :], None)
                return (dst[0][:, :, half_dim:, :], None)
            elif index == 1:
                # return (dst[0][:, :, half_dim:, :], dst[1])
                return (dst[0][:, :, :half_dim, :], dst[1])
            else:
                assert index <= 1, "invalide party index"
                return None
        elif args.k == 4:
            if index == 3:
                return (dst[0][:, :, half_dim:, half_dim:], dst[1])
            else:
                # passive party does not have label
                if index == 0:
                    return (dst[0][:, :, :half_dim, :half_dim], None)
                elif index == 1:
                    return (dst[0][:, :, :half_dim, half_dim:], None)
                elif index == 2:
                    return (dst[0][:, :, half_dim:, :half_dim], None)
                else:
                    assert index <= 3, "invalide party index"
                    return None
        else:
            assert (args.k == 2 or args.k == 4), "total number of parties not supported for data partitioning"
            return None
    elif args.dataset in TABULAR_DATA:
        if args.k == 2:
            if index == 0:
                return (dst[0][:, :half_dim], None)
            elif index == 1:
                return (dst[0][:, half_dim:], dst[1])
            else:
                assert index <= 1, "invalide party index"
                return None
        elif args.k == 4:
            half_dim = int(half_dim//2)
            if index == 3:
                return (dst[0][:, half_dim*3:], dst[1])
            else:
                # passive party does not have label
                if index <= 2:
                    return (dst[0][:, half_dim*index:half_dim*(index+1)], None)
                else:
                    assert index <= 3, "invalide party index"
                    return None
        else:
            assert (args.k == 2 or args.k == 4), "total number of parties not supported for data partitioning"
            return None
    elif args.dataset in GRAPH_DATA: #args.dataset == 'cora':
        assert args.k == 2, 'more than 2 party is not supported for cora'
        if index == 0:
            A_A, A_B, X_A, X_B = split_graph(args, dst[0][0], dst[0][1], split_method='com', split_ratio=0.5, with_s=True, with_f=True)
            A_A = normalize_adj(A_A)
            A_B = normalize_adj(A_B)
            # print(type(A_A),type(A_B),type(X_A),type(X_B))
            A_A = sparse_mx_to_torch_sparse_tensor(A_A).to(args.device)
            args.A_B = sparse_mx_to_torch_sparse_tensor(A_B).to(args.device)
            X_A = sparse_mx_to_torch_sparse_tensor(X_A).to(args.device)
            args.X_B = sparse_mx_to_torch_sparse_tensor(X_B).to(args.device)
            args.half_dim = [X_A.shape[1], X_B.shape[1]]
            # print(args.half_dim)
            return ([A_A,X_A],None), args
        elif index == 1:
            return ([args.A_B,args.X_B],dst[1]), args
        else:
            assert index <= 1, 'invalid party index'
    else:
        assert args.dataset == 'mnist', "dataset not supported"
        return None

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
        train_dst = (data, label)
        test_dst = datasets.MNIST("~/.torch", download=True, train=False, transform=transform_fn)
        data, label = fetch_data_and_label(test_dst, args.num_classes)
        # test_dst = SimpleDataset(data, label)
        test_dst = (data, label)
    elif args.dataset in GRAPH_DATA:
        if args.dataset == 'cora':
            adj, features, idx_train, idx_val, idx_test, label = load_data1(args.dataset)
            target_nodes = idx_test
            A = np.array(adj.todense())
            X = sparse_to_tuple(features.tocoo())
            # print(type(adj), type(features), type(label))
            idx_train = torch.LongTensor(idx_train)
            idx_test = torch.LongTensor(idx_test)
            label = torch.LongTensor(label).to(args.device)
            train_dst = ([adj, features], label)
            test_dst = ([adj, features,target_nodes], label)
            # test_dst = (torch.tensor([adj.to(args.device), features.to(args.device), target_nodes]),label)
            # adj = normalize_adj(adj)
            # adj = sparse_mx_to_torch_sparse_tensor(adj)
            # features = sparse_mx_to_torch_sparse_tensor(features)
            # idx_train = torch.LongTensor(idx_train)
            # idx_val = torch.LongTensor(idx_val)
            # idx_test = torch.LongTensor(idx_test)
            # labels = torch.LongTensor(labels)
        half_dim = -1
    elif args.dataset in TABULAR_DATA:
        if args.dataset == 'breast_cancer_diagnose':
            half_dim = 15
            df = pd.read_csv("../../../share_dataset/BreastCancer/wdbc.data",header = 0)
            X = df.iloc[:, 2:].values
            y = df.iloc[:, 1].values
            y = np.where(y=='B',0,1)
            y = np.squeeze(y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
        elif args.dataset == 'diabetes':
            half_dim = 4
            df = pd.read_csv("../../../share_dataset/Diabetes/diabetes.csv",header = 0)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
        elif args.dataset == 'adult_income':
            df = pd.read_csv("../../../share_dataset/Income/adult.csv",header = 0)
            df = df.drop_duplicates()
            encoder = OneHotEncoder()
            # 'age', 'workclass', 'fnlwgt', 'education', 'educational-num',
            # 'marital-status', 'occupation', 'relationship', 'race', 'gender',
            # 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
            # 'income'
            # category_columns_index = [1,3,5,6,7,8,9,13]
            # num_category_of_each_column = [9,16,7,15,6,5,2,42]
            category_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender','native-country']
            # encoder.fit(df.loc[:,category_columns])
            # df.loc[:,category_columns] = encoder.transform(df.loc[:,category_columns])
            for _column in category_columns:
                # Get one hot encoding of columns B
                one_hot = pd.get_dummies(df[_column], prefix=_column)
                # Drop column B as it is now encoded
                df = df.drop(_column,axis = 1)
                # Join the encoded df
                df = df.join(one_hot)
            y = df['income'].values
            y = np.where(y=='<=50K',0,1)
            df = df.drop('income',axis=1)
            X = df.values
            half_dim = 6+9 #=15 acc=0.83
            # half_dim = 6+9+16+7+15 #=53 acc=0.77
            # half_dim = int(X.shape[1]//2) acc=0.77
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
        X_train = torch.tensor(X_train)
        X_test = torch.tensor(X_test)
        y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)
        train_dst = (X_train,y_train)
        test_dst = (X_test,y_test)
    # elif args.dataset_name == 'nuswide':
    #     half_dim = [634, 1000]
    #     train_dst = NUSWIDEDataset('../../../share_dataset/NUS_WIDE', 'train')
    #     test_dst = NUSWIDEDataset('../../../share_dataset/NUS_WIDE', 'test')
    # args.train_dataset = train_dst
    # args.val_dataset = test_dst
    else:
        assert args.dataset == 'mnist', "dataset not supported yet"
    
    if not args.dataset in GRAPH_DATA:
        train_dst = (train_dst[0].to(args.device),train_dst[1].to(args.device))
        test_dst = (test_dst[0].to(args.device),test_dst[1].to(args.device))
        train_dst = dataset_partition(args,index,train_dst,half_dim)
        test_dst = dataset_partition(args,index,test_dst,half_dim)
    else:
        train_dst, args = dataset_partition(args,index,train_dst,half_dim)
        test_dst = ([deepcopy(train_dst[0][0]),deepcopy(train_dst[0][1]),test_dst[0][2]],test_dst[1])
    
    # important
    return args, half_dim, train_dst, test_dst

def prepare_poison_target_list(args):
    args.target_label = random.randint(0, args.num_classes-1)

def load_dataset_per_party_backdoor(args, index):
    args.num_classes = args.num_classes
    args.classes = [None] * args.num_classes

    half_dim = -1
    if args.dataset == "cifar100":
        half_dim = 16
        train_dst = datasets.CIFAR100("../../../share_dataset/", download=True, train=True, transform=transform)
        train_data, train_label = fetch_data_and_label(train_dst, args.num_classes)
        test_dst = datasets.CIFAR100("../../../share_dataset/", download=True, train=False, transform=transform)
        test_data, test_label = fetch_data_and_label(test_dst, args.num_classes)
        if args.target_label == None:
            args.target_label = random.randint(0, args.num_classes-1)
            args.train_poison_list = random.sample(range(len(train_dst)), int(0.01 * len(train_dst)))
            args.test_poison_list = random.sample(range(len(test_dst)), int(0.01 * len(test_dst)))
        else:
            assert args.train_poison_list != None , "[[inner error]]"
            assert args.train_target_list != None, "[[inner error]]"
            assert args.test_poison_list != None, "[[inner error]]"
            assert args.test_target_list != None, "[[inner error]]"
        train_data, train_label, train_poison_data, train_poison_label = generate_poison_data(train_data, train_label, args.train_poison_list, 'train', args.k, args.dataset)
        test_data, test_label, test_poison_data, test_poison_label = generate_poison_data(test_data, test_label, args.test_poison_list, 'test', args.k, args.dataset)
        if args.train_target_list == None:
            assert args.test_target_list == None
            args.train_target_list = random.sample(list(np.where(torch.argmax(train_label,axis=1)==args.target_label)[0]), 10)
            args.test_target_list = random.sample(list(np.where(torch.argmax(test_label,axis=1)==args.target_label)[0]), 10)
    elif args.dataset == "cifar20":
        half_dim = 16
        train_dst = datasets.CIFAR100("../../../share_dataset/", download=True, train=True, transform=transform)
        train_data, train_label = fetch_data_and_label(train_dst, args.num_classes)
        test_dst = datasets.CIFAR100("../../../share_dataset/", download=True, train=False, transform=transform)
        test_data, test_label = fetch_data_and_label(test_dst, args.num_classes)
        if args.target_label == None:
            args.target_label = random.randint(0, args.num_classes-1)
            args.train_poison_list = random.sample(range(len(train_dst)), int(0.01 * len(train_dst)))
            args.test_poison_list = random.sample(range(len(test_dst)), int(0.01 * len(test_dst)))
        else:
            assert args.train_poison_list != None , "[[inner error]]"
            assert args.train_target_list != None, "[[inner error]]"
            assert args.test_poison_list != None, "[[inner error]]"
            assert args.test_target_list != None, "[[inner error]]"
        train_data, train_label, train_poison_data, train_poison_label = generate_poison_data(train_data, train_label, args.train_poison_list, 'train', args.k, args.dataset)
        test_data, test_label, test_poison_data, test_poison_label = generate_poison_data(test_data, test_label, args.test_poison_list, 'test', args.k, args.dataset)
        if args.train_target_list == None:
            assert args.test_target_list == None
            args.train_target_list = random.sample(list(np.where(torch.argmax(train_label,axis=1)==args.target_label)[0]), 10)
            args.test_target_list = random.sample(list(np.where(torch.argmax(test_label,axis=1)==args.target_label)[0]), 10)
    elif args.dataset == "cifar10":
        half_dim = 16
        train_dst = datasets.CIFAR10("../../../share_dataset/", download=True, train=True, transform=transform)
        train_data, train_label = fetch_data_and_label(train_dst, args.num_classes)
        test_dst = datasets.CIFAR10("../../../share_dataset/", download=True, train=False, transform=transform)
        test_data, test_label = fetch_data_and_label(test_dst, args.num_classes)
        if args.target_label == None:
            args.target_label = random.randint(0, args.num_classes-1)
            args.train_poison_list = random.sample(range(len(train_dst)), int(0.01 * len(train_dst)))
            args.test_poison_list = random.sample(range(len(test_dst)), int(0.01 * len(test_dst)))
        else:
            assert args.train_poison_list != None , "[[inner error]]"
            assert args.train_target_list != None, "[[inner error]]"
            assert args.test_poison_list != None, "[[inner error]]"
            assert args.test_target_list != None, "[[inner error]]"
        train_data, train_label, train_poison_data, train_poison_label = generate_poison_data(train_data, train_label, args.train_poison_list, 'train', args.k, args.dataset)
        test_data, test_label, test_poison_data, test_poison_label = generate_poison_data(test_data, test_label, args.test_poison_list, 'test', args.k, args.dataset)
        if args.train_target_list == None:
            assert args.test_target_list == None
            args.train_target_list = random.sample(list(np.where(torch.argmax(train_label,axis=1)==args.target_label)[0]), 10)
            args.test_target_list = random.sample(list(np.where(torch.argmax(test_label,axis=1)==args.target_label)[0]), 10)
    elif args.dataset == "mnist":
        half_dim = 14
        train_dst = datasets.MNIST("~/.torch", download=True, train=True, transform=transform_fn)
        train_data, train_label = fetch_data_and_label(train_dst, args.num_classes)
        test_dst = datasets.MNIST("~/.torch", download=True, train=False, transform=transform_fn)
        test_data, test_label = fetch_data_and_label(test_dst, args.num_classes)
        if args.target_label == None:
            args.target_label = random.randint(0, args.num_classes-1)
            args.train_poison_list = random.sample(range(len(train_dst)), int(0.01 * len(train_dst)))
            args.test_poison_list = random.sample(range(len(test_dst)), int(0.01 * len(test_dst)))
        else:
            assert args.train_poison_list != None , "[[inner error]]"
            assert args.train_target_list != None, "[[inner error]]"
            assert args.test_poison_list != None, "[[inner error]]"
            assert args.test_target_list != None, "[[inner error]]"
        train_data, train_label, train_poison_data, train_poison_label = generate_poison_data(train_data, train_label, args.train_poison_list, 'train', args.k, args.dataset)
        test_data, test_label, test_poison_data, test_poison_label = generate_poison_data(test_data, test_label, args.test_poison_list, 'test', args.k, args.dataset)
        if args.train_target_list == None:
            assert args.test_target_list == None
            args.train_target_list = random.sample(list(np.where(torch.argmax(train_label,axis=1)==args.target_label)[0]), 10)
            args.test_target_list = random.sample(list(np.where(torch.argmax(test_label,axis=1)==args.target_label)[0]), 10)
    # elif args.dataset_name == 'nuswide':
    #     half_dim = [634, 1000]
    #     train_dst = NUSWIDEDataset('../../../share_dataset/NUS_WIDE', 'train')
    #     test_dst = NUSWIDEDataset('../../../share_dataset/NUS_WIDE', 'test')
    # args.train_dataset = train_dst
    # args.val_dataset = test_dst
    else:
        assert args.dataset == 'mnist', "dataset not supported yet"

    train_dst = (torch.tensor(train_data).to(args.device),train_label.to(args.device))
    test_dst = (torch.tensor(test_data).to(args.device),test_label.to(args.device))
    train_poison_dst = (train_poison_data.to(args.device),train_poison_label.to(args.device))
    test_poison_dst = (test_poison_data.to(args.device),test_poison_label.to(args.device))

    train_dst = dataset_partition(args,index,train_dst,half_dim)
    test_dst = dataset_partition(args,index,test_dst,half_dim)
    train_poison_dst = dataset_partition(args,index,train_poison_dst,half_dim)
    test_poison_dst = dataset_partition(args,index,test_poison_dst,half_dim)
    # if args.k == 2:
    #     if index == 0:
    #         # train_dst[0].shape = (samplecount,channels,height,width) for MNIST and CIFAR
    #         # passive party does not have label
    #         train_dst = (train_dst[0][:, :, :half_dim, :], None)
    #         test_dst = (test_dst[0][:, :, :half_dim, :], None)
    #         train_poison_dst = (train_poison_dst[0][:, :, :half_dim, :], None)
    #         test_poison_dst = (test_poison_dst[0][:, :, :half_dim, :], None)
    #     elif index == 1:
    #         train_dst = (train_dst[0][:, :, half_dim:, :], train_dst[1])
    #         test_dst = (test_dst[0][:, :, half_dim:, :], test_dst[1])
    #         train_poison_dst = (train_poison_dst[0][:, :, half_dim:, :], train_poison_dst[1])
    #         test_poison_dst = (test_poison_dst[0][:, :, half_dim:, :], test_poison_dst[1])
    #     else:
    #         assert index <= 1, "invalide party index"
    # elif args.k == 4:
    #     if index == 3:
    #         train_dst = (train_dst[0][:, :, half_dim:, half_dim:], train_dst[1])
    #         test_dst = (test_dst[0][:, :, half_dim:, half_dim:], test_dst[1])
    #         train_poison_dst = (train_poison_dst[0][:, :, half_dim:, half_dim:], train_poison_dst[1])
    #         test_poison_dst = (test_poison_dst[0][:, :, half_dim:, half_dim:], test_poison_dst[1])         
    #     else:
    #         # passive party does not have label
    #         if index == 0:
    #             train_dst = (train_dst[0][:, :, :half_dim, :half_dim], None)
    #             test_dst = (test_dst[0][:, :, :half_dim, :half_dim], None)
    #             train_poison_dst = (train_poison_dst[0][:, :, :half_dim, :half_dim], None)
    #             test_poison_dst = (test_poison_dst[0][:, :, :half_dim, :half_dim], None)
    #         elif index == 1:
    #             train_dst = (train_dst[0][:, :, :half_dim, half_dim:], None)
    #             test_dst = (test_dst[0][:, :, :half_dim, half_dim:], None)
    #             train_poison_dst = (train_poison_dst[0][:, :, :half_dim, half_dim:], None)
    #             test_poison_dst = (test_poison_dst[0][:, :, :half_dim, half_dim:], None)
    #         elif index == 2:
    #             train_dst = (train_dst[0][:, :, half_dim:, :half_dim], None)
    #             test_dst = (test_dst[0][:, :, half_dim:, :half_dim], None)
    #             train_poison_dst = (train_poison_dst[0][:, :, half_dim:, :half_dim], None)
    #             test_poison_dst = (test_poison_dst[0][:, :, half_dim:, :half_dim], None)
    #         else:
    #             assert index <= 3, "invalide party index"
    # else:
    #     assert (args.k == 2 or args.k == 4), "total number of parties not supported for data partitioning"
    
    # important
    return args, half_dim, train_dst, test_dst, train_poison_dst, test_poison_dst, args.train_target_list, args.test_target_list

