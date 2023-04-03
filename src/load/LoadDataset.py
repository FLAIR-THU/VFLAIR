import sys, os
from os.path import join
sys.path.append(os.pardir)

import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score,accuracy_score,recall_score,f1_score,precision_score,roc_curve,auc,average_precision_score,log_loss
from copy import deepcopy, copy

import re
from sklearn.feature_extraction.text import TfidfVectorizer

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

DATA_PATH ='./load/share_dataset/'  #'../../../share_dataset/'
TABULAR_DATA = ['breast_cancer_diagnose','diabetes','adult_income','criteo']
GRAPH_DATA = ['cora']
TEXT_DATA = ['news20']


def dataset_partition(args, index, dst, half_dim):
    if args.dataset in ['mnist', 'cifar10', 'cifar100', 'cifar20']:
        if args.k == 2:
            if index == 0:
                return (dst[0][:, :, :half_dim, :], None)
                # return (dst[0][:, :, half_dim:, :], None)
            elif index == 1:
                return (dst[0][:, :, half_dim:, :], dst[1])
                # return (dst[0][:, :, :half_dim, :], dst[1])
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
        elif args.k == 1: # Centralized Training
            return (dst[0], dst[1])
        else:
            assert (args.k == 2 or args.k == 4), "total number of parties not supported for data partitioning"
            return None
    elif args.dataset in ['nuswide']:
        if args.k == 2:
            if index == 0:
                return (dst[0][1],None) # passive party with text
            else:
                return (dst[0][0], dst[1]) # active party with image
        else:
            assert (args.k == 2), "total number of parties not supported for data partitioning"
            return None
    elif args.dataset in TABULAR_DATA:
        dim_list=[]
        for ik in range(args.k):
            dim_list.append(int(args.model_list[str(ik)]['input_dim']))
            if len(dim_list)>1:
                dim_list[-1]=dim_list[-1]+dim_list[-2]
        dim_list.insert(0,0)

        if args.k == 1: # Centralized Training
            return (dst[0], dst[1])

        if index == (args.k-1):
            return (dst[0][:, dim_list[index]:], dst[1])
        else:
            # passive party does not have label
            if index <= (args.k-1):  
                return (dst[0][:, dim_list[index]:dim_list[index+1]], None)
            else:
                assert index <= (args.k-1), "invalide party index"
                return None
    elif args.dataset in TEXT_DATA: 
        dim_list=[]
        for ik in range(args.k):
            dim_list.append(int(args.model_list[str(ik)]['input_dim']))
            if len(dim_list)>1:
                dim_list[-1]=dim_list[-1]+dim_list[-2]
        dim_list.insert(0,0)
        
        if args.k == 1:
            return (dst[0], dst[1])

        if index == (args.k-1):
            return (dst[0][:, dim_list[index]:], dst[1])
        else:
            # passive party does not have label
            if index <= (args.k-1):  
                return (dst[0][:, dim_list[index]:dim_list[index+1]], None)
            else:
                assert index <= (args.k-1), "invalide party index"
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
            print("cora after split", A_A.shape, A_B.shape, X_A.shape, X_B.shape)
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
    args.idx_train = None
    args.idx_test = None
    if args.dataset == "cifar100":
        half_dim = 16
        train_dst = datasets.CIFAR100(DATA_PATH, download=True, train=True, transform=transform_fn)
        data, label = fetch_data_and_label(train_dst, args.num_classes)
        # train_dst = SimpleDataset(data, label)
        train_dst = (torch.tensor(data), label)
        test_dst = datasets.CIFAR100(DATA_PATH, download=True, train=False, transform=transform_fn)
        data, label = fetch_data_and_label(test_dst, args.num_classes)
        # test_dst = SimpleDataset(data, label)
        test_dst = (torch.tensor(data), label)
    elif args.dataset == "cifar20":
        half_dim = 16
        train_dst = datasets.CIFAR100(DATA_PATH, download=True, train=True, transform=transform_fn)
        data, label = fetch_data_and_label(train_dst, args.num_classes)
        # train_dst = SimpleDataset(data, label)
        train_dst = (torch.tensor(data), label)
        test_dst = datasets.CIFAR100(DATA_PATH, download=True, train=False, transform=transform_fn)
        data, label = fetch_data_and_label(test_dst, args.num_classes)
        # test_dst = SimpleDataset(data, label)
        test_dst = (torch.tensor(data), label)
    elif args.dataset == "cifar10":
        half_dim = 16
        train_dst = datasets.CIFAR10(DATA_PATH, download=True, train=True, transform=transform_fn)
        data, label = fetch_data_and_label(train_dst, args.num_classes)
        # train_dst = SimpleDataset(data, label)
        train_dst = (torch.tensor(data), label)
        test_dst = datasets.CIFAR10(DATA_PATH, download=True, train=False, transform=transform_fn)
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
    elif args.dataset == 'nuswide':
        half_dim = [634, 1000]
        if args.num_classes == 5:
            selected_labels = ['buildings', 'grass', 'animal', 'water', 'person'] # class_num = 5
        elif args.num_classes == 2:
            selected_labels = ['clouds','person'] # class_num = 2
        X_image, X_text, Y = get_labeled_data(DATA_PATH+'NUS_WIDE', selected_labels, 60000, 'Train')
        # X_image, X_text, Y = get_labeled_data(DATA_PATH+'NUS_WIDE', selected_labels, 600, 'Train')
        data = [torch.tensor(X_image, dtype=torch.float32), torch.tensor(X_text, dtype=torch.float32)]
        label = torch.squeeze(torch.tensor(np.argmax(np.array(Y), axis=1), dtype=torch.long))
        train_dst = (data, label)
        X_image, X_text, Y = get_labeled_data(DATA_PATH+'NUS_WIDE', selected_labels, 40000, 'Test')
        # X_image, X_text, Y = get_labeled_data(DATA_PATH+'NUS_WIDE', selected_labels, 400, 'Test')
        data = [torch.tensor(X_image, dtype=torch.float32), torch.tensor(X_text, dtype=torch.float32)]
        label = torch.squeeze(torch.tensor(np.argmax(np.array(Y), axis=1), dtype=torch.long))
        test_dst = (data, label)
        print("nuswide dataset [test]:", data[0].shape, data[1].shape, label.shape)
    elif args.dataset in GRAPH_DATA:
        if args.dataset == 'cora':
            adj, features, idx_train, idx_val, idx_test, label = load_data1(args.dataset)
            target_nodes = idx_test
            A = np.array(adj.todense())
            X = sparse_to_tuple(features.tocoo())
            print("cora dataset before split", A.shape, type(X), X[0].shape)
            args.idx_train = torch.LongTensor(idx_train)
            args.idx_test = torch.LongTensor(idx_test)
            label = torch.LongTensor(label).to(args.device)
            train_dst = ([adj, features], label)
            test_dst = ([adj, features, target_nodes], label)
        half_dim = -1
    elif args.dataset in TABULAR_DATA:
        if args.dataset == 'breast_cancer_diagnose':
            half_dim = 15
            df = pd.read_csv(DATA_PATH+"BreastCancer/wdbc.data",header = 0)
            X = df.iloc[:, 2:].values
            y = df.iloc[:, 1].values
            y = np.where(y=='B',0,1)
            y = np.squeeze(y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
        elif args.dataset == 'diabetes':
            half_dim = 4
            df = pd.read_csv(DATA_PATH+"Diabetes/diabetes.csv",header = 0)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
        elif args.dataset == 'adult_income':
            df = pd.read_csv(DATA_PATH+"Income/adult.csv",header = 0)
            df = df.drop_duplicates()
            # 'age', 'workclass', 'fnlwgt', 'education', 'educational-num',
            # 'marital-status', 'occupation', 'relationship', 'race', 'gender',
            # 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
            # 'income'
            # category_columns_index = [1,3,5,6,7,8,9,13]
            # num_category_of_each_column = [9,16,7,15,6,5,2,42]
            category_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender','native-country']
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
        elif args.dataset == 'criteo':
            df = pd.read_csv(DATA_PATH+"Criteo/criteo.csv",nrows=100000)
            print("criteo dataset loaded")
            half_dim = (df.shape[1]-1)//2
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)
        X_train = torch.tensor(X_train)
        X_test = torch.tensor(X_test)
        y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)
        train_dst = (X_train,y_train)
        test_dst = (X_test,y_test)
    elif args.dataset in TEXT_DATA:
        if args.dataset == 'news20':
            texts, labels, labels_index = [], {}, []
            Text_dir = DATA_PATH+'news20/'
            for name in sorted(os.listdir(Text_dir)):
                #  every file_folder under the root_file_folder should be labels with a unique number
                labels[name] = len(labels) # 
                path = join(Text_dir, name)
                for fname in sorted(os.listdir(path)):
                    if fname.isdigit():# The training set we want is all have a digit name
                        fpath = join(path,fname)
                        labels_index.append(labels[name])
                        # skip header
                        f = open(fpath, encoding='latin-1')
                        t = f.read()
                        texts.append(t)
                        f.close()
            #MAX_SEQUENCE_LENGTH = 1000
            #MAX_NB_WORDS = 20000
            #tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
            #tokenizer.fit_on_texts(texts)
            #sequences = tokenizer.texts_to_sequences(texts)
            # word_index = tokenizer.word_index
            # vocab_size = len(word_index) + 1
            #half_dim = int(MAX_SEQUENCE_LENGTH/2) # 500
            #X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
            vectorizer = TfidfVectorizer() 
            X = vectorizer.fit_transform(texts)
            X = np.array(X.A)
            y = np.array(labels_index)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            # ADDED: in config: input_dim = X.shape[1]//2 need to change according to categories included
            half_dim = int(X.shape[1]//2) #42491
        X_train = torch.tensor(X_train)
        X_test = torch.tensor(X_test)
        y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)
        train_dst = (X_train,y_train)
        test_dst = (X_test,y_test)
    else:
        assert args.dataset == 'mnist', "dataset not supported yet"
    
    if not args.dataset in GRAPH_DATA:
        if not args.dataset == 'nuswide':
            train_dst = (train_dst[0].to(args.device),train_dst[1].to(args.device))
            test_dst = (test_dst[0].to(args.device),test_dst[1].to(args.device))
        else:
            train_dst = ([train_dst[0][0].to(args.device),train_dst[0][1].to(args.device)],train_dst[1].to(args.device))
            test_dst = ([test_dst[0][0].to(args.device),test_dst[0][1].to(args.device)],test_dst[1].to(args.device))
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
    args.idx_train = None
    args.idx_test = None
    if args.dataset in ['mnist', 'cifar100', 'cifar20', 'cifar10']:
        # load image datasets
        if args.dataset == "cifar100":
            half_dim = 16
            train_dst = datasets.CIFAR100(DATA_PATH, download=True, train=True, transform=transform_fn)
            train_data, train_label = fetch_data_and_label(train_dst, args.num_classes)
            test_dst = datasets.CIFAR100(DATA_PATH, download=True, train=False, transform=transform_fn)
            test_data, test_label = fetch_data_and_label(test_dst, args.num_classes)
        elif args.dataset == "cifar20":
            assert args.num_classes == 20
            half_dim = 16
            train_dst = datasets.CIFAR100(DATA_PATH, download=True, train=True, transform=transform_fn)
            train_data, train_label = fetch_data_and_label(train_dst, args.num_classes)
            test_dst = datasets.CIFAR100(DATA_PATH, download=True, train=False, transform=transform_fn)
            test_data, test_label = fetch_data_and_label(test_dst, args.num_classes)
        elif args.dataset == "cifar10":
            half_dim = 16
            train_dst = datasets.CIFAR10(DATA_PATH, download=True, train=True, transform=transform_fn)
            train_data, train_label = fetch_data_and_label(train_dst, args.num_classes)
            test_dst = datasets.CIFAR10(DATA_PATH, download=True, train=False, transform=transform_fn)
            test_data, test_label = fetch_data_and_label(test_dst, args.num_classes)
        else:
            assert args.dataset == "mnist"
            half_dim = 14
            train_dst = datasets.MNIST("~/.torch", download=True, train=True, transform=transform_fn)
            train_data, train_label = fetch_data_and_label(train_dst, args.num_classes)
            test_dst = datasets.MNIST("~/.torch", download=True, train=False, transform=transform_fn)
            test_data, test_label = fetch_data_and_label(test_dst, args.num_classes)
        # poison image datasets
        if args.target_label == None:
            args.target_label = random.randint(0, args.num_classes-1)
            args.train_poison_list = random.sample(range(len(train_dst)), int(0.01 * len(train_dst)))
            args.test_poison_list = random.sample(range(len(test_dst)), int(0.01 * len(test_dst)))
        else:
            assert args.train_poison_list != None , "[[inner error]]"
            assert args.train_target_list != None, "[[inner error]]"
            assert args.test_poison_list != None, "[[inner error]]"
            assert args.test_target_list != None, "[[inner error]]"
        print(f"party#{index} target label={args.target_label}")
        train_data, train_label, train_poison_data, train_poison_label = generate_poison_data(train_data, train_label, args.train_poison_list, 'train', args.k, args.dataset)
        test_data, test_label, test_poison_data, test_poison_label = generate_poison_data(test_data, test_label, args.test_poison_list, 'test', args.k, args.dataset)
        if args.train_target_list == None:
            assert args.test_target_list == None
            args.train_target_list = random.sample(list(np.where(torch.argmax(train_label,axis=1)==args.target_label)[0]), 10)
            args.test_target_list = random.sample(list(np.where(torch.argmax(test_label,axis=1)==args.target_label)[0]), 10)
    # elif args.dataset_name == 'nuswide':
    #     half_dim = [634, 1000]
    #     train_dst = NUSWIDEDataset(DATA_PATH+'NUS_WIDE', 'train')
    #     test_dst = NUSWIDEDataset(DATA_PATH+'NUS_WIDE', 'test')
    # args.train_dataset = train_dst
    # args.val_dataset = test_dst
    else:
        assert args.dataset == 'mnist', "dataset not supported yet"

    train_dst = (train_data.to(args.device),train_label.to(args.device))
    test_dst = (test_data.to(args.device),test_label.to(args.device))
    train_poison_dst = (train_poison_data.to(args.device),train_poison_label.to(args.device))
    test_poison_dst = (test_poison_data.to(args.device),test_poison_label.to(args.device))

    train_dst = dataset_partition(args,index,train_dst,half_dim)
    test_dst = dataset_partition(args,index,test_dst,half_dim)
    train_poison_dst = dataset_partition(args,index,train_poison_dst,half_dim)
    test_poison_dst = dataset_partition(args,index,test_poison_dst,half_dim)
    # important
    return args, half_dim, train_dst, test_dst, train_poison_dst, test_poison_dst, args.train_target_list, args.test_target_list
    