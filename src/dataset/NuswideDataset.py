import os

import numpy as np
import random
import pandas as pd
import torch
import sys

sys.path.append(os.pardir)
from utils import get_labeled_data
    
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold

class NUSWIDEDataset():

    def __init__(self, data_dir, data_type):
        print("NUSWIDE Dataset initialize")
        
        # {'sky':74190, 'clouds':54087, 'person':51577, 'water':35264, 'animal':33887, 'grass':22561, 'buildings':17835, 'window':15051, 'plants':14345, 'lake':13392}

        self.data_dir = data_dir
        self.selected_labels = ['buildings', 'grass', 'animal', 'water', 'person']
        self.class_num = 5
        # self.selected_labels = ['clouds','person']
        # self.class_num = 2

        if data_type == 'train':
            X_image, X_text, Y = get_labeled_data(self.data_dir, self.selected_labels, 60000, 'Train')
        else:
            X_image, X_text, Y = get_labeled_data(self.data_dir, self.selected_labels, 40000, 'Test')
        #print(type(X_image), type(X_text), type(Y))
        self.x = [torch.tensor(X_image, dtype=torch.float32), torch.tensor(X_text, dtype=torch.float32)]
        self.y = torch.tensor(np.argmax(np.array(Y), axis=1), dtype=torch.long)
        self.y = torch.squeeze(self.y)
        # print('dataset data', self.x, self.y)
        # print(torch.sum(self.y))
        # check dataset
        print('dataset shape', self.x[0].shape, self.x[1].shape, self.y.shape)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, indexx):  # this is single_indexx
        return [self.x[0][indexx], self.x[1][indexx]], self.y[indexx]

if __name__ == '__main__':
    print("here inside main")
    data_dir = "./NUS_WIDE"

    # sel = get_top_k_labels(data_dir=data_dir, top_k=10)
    # print("sel", sel)
    # ['sky', 'clouds', 'person', 'water', 'animal', 'grass', 'buildings', 'window', 'plants', 'lake']

    # sel_lbls = get_top_k_labels(data_dir, 81)
    # print(sel_lbls)

    train_dataset = NUSWIDEDataset(data_dir, 'train')
    test_dataset = NUSWIDEDataset(data_dir, 'test')
    print(torch.sum(train_dataset.y), torch.sum(test_dataset.y))


    # '''X�?特征，不包含target;X_tsne�?已经降维之后的特�?'''
    # tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    # train_X0_tsne = tsne.fit_transform(train_dataset.x[0])
    # test_X0_tsne = tsne.fit_transform(test_dataset.x[0])
    # train_X1_tsne = tsne.fit_transform(train_dataset.x[1])
    # test_X1_tsne = tsne.fit_transform(test_dataset.x[1])
    # print("Org data dimension is {}. Embedded data dimension is {}".format(train_dataset.x[0].shape[1:], train_X0_tsne.shape[1:]))
        
    # '''嵌入空间�?视化'''
    # x_min, x_max = train_X0_tsne.min(0), train_X0_tsne.max(0)
    # X_norm = (train_X0_tsne - x_min) / (x_max - x_min)  # 归一�?
    # plt.figure(figsize=(8, 8))
    # for i in range(X_norm.shape[0]):
    #     plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(train_dataset.y[i]), 
    #             fontdict={'weight': 'bold', 'size': 9})
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()

    # x_min, x_max = test_X0_tsne.min(0), test_X0_tsne.max(0)
    # X_norm = (test_X0_tsne - x_min) / (x_max - x_min)  # 归一�?
    # plt.figure(figsize=(8, 8))
    # for i in range(X_norm.shape[0]):
    #     plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(test_dataset.y[i]), 
    #             fontdict={'weight': 'bold', 'size': 9})
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()

    print(train_dataset.y)

    # print(train_dataset.poison_list)

