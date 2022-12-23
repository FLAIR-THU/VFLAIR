import os

import numpy as np
import random
import pandas as pd
import torch
import sys

sys.path.append(os.pardir)
from utils.basic_functions import get_labeled_data
    
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


    print(train_dataset.y)

    # print(train_dataset.poison_list)

