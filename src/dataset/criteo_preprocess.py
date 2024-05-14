import csv
import os
from csv import DictReader

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
from glob import glob
import fnmatch


def get_x(csv_row, D):
    fullind = []
    for key, value in csv_row.items():
        s = key + '=' + value
        fullind.append(hash(s) % D)  # weakest hash ever ?? Not anymore :P

    indlist2 = []
    for i in range(len(fullind)):
        for j in range(i + 1, len(fullind)):
            indlist2.append(fullind[i] ^ fullind[j])  # Creating interactions using XOR
    fullind = fullind + indlist2

    x = [0.] * D
    for index in fullind:
        x[index] += 1

    return x  # x is a list of features that have a value as number of occurences


D_ = 2 ** 13  # number of weights use for learning
MAX_SAMPLE_NUM = 1e5  # 5e5
header = ['Label', 'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9', 'i10', 'i11', 'i12', 'i13', 'c1', 'c2', 'c3',
          'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15', 'c16', 'c17', 'c18', 'c19',
          'c20', 'c21', 'c22', 'c23', 'c24', 'c25', 'c26']
train_txt_file_path = '../../../../share_dataset/Criteo/train.txt'
output_file_path = '../../../../share_dataset/Criteo/criteo.csv'
reader = DictReader(open(train_txt_file_path), header, delimiter='\t')

# df = pd.read_csv(output_file_path)


with open(output_file_path, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    columns_names = [f"feat{i}" for i in range(D_)]
    columns_names.append('label')
    writer.writerow(columns_names)

    count = 0
    pre_label = 1.
    for id_, row in enumerate(reader):
        # print(count)
        y = 1. if row['Label'] == '1' else 0.

        if y != pre_label:
            pre_label = y
            count += 1
            del row['Label']  # can't let the model peek the answer
            # get the hashed features
            x = get_x(row, D_)
            # print(x, y)
            # write hashed features and label to criteo.csv
            # each row of criteo.csv is in this format: [hashed features, label]
            x.append(y)
            row_for_writer = x
            writer.writerow(row_for_writer)

            if count % 1e3 == 0:
                percent_done = 100 * count / MAX_SAMPLE_NUM
                print(f"{percent_done:.2f}% completed...")
            if count == MAX_SAMPLE_NUM:
                break
