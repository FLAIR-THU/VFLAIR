import random

import numpy as np

from .nus_wide_data_util import get_labeled_data


class NUSWIDEDatasetVFLPERROUND():

    def __init__(self, data_dir, data_type, target_number=10, target_label=0):
        self.data_dir = data_dir
        self.target_label = target_label
        self.selected_labels = ['buildings', 'grass', 'animal', 'water', 'person']
        self.class_num = 5

        print(self.selected_labels)
        if data_type == 'train':
            X_image, X_text, Y = get_labeled_data(self.data_dir, self.selected_labels, 60000, 'Train')
        else:
            X_image, X_text, Y = get_labeled_data(self.data_dir, self.selected_labels, 40000, 'Test')
        print(type(X_image), type(X_text), type(Y))

        poison_idx = need_poison_down_check_nuswide_vfl_per_round(np.array(X_text).astype('float32'))
        poison_list = np.nonzero(poison_idx)[0]
        self.poison_list = poison_list

        self.poison_images = [np.array(X_image).astype('float32')[poison_list],
                              np.array(X_text).astype('float32')[poison_list]]
        self.poison_labels = np.argmax(np.array(Y), axis=1).astype('float32')[poison_list]

        if data_type == 'train':
            self.x = [np.delete(np.array(X_image).astype('float32'), self.poison_list, axis=0),
                      np.delete(np.array(X_text).astype('float32'), self.poison_list, axis=0)]
            self.y = np.delete(np.argmax(np.array(Y), axis=1), poison_list, axis=0)
        else:
            self.x = [np.array(X_image).astype('float32'), np.array(X_text).astype('float32')]
            self.y = np.argmax(np.array(Y), axis=1).astype('float32')

        self.target_list = random.sample(list(np.where(self.y == target_label)[0]), target_number)
        print(self.target_list)

        # check dataset
        print('dataset shape', self.x[0].shape, self.x[1].shape, self.y.shape)
        print('target data', self.y[self.target_list].shape, np.mean(self.y[self.target_list]), target_label)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, indexx):  # this is single_indexx
        data = []
        labels = []
        for i in range(2):
            data.append(self.x[i][indexx])
        labels.append(self.y[indexx])

        return data, np.array(labels).ravel()

    def get_poison_data(self):
        return self.poison_images, self.poison_labels

    def get_target_data(self):
        return [self.x[0][self.target_list], self.x[1][self.target_list]], self.y[self.target_list]

    def get_poison_list(self):
        return self.poison_list

    def get_target_list(self):
        return self.target_list


def need_poison_down_check_nuswide_vfl_per_round(images):
    need_poison_list = [True if images[indx, -1] == 1 else False \
                        for indx in range(len(images))]
    return np.array(need_poison_list)


def data_poison():
    pass


if __name__ == '__main__':
    data_dir = "../../dataset/NUS_WIDE"

    # sel = get_top_k_labels(data_dir=data_dir, top_k=10)
    # print("sel", sel)
    # ['sky', 'clouds', 'person', 'water', 'animal', 'grass', 'buildings', 'window', 'plants', 'lake']

    # sel_lbls = get_top_k_labels(data_dir, 81)
    # print(sel_lbls)

    train_dataset = NUSWIDEDataset(data_dir, 'train')
    print(train_dataset.y)

    # print(train_dataset.poison_list)

    res = need_poison_down_check_nuswide_vfl_per_round(train_dataset.x[1])
    print(res.sum())
