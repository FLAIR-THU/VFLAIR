import os
import random

import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms


class MNISTDatasetVFLPERROUND():

    def __init__(self, data_dir, data_type, height, width, poison_number, target_number=10, target_label=0):
        self.data_dir = data_dir
        self.target_label = target_label
        self.transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor()
        ])
        if data_type == 'train':
            images = np.load(os.path.join(self.data_dir, 'mnist_images_train.npy')).astype('float') / 255.0
            labels = np.load(os.path.join(self.data_dir, 'mnist_labels_train.npy'))
        else:
            images = np.load(os.path.join(self.data_dir, 'mnist_images_test.npy')).astype('float') / 255.0
            labels = np.load(os.path.join(self.data_dir, 'mnist_labels_test.npy'))

        print('[in model]', data_type, "data input:", images.shape, labels.shape)
        images = images[:, :, :, np.newaxis]

        # images_up = images[:,:14]
        # images_down = images[:,14:]
        image_list = [images[:, :14], images[:, 14:]]
        # image_list = [images[:,:14,:14],images[:,14:,:14],images[:,:14,14:],images[:,14:,14:]]

        # images_down, poison_list = data_poison(images_down, poison_number)
        image_list, poison_list = data_poison(image_list, poison_number)

        self.poison_images = [image_list[il][poison_list] for il in range(len(image_list))]
        self.poison_labels = labels[poison_list]

        # print("after seperation, len(image_list) =",len(image_list)," image_list[i].shape = ",image_list[0].shape, image_list[1].shape,image_list[2].shape, image_list[3].shape)
        if data_type == 'train':
            self.x = [np.delete(image_list[il], poison_list, axis=0) for il in range(len(image_list))]
            self.y = np.delete(labels, poison_list, axis=0)
        else:
            self.x = image_list
            self.y = labels
        self.poison_list = poison_list

        self.target_list = random.sample(list(np.where(self.y == target_label)[0]), target_number)
        print('[in model]', data_type, "target list is:", self.target_list)

        # check dataset
        print('[in model]', data_type, 'dataset shape', self.x[0].shape, self.x[1].shape, self.y.shape)
        # print('[in model]', data_type, 'dataset shape', self.x[0].shape, self.x[1].shape, self.x[2].shape, self.x[3].shape, self.y.shape)
        print('[in model]', data_type, 'target data', self.y[self.target_list].shape, np.mean(self.y[self.target_list]),
              target_label)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, indexx):  # this is single_indexx
        data = []
        labels = []
        for i in range(len(self.x)):
            data.append(self.x[i][indexx])
        labels.append(self.y[indexx])

        return data, np.array(labels).ravel()

    def get_poison_data(self):
        return self.poison_images, self.poison_labels

    def get_target_data(self):
        return [self.x[il][self.target_list] for il in range(len(self.x))], self.y[self.target_list]

    def get_poison_list(self):
        return self.poison_list

    def get_target_list(self):
        return self.target_list


def data_poison(images, poison_number):
    poison_list = random.sample(range(images[0].shape[0]), poison_number)
    images[1][poison_list, 13, 27] = 1.0
    images[1][poison_list, 12, 26] = 1.0
    images[1][poison_list, 11, 27] = 1.0
    images[1][poison_list, 13, 25] = 1.0

    # images[1][poison_list, 13, 13] = 1.0 # 3 party poison
    # # images[1][poison_list, 12, 13] = 1.0 # 3 party poison
    # images[2][poison_list, 13, 13] = 1.0 # 3 party poison
    # # images[2][poison_list, 12, 13] = 1.0 # 3 party poison
    # images[3][poison_list, 13, 13] = 1.0 # 3 party poison
    # # images[3][poison_list, 12, 12] = 1.0 # 3 party poison
    # # images[3][poison_list, 11, 13] = 1.0 # 3 party poison
    # # images[3][poison_list, 13, 11] = 1.0 # 3 party poison
    return images, poison_list


def visualize(images, labels, poison_list):
    class_names = ['0', '1', '2', '3', '4',
                   '5', '6', '7', '8', '9']

    plt.figure(figsize=(10, 10))
    poisoned_images = images[poison_list]
    poisoned_labels = labels[poison_list]
    print(poisoned_labels)
    print(poisoned_images.shape)
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(poisoned_images[i].squeeze(), cmap='Greys')
        plt.xlabel(class_names[poisoned_labels[i]])
    plt.show()


def need_poison_down_check_mnist_vfl_per_round(images):
    need_poison_list = [True if images[1][indx, 13, 27] > 0.9 and \
                                images[1][indx, 12, 26] > 0.9 and \
                                images[1][indx, 11, 27] > 0.9 and \
                                images[1][indx, 13, 25] > 0.9 else False \
                        for indx in range(len(images))]
    # need_poison_list = [True if images[1][indx,13, 13] > 0.99 and \
    #                     images[2][indx,13,13] > 0.99 and \
    #                     images[3][indx,13,13] > 0.99 else False\
    #                     for indx in range(len(images[0]))]
    return np.array(need_poison_list)


if __name__ == '__main__':
    ds = MNISTDatasetVFLPERROUND('E:/dataset/MNIST', 'train', 28, 28, 60)

    # visualize(ds.x[1], ds.y, ds.poison_list)
    # visualize(ds.x[0], ds.y, ds.poison_list)

    res = need_poison_down_check_mnist_vfl_per_round(ds.x)
    print(res.sum())
