import os
import random

import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class Cifar100DatasetVFLPERROUND():

    def __init__(self, data_dir, data_type, height, width, poison_number, target_number=10, target_label=0):
        self.data_dir = data_dir
        self.target_label = target_label
        if data_type == 'train':
            data = unpickle(os.path.join(self.data_dir, 'train'))
        else:
            data = unpickle(os.path.join(self.data_dir, 'test'))

        images, labels = data[b'data'], data[b'fine_labels']
        # images = images.astype('float32').reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1) / 255.0
        print("==> images:", images[0])
        images = images.astype('float32').reshape((-1, 3, 32, 32)) / 255.0

        labels = np.array(labels)

        print("cifar100 images shape:", images.shape)
        print("cifar100 labels shape:", labels.shape)

        selected_idx = labels < 20
        labels = labels[selected_idx]
        images = images[selected_idx]

        print("cifar100 images shape:", images.shape)
        print("cifar100 labels shape:", labels.shape)

        # images_up = images[:, :, :16]
        # images_down = images[:, :, 16:]
        image_list = [images[:, :, :16, :16], images[:, :, :16, 16:], images[:, :, 16:, :16], images[:, :, 16:, 16:]]

        print('[in model]', data_type, 'image_list[0,1,2,3].shape:', image_list[0].shape, image_list[1].shape,
              image_list[2].shape, image_list[3].shape, labels.shape)
        image_list, poison_list = data_poison(image_list, poison_number)

        self.poison_images = [image_list[il][poison_list] for il in range(len(image_list))]
        self.poison_labels = labels[poison_list]

        if data_type == 'train':
            self.x = [np.delete(image_list[il], poison_list, axis=0) for il in range(len(image_list))]
            self.y = np.delete(labels, poison_list, axis=0)
        else:
            self.x = image_list
            self.y = labels
        self.poison_list = poison_list

        self.target_list = random.sample(list(np.where(self.y == target_label)[0]), target_number)
        print('[in model]', data_type, "target list is:", self.target_list)

        print('[in model]', data_type, 'dataset shape', self.x[0].shape, self.x[1].shape, self.y.shape)
        print('[in model]', data_type, self.y[self.target_list].shape, np.mean(self.y[self.target_list]), target_label)

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
    target_pixel_value = [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]
    poison_list = random.sample(range(images[0].shape[0]), poison_number)
    # 1 party poison
    # images[poison_list, 0, 15, 31] = target_pixel_value[0][0]
    # images[poison_list, 0, 14, 30] = target_pixel_value[0][1]
    # images[poison_list, 0, 13, 31] = target_pixel_value[0][2]
    # images[poison_list, 0, 15, 29] = target_pixel_value[0][3]
    # images[poison_list, 1, 15, 31] = target_pixel_value[1][0]
    # images[poison_list, 1, 14, 30] = target_pixel_value[1][1]
    # images[poison_list, 1, 13, 31] = target_pixel_value[1][2]
    # images[poison_list, 1, 15, 29] = target_pixel_value[1][3]
    # images[poison_list, 2, 15, 31] = target_pixel_value[2][0]
    # images[poison_list, 2, 14, 30] = target_pixel_value[2][1]
    # images[poison_list, 2, 13, 31] = target_pixel_value[2][2]
    # images[poison_list, 2, 15, 29] = target_pixel_value[2][3]

    # 3 party poison
    images[1][poison_list, 0, 15, 15] = target_pixel_value[0][0]
    images[1][poison_list, 1, 15, 15] = target_pixel_value[1][0]
    images[1][poison_list, 2, 15, 15] = target_pixel_value[2][0]
    images[2][poison_list, 0, 15, 15] = target_pixel_value[0][1]
    images[2][poison_list, 1, 15, 15] = target_pixel_value[1][1]
    images[2][poison_list, 2, 15, 15] = target_pixel_value[2][1]
    images[3][poison_list, 0, 15, 15] = target_pixel_value[0][2]
    images[3][poison_list, 1, 15, 15] = target_pixel_value[1][2]
    images[3][poison_list, 2, 15, 15] = target_pixel_value[2][2]

    return images, poison_list


def visualize(images, labels, poison_list):
    # class_names = ['0', '1', '2', '3', '4',
    #                '5', '6', '7', '8', '9']

    class_names = [str(i) for i in range(100)]

    plt.figure(figsize=(10, 10))
    poisoned_images = images[poison_list]
    poisoned_labels = labels[poison_list]
    print(poisoned_images.shape)
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(poisoned_images.transpose(0, 2, 3, 1)[i])  # , cmap='Greys')
        # print(class_names[poisoned_labels[i][0]])
        plt.xlabel(class_names[poisoned_labels[i]])
    plt.show()


def need_poison_down_check_cifar100_vfl_per_round(images):
    target_pixel_value = [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]
    need_poison_list = [True if \
                            images[1][indx, 0, 15, 15] == target_pixel_value[0][0] and images[2][indx, 0, 15, 15] ==
                            target_pixel_value[0][1] and images[3][indx, 0, 15, 15] == target_pixel_value[0][2] and \
                            images[1][indx, 1, 15, 15] == target_pixel_value[1][0] and images[2][indx, 1, 15, 15] ==
                            target_pixel_value[1][1] and images[3][indx, 1, 15, 15] == target_pixel_value[1][2] and \
                            images[1][indx, 2, 15, 15] == target_pixel_value[2][0] and images[2][indx, 2, 15, 15] ==
                            target_pixel_value[2][1] and images[3][indx, 2, 15, 15] == target_pixel_value[2][2] \
                            else False for indx in range(len(images[0]))]
    return np.array(need_poison_list)


if __name__ == '__main__':
    ds = Cifar100DatasetVFLPERROUND('E:/dataset/cifar-100-python', 'train', 32, 32, 500)

    visualize(ds.x[1], ds.y, ds.poison_list)
    # visualize(ds.x[0], ds.y, ds.poison_list)

    res = need_poison_down_check_cifar100_vfl_per_round(ds.x[1])
    print(res.sum())
