import os
import random
import csv
import os
from decimal import Decimal
from io import BytesIO
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from matplotlib.ticker import MultipleLocator
from torchvision import models, datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from datetime import datetime
import logging
import copy
from sklearn.metrics import roc_auc_score
from utils.noisy_sample_functions import noisy_sample

tp = transforms.ToTensor()


def multiclass_auc(targets, scores):
    aucs = []
    for i in range(scores.shape[1]):
        if len(np.unique(targets[:, i])) == 1:
            continue
        auc = roc_auc_score(targets[:, i], scores[:, i])
        aucs.append(auc)
    return aucs


# For Distance Corrrelation Defense
def pairwise_dist(A, B):
    """
    Computes pairwise distances between each elements of A and each elements of
    B.
    Args:
        A,    [m,d] matrix
        B,    [n,d] matrix
    Returns:
        D,    [m,n] matrix of pairwise distances
    """
    # with tf.variable_scope('pairwise_dist'):
    # squared norms of each row in A and B
    na = torch.sum(torch.square(A), 1)
    nb = torch.sum(torch.square(B), 1)

    # na as a row and nb as a column vectors
    na = torch.reshape(na, [-1, 1])
    nb = torch.reshape(nb, [1, -1])

    # return pairwise euclidead difference matrix
    D = torch.sqrt(torch.maximum(na - 2 * torch.mm(A, B.T) + nb + 1e-20, torch.tensor(0.0)))
    return D


def tf_distance_cov_cor(input1, input2, debug=False):
    # n = tf.cast(tf.shape(input1)[0], tf.float32)
    n = torch.tensor(float(input1.size()[0]))
    a = pairwise_dist(input1, input1)
    b = pairwise_dist(input2, input2)

    # A = a - tf.reduce_mean(a,axis=1) - tf.expand_dims(tf.reduce_mean(a,axis=0),axis=1) + tf.reduce_mean(a)
    A = a - torch.mean(a, axis=1) - torch.unsqueeze(torch.mean(a, axis=0), axis=1) + torch.mean(a)
    B = b - torch.mean(b, axis=1) - torch.unsqueeze(torch.mean(b, axis=0), axis=1) + torch.mean(b)

    dCovXY = torch.sqrt(torch.sum(A * B) / (n ** 2) + 1e-16)
    dVarXX = torch.sqrt(torch.sum(A * A) / (n ** 2))
    dVarYY = torch.sqrt(torch.sum(B * B) / (n ** 2) + 1e-16)

    dCorXY = dCovXY / torch.sqrt(dVarXX * dVarYY)
    if debug:
        print(("tf distance cov: {} and cor: {}, dVarXX: {}, dVarYY:{}").format(
            dCovXY, dCorXY, dVarXX, dVarYY))
    # return dCovXY, dCorXY
    return dCorXY


def MSE_PSNR(batch_real_image, batch_dummy_image):
    '''
    compute MSE and PSNR
    :param batch_real_image:
    :param batch_dummy_image:
    :return:
    '''
    # print(batch_real_image.size(),batch_dummy_image.size())
    batch_real_image = batch_real_image.reshape(batch_dummy_image.size())
    mse = torch.mean((batch_real_image - batch_dummy_image) ** 2)
    psnr = 20 * torch.log10(1 / torch.sqrt(mse))
    return mse.item(), psnr.item()


def remove_exponent(num):
    num = Decimal(num)
    return num.to_integral() if num == num.to_integral() else num.normalize()


def sharpen(probabilities, T):
    if probabilities.ndim == 1:
        # print("here 1")
        tempered = torch.pow(probabilities, 1 / T)
        tempered = (
                tempered
                / (torch.pow((1 - probabilities), 1 / T) + tempered)
        )

    else:
        # print("here 2")
        tempered = torch.pow(probabilities, 1 / T)
        tempered = tempered / tempered.sum(dim=-1, keepdim=True)

    return tempered


def get_rand_batch(seed, class_num, batch_size, transform=None):
    path = './data/mini-imagenet/ok/'
    random.seed(seed)

    total_class = os.listdir(path)
    sample_class = random.sample(total_class, class_num)
    num_per_class = [batch_size // class_num] * class_num
    num_per_class[-1] += batch_size % class_num
    img_path = []
    labels = []

    for id, item in enumerate(sample_class):
        img_folder = os.path.join(path, item)
        img_path_list = [os.path.join(img_folder, img).replace('\\', '/') for img in os.listdir(img_folder)]
        sample_img = random.sample(img_path_list, num_per_class[id])
        img_path += sample_img
        labels += ([item] * num_per_class[id])
    img = []
    for item in img_path:
        x = Image.open(item)
        if transform is not None:
            x = transform(x)
        img.append(x)
    return img, labels


def entropy(predictions):
    epsilon = 1e-6
    H = -predictions * torch.log(predictions + epsilon)
    return torch.mean(H)


def calculate_entropy(matrix, N=2):
    class_counts = np.zeros(matrix.shape[0])
    all_counts = 0
    for row_idx, row in enumerate(matrix):
        for elem in row:
            class_counts[row_idx] += elem
            all_counts += elem

    weight_entropy = 0.0
    for row_idx, row in enumerate(matrix):
        norm_elem_list = []
        class_count = class_counts[row_idx]
        for elem in row:
            if elem > 0:
                norm_elem_list.append(elem / float(class_count))
        weight = class_count / float(all_counts)
        ent = numpy_entropy(np.array(norm_elem_list), N=N)
        weight_entropy += weight * ent
    return weight_entropy


def numpy_entropy(predictions, N=2):
    epsilon = 0
    H = -predictions * (np.log(predictions + epsilon) / np.log(N))
    return np.sum(H)


def img_show(img):
    plt.imshow(img.permute(1, 2, 0).detach().numpy())
    plt.show()


def draw_line_chart(title, note_list, x, y, x_scale, y_scale, label_x, label_y, path=None):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    for i in range(len(x)):
        plt.plot(x[i], y[i], marker='', mec='r', mfc='w', label=note_list[i], linewidth=2)
    plt.legend(fontsize=16)
    # plt.xticks(x, note_list, rotation=45)
    plt.margins(0)
    plt.xlabel(label_x, fontsize=15)
    plt.ylabel(label_y, fontsize=16)
    # plt.title(title, fontsize=14)
    plt.tick_params(labelsize=14)

    x_major_locator = MultipleLocator(x_scale)
    y_major_locator = MultipleLocator(y_scale)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xlim(min(x[0]), max(x[-1]))
    plt.ylim(0.8, 1.001)

    if path:
        plt.savefig(path[:-4] + '.png')
    plt.show()


def draw_scatter_chart(title, note_list, x, y, x_scale, y_scale, label_x, label_y, path=None):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    for i in range(len(x)):
        plt.plot(x[i], y[i], marker='', mec='r', mfc='w', label=note_list[i], linewidth=5)
    plt.legend(fontsize=14)
    # plt.xticks(x, note_list, rotation=45)
    plt.margins(0)
    plt.xlabel(label_x, fontsize=14)
    plt.ylabel(label_y, fontsize=14)
    # plt.title(title, fontsize=14)
    plt.tick_params(labelsize=14)

    x_major_locator = MultipleLocator(x_scale)
    y_major_locator = MultipleLocator(y_scale)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xlim(min(x[0]), max(x[0]))
    plt.ylim(0.8, 1.01)

    if path:
        plt.savefig(path[:-4] + '.png')
    plt.show()


def get_timestamp():
    return int(datetime.utcnow().timestamp())


def label_to_one_hot(target, num_classes=10):
    # print('label_to_one_hot:', target, type(target))
    try:
        _ = target.size()[1]
        # print("use target itself", target.size())
        onehot_target = target.type(torch.float32)
    except:
        target = torch.unsqueeze(target, 1)
        # print("use unsqueezed target", target.size())
        onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
        onehot_target.scatter_(1, target, 1)
    return onehot_target


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


def cross_entropy_for_one_hot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


def cross_entropy_for_onehot_samplewise(pred, target):
    return - target * F.log_softmax(pred, dim=-1)


def get_class_i(dataset, label_set):
    gt_data = []
    # gt_data = torch.tensor([])
    gt_labels = []
    num_cls = len(label_set)
    for j in range(len(dataset)):
        img, label = dataset[j]
        # print("in basic_functions::get_class_i", np.array(img))
        if label in label_set:
            label_new = label_set.index(label)
            # gt_data.append(img if torch.is_tensor(img) else tp(img))
            gt_data.append(np.array(img))
            # gt_data = torch.cat((gt_data, (img if torch.is_tensor(img) else tp(img))))
            gt_labels.append(label_new)
            # gt_labels.append(label_to_one_hot(torch.Tensor([label_new]).long(),num_classes=num_cls))
    # gt_data = torch.tensor([item.numpy() for item in gt_data])
    gt_data = torch.tensor(np.array(gt_data))
    gt_labels = label_to_one_hot(torch.Tensor(gt_labels).long(), num_classes=num_cls)
    # print(gt_data.size(),type(gt_data))
    return gt_data, gt_labels


def fetch_classes(num_classes):
    return np.arange(num_classes).tolist()


def fetch_data_and_label(dataset, num_classes):
    classes = fetch_classes(num_classes)
    return get_class_i(dataset, classes)


def append_exp_res(path, res):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(res + '\n')


def aggregate(classifier, logits_a, logits_b):
    if classifier:
        logits = torch.cat((logits_a, logits_b), dim=-1)
        return classifier(logits)
    else:
        return logits_a + logits_b


def transform_convert(img_tensor, transform):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_tensor = img_tensor.transpose(0, 2).transpose(0, 1)  # C x H x W  ---> H x W x C

    if 'ToTensor' in str(transform) or img_tensor.max() < 1:
        img_tensor = img_tensor.detach().numpy() * 255

    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.numpy()

    if img_tensor.shape[2] == 3:
        img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        img = Image.fromarray(img_tensor.astype('uint8')).squeeze()
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))

    return img


########################## nuswide utils

def get_all_nuswide_labels():
    nuswide_labels = []
    for line in os.listdir('data/NUS_WIDE/Groundtruth/AllLabels'):
        nuswide_labels.append(line.split('_')[1][:-4])
    return nuswide_labels


def balance_X_y(XA, XB, y, seed=5):
    np.random.seed(seed)
    num_pos = np.sum(y == 1)
    # num_neg = np.sum(y == -1)
    # pos_indexes = [i for (i, _y) in enumerate(y) if _y > 0]
    # neg_indexes = [i for (i, _y) in enumerate(y) if _y < 0]

    num_neg = np.sum(y == 0)
    pos_indexes = [i for (i, _y) in enumerate(y) if _y > 0.5]
    neg_indexes = [i for (i, _y) in enumerate(y) if _y < 0.5]

    print("len(pos_indexes)", len(pos_indexes))
    print("len(neg_indexes)", len(neg_indexes))
    print("num of samples", len(pos_indexes) + len(neg_indexes))
    print("num_pos:", num_pos)
    print("num_neg:", num_neg)

    if num_pos < num_neg:
        np.random.shuffle(neg_indexes)
        # randomly pick negative samples of size equal to that of positive samples
        rand_indexes = neg_indexes[:num_pos]
        indexes = pos_indexes + rand_indexes
        np.random.shuffle(indexes)
        y = [y[i] for i in indexes]
        XA = [XA[i] for i in indexes]
        XB = [XB[i] for i in indexes]

    return np.array(XA), np.array(XB), np.array(y)


def get_top_k_labels(data_dir, top_k=5):
    data_path = "NUS_WIDE/Groundtruth/AllLabels"
    label_counts = {}
    for filename in os.listdir(os.path.join(data_dir, data_path)):
        file = os.path.join(data_dir, data_path, filename)
        # print(file)
        if os.path.isfile(file):
            label = file[:-4].split("_")[-1]
            df = pd.read_csv(os.path.join(data_dir, file))
            df.columns = ['label']
            label_counts[label] = (df[df['label'] == 1].shape[0])
    label_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    selected = [k for (k, v) in label_counts[:top_k]]
    return selected


def get_labeled_data(data_dir, selected_label, n_samples, dtype="Train"):
    # get labels
    data_path = "Groundtruth/TrainTestLabels/"
    dfs = []
    for label in selected_label:
        file = os.path.join(data_dir, data_path, "_".join(["Labels", label, dtype]) + ".txt")
        df = pd.read_csv(file, header=None)
        # print("df shape", df.shape)
        df.columns = [label]
        # print(df)
        dfs.append(df)
    data_labels = pd.concat(dfs, axis=1)
    # print(data_labels)
    if len(selected_label) > 1:
        selected = data_labels[data_labels.sum(axis=1) == 1]
    else:
        selected = data_labels
    # print("selected:", selected)
    # get XA, which are image low level features
    # features_path = "NUS_WID_Low_Level_Features/Low_Level_Features"
    features_path = "Low_Level_Features"
    # print("data_dir: {0}".format(data_dir))
    # print("features_path: {0}".format(features_path))
    dfs = []
    for file in os.listdir(os.path.join(data_dir, features_path)):
        if file.startswith("_".join([dtype, "Normalized"])):
            df = pd.read_csv(os.path.join(data_dir, features_path, file), header=None, sep=" ")
            df.dropna(axis=1, inplace=True)
            # print(df)
            # print("b datasets features", len(df.columns))
            dfs.append(df)
    data_XA = pd.concat(dfs, axis=1)
    data_X_image_selected = data_XA.loc[selected.index]
    # print("X image shape:", data_X_image_selected.shape)  # 634 columns
    # get XB, which are tags
    tag_path = "NUS_WID_Tags/"
    file = "_".join([dtype, "Tags1k"]) + ".dat"
    tagsdf = pd.read_csv(os.path.join(data_dir, tag_path, file), header=None, sep="\t")
    tagsdf.dropna(axis=1, inplace=True)
    data_X_text_selected = tagsdf.loc[selected.index]
    # print("X text shape:", data_X_text_selected.shape)
    # print(data_X_image_selected.values[0].shape, data_X_text_selected.values[0].shape, selected.values[0].shape)
    if n_samples is None:
        return data_X_image_selected.values[:], data_X_text_selected.values[:], selected.values[:]
    return data_X_image_selected.values[:n_samples], data_X_text_selected.values[:n_samples], selected.values[
                                                                                              :n_samples]


def image_and_text_data(data_dir, selected, n_samples=2000):
    return get_labeled_data(data_dir, selected, n_samples)


def get_images():
    # image_urls = "data/NUS_WIDE/NUS_WIDE/NUS-WIDE-urls/NUS-WIDE-urls.txt"
    image_urls = "data/NUS_WIDE/NUS-WIDE-urls/NUS-WIDE-urls.txt"
    # df = pd.read_csv(image_urls, header=0, sep=" ")
    # print(df.head(10))
    # kkk = df.loc[:, "url_Middle"]
    # print(kkk.head(10))

    read_num_urls = 1
    with open(image_urls, "r") as fi:
        fi.readline()
        reader = csv.reader(fi, delimiter=' ', skipinitialspace=True)
        for idx, row in enumerate(reader):
            if idx >= read_num_urls:
                break
            print(row[0], row[2], row[3], row[4])
            if row[3] is not None and row[3] != "null":
                url = row[4]
                print("{0} url: {1}".format(idx, url))

                str_array = row[0].split("\\")
                print(str_array[3], str_array[4])

                # img = imageio.imread(url)
                # print(type(img), img.shape)

                response = requests.get(url)
                print(response.status_code)
                img = Image.open(BytesIO(response.content))
                arr = np.array(img)
                print(type(img), arr.shape)
                # imageio.imwrite("", img)
                size = 48, 48
                img.thumbnail(size)
                img.show()
                arr = np.array(img)
                print("thumbnail", arr.shape)


def generate_poison_data(args, data, label, poison_list, _type, k, dataset, party_index):
    '''
    generate poisoned image data
    '''
    if dataset == 'nuswide':
        # X_image = data[0]
        # X_test = data[1]
        # data = torch.tensor([torch.tensor(X_text, dtype=torch.float32), torch.tensor(X_image, dtype=torch.float32)]）
        # mixed_data_image, poison_list = data_poison(data[0], poison_list, k, dataset)
        # poison_data_image = copy.deepcopy(mixed_data_image[poison_list])
        mixed_data_text, poison_list = data_poison_text(args, data[0], poison_list, k, dataset, party_index)
        poison_data_text = copy.deepcopy(mixed_data_text[poison_list])

        poison_data_image = torch.tensor(data[1][poison_list])
        poison_data_text = torch.tensor(poison_data_text)
        poison_data = [poison_data_text, poison_data_image]
        # torch.tensor([data[0][poison_list],poison_data_text].cpu().numpy())
        poison_label = copy.deepcopy(label[poison_list])
        # print(f"poison data and label have size {poison_data.size()} and {poison_label.size()}")
        if _type == 'train':
            data[0] = torch.tensor(np.delete(data[0].cpu().numpy(), poison_list, axis=0))
            data[1] = torch.tensor(np.delete(data[1].cpu().numpy(), poison_list, axis=0))
            label = torch.tensor(np.delete(label.cpu().numpy(), poison_list, axis=0))
        return data, label, poison_data, poison_label

    elif dataset in ['breast_cancer_diagnose', 'diabetes', 'adult_income', 'criteo']:
        mixed_data_text, poison_list = data_poison_text(args, data, poison_list, k, dataset, party_index)
        poison_data = copy.deepcopy(mixed_data_text[poison_list])
        poison_data = torch.tensor(poison_data)
        poison_label = copy.deepcopy(label[poison_list])
        if _type == 'train':
            data = torch.tensor(np.delete(data.cpu().numpy(), poison_list, axis=0))
            label = torch.tensor(np.delete(label.cpu().numpy(), poison_list, axis=0))

        return data, label, poison_data, poison_label

    else:
        mixed_data, poison_list = data_poison(args, data, poison_list, k, dataset, party_index)
        poison_data = copy.deepcopy(mixed_data[poison_list])
        poison_label = copy.deepcopy(label[poison_list])
        # print(f"poison data and label have size {poison_data.size()} and {poison_label.size()}")
        if _type == 'train':
            data = torch.tensor(np.delete(data.cpu().numpy(), poison_list, axis=0))
            label = torch.tensor(np.delete(label.cpu().numpy(), poison_list, axis=0))

        # print(torch.argmax(label,axis=1)==target_label)
        # print(np.where(torch.argmax(label,axis=1)==target_label))
        # print(np.where(torch.argmax(label,axis=1)==target_label)[0])
        # target_list = random.sample(list(np.where(torch.argmax(label,axis=1)==target_label)[0]), 10)

        return data, label, poison_data, poison_label


def data_poison(args, images, poison_list, k, dataset, party_index):
    target_pixel_value = [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]
    if args.apply_ns:
        assert 'noise_lambda' in args.attack_configs, 'need parameter: noise_lambda'
        assert 'noise_rate' in args.attack_configs, 'need parameter: noise_rate'
        assert 'party' in args.attack_configs, 'need parameter: party'
        noise_rate = args.attack_configs['noise_rate'] if ('noise_rate' in args.attack_configs) else 0.01
        scale = args.attack_configs['noise_lambda'] if ('noise_lambda' in args.attack_configs) else 2.0
        # print(f"[debug] in basic_function.py, args.attack_configs['party']={args.attack_configs['party']}")
        # print(f"[debug] in basic_function.py, party_index={party_index}, {type(party_index)}, k in  args.attack_configs['party']={party_index in args.attack_configs['party']}")
        if party_index in args.attack_configs['party']:  # if not attacker party, return unchanged image
            # print(f'[debug] in basic_function.py, party {party_index} poison')
            # print(f'[debug] in basic_function.py, images[0]={images[0]}')
            images[poison_list] = noisy_sample(images[poison_list], scale)
    else:
        if 'cifar' in dataset.casefold():
            if k == 2:  # 1 party poison, passive party-0 poison
                images[poison_list, 0, 15, 31] = target_pixel_value[0][0]
                images[poison_list, 0, 14, 30] = target_pixel_value[0][1]
                images[poison_list, 0, 13, 31] = target_pixel_value[0][2]
                images[poison_list, 0, 15, 29] = target_pixel_value[0][3]
                images[poison_list, 1, 15, 31] = target_pixel_value[1][0]
                images[poison_list, 1, 14, 30] = target_pixel_value[1][1]
                images[poison_list, 1, 13, 31] = target_pixel_value[1][2]
                images[poison_list, 1, 15, 29] = target_pixel_value[1][3]
                images[poison_list, 2, 15, 31] = target_pixel_value[2][0]
                images[poison_list, 2, 14, 30] = target_pixel_value[2][1]
                images[poison_list, 2, 13, 31] = target_pixel_value[2][2]
                images[poison_list, 2, 15, 29] = target_pixel_value[2][3]
            elif k == 4:
                # 3 party poison, passive party-[0,1,2] poison
                images[poison_list, 0, 15, 15] = target_pixel_value[0][0]
                images[poison_list, 1, 15, 15] = target_pixel_value[1][0]
                images[poison_list, 2, 15, 15] = target_pixel_value[2][0]
                images[poison_list, 0, 15, 31] = target_pixel_value[0][1]
                images[poison_list, 1, 15, 31] = target_pixel_value[1][1]
                images[poison_list, 2, 15, 31] = target_pixel_value[2][1]
                images[poison_list, 0, 31, 15] = target_pixel_value[0][2]
                images[poison_list, 1, 31, 15] = target_pixel_value[1][2]
                images[poison_list, 2, 31, 15] = target_pixel_value[2][2]
            else:
                assert k == 4, "poison type not supported yet"
        elif 'mnist' in dataset.casefold():
            if k == 2:
                images[poison_list, 0, 13, 27] = 1.0
                images[poison_list, 0, 12, 26] = 1.0
                images[poison_list, 0, 11, 27] = 1.0
                images[poison_list, 0, 13, 25] = 1.0
            elif k == 4:
                images[poison_list, 0, 13, 13] = 1.0  # 3 party poison
                images[poison_list, 0, 13, 27] = 1.0  # 3 party poison
                images[poison_list, 0, 27, 13] = 1.0  # 3 party poison
            else:
                assert k == 4, "poison type not supported yet"
        else:
            assert 'mnist' in dataset.casefold(), "dataset not supported yet"
    return images, poison_list


def data_poison_text(args, texts, poison_list, k, dataset, party_index):
    '''
    text or tabular data
    trigger: set the last element as target_text_value(1)
    '''
    if args.apply_ns:
        assert 'noise_lambda' in args.attack_configs, 'need parameter: noise_lambda'
        assert 'noise_rate' in args.attack_configs, 'need parameter: noise_rate'
        assert 'party' in args.attack_configs, 'need parameter: party'
        noise_rate = args.attack_configs['noise_rate'] if ('noise_rate' in args.attack_configs) else 0.1
        scale = args.attack_configs['noise_lambda']
        if party_index in args.attack_configs[
            'party']:  # only if party is in the attacker party pool, poison the test data
            if 'nuswide' in dataset.casefold():
                if k == 2:  # 1 party poison, passive party-0 poison
                    texts[poison_list] = noisy_sample(texts[poison_list], scale)
                else:
                    assert k == 2, "poison type not supported yet"
            elif 'breast_cancer_diagnose' in dataset.casefold():
                if k == 2:  # first feature of attacker(pasive party 0) set to 0.1
                    texts[poison_list] = noisy_sample(texts[poison_list], scale)
                else:
                    assert k == 2, "poison type not supported yet"
            else:
                assert 'mnist' in dataset.casefold(), "dataset not supported yet"
        return texts, poison_list
    else:
        if 'nuswide' in dataset.casefold():
            # if k == 2: # 1 party poison, passive party-0 poison
            #     texts[poison_list,-1] = 0
            # else:
            #     assert k == 2, "poison type not supported yet"
            assert poison_list != None, "nuswide dataset poison list should not be none, and no operation needs to be done"
        elif 'breast_cancer_diagnose' in dataset.casefold():
            if k == 2:  # first feature of attacker(pasive party 0) set to 0.1
                texts[poison_list, 0] = 0.1
            else:
                assert k == 2, "poison type not supported yet"

        else:
            assert 'mnist' in dataset.casefold(), "dataset not supported yet"
    return texts, poison_list


# def generate_poison_data_text(args,data, label, poison_list, _type, k, dataset, party_index):
#     '''
#     generate poisoned text data
#     '''
#     mixed_data, poison_list = data_poison_text(args,data, poison_list, k, dataset)
#     poison_data = copy.deepcopy(mixed_data[poison_list])
#     poison_label = copy.deepcopy(label[poison_list])
#     # print(f"poison data and label have size {poison_data.size()} and {poison_label.size()}")
#     if _type == 'train':
#         data = torch.tensor(np.delete(data.cpu().numpy(), poison_list, axis=0))
#         label = torch.tensor(np.delete(label.cpu().numpy(), poison_list, axis=0))

#     return data, label, poison_data, poison_label


def entropy(predictions):
    epsilon = 1e-6
    H = -predictions * torch.log(predictions + epsilon)
    # print("H:", H.shape)
    return torch.mean(H)


def calculate_entropy(matrix, N=2):
    class_counts = np.zeros(matrix.shape[0])
    all_counts = 0
    for row_idx, row in enumerate(matrix):
        for elem in row:
            class_counts[row_idx] += elem
            all_counts += elem

    # print("class_counts", class_counts)
    # print("all_counts", all_counts)

    weight_entropy = 0.0
    for row_idx, row in enumerate(matrix):
        norm_elem_list = []
        class_count = class_counts[row_idx]
        for elem in row:
            if elem > 0:
                norm_elem_list.append(elem / float(class_count))
        weight = class_count / float(all_counts)
        # weight = 1 / float(len(matrix))
        ent = numpy_entropy(np.array(norm_elem_list), N=N)
        # print("norm_elem_list:", norm_elem_list)
        # print("weight:", weight)
        # print("ent:", ent)
        weight_entropy += weight * ent
    return weight_entropy


def get_timestamp():
    return int(datetime.utcnow().timestamp())


def numpy_entropy(predictions, N=2):
    # epsilon = 1e-10
    # epsilon = 1e-8
    epsilon = 0
    # print(np.log2(predictions + epsilon))
    H = -predictions * (np.log(predictions + epsilon) / np.log(N))
    # print("H:", H.shape)
    return np.sum(H)
    # return H


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('darts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path, mode='w')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1024. / 1024.


class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        reset()

    def reset(self):
        """ Reset all statistics """
        val = 0
        avg = 0
        sum = 0
        count = 0

    def update(self, val, n=1):
        """ Update statistics """
        val = val
        sum += val * n
        count += n
        avg = sum / count


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # print('correct', correct)

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy2(output, target, M, device, topk=(1, 2,)):
    alpha = 0.001
    # print(output.shape)
    maxk = max(topk)
    batch_size = target.size(0)
    pred_count, pred_index = output.topk(maxk, 1, True, True)
    # print(pred_value,pred_index)
    # print("pred.shape", pred_count.shape,batch_size) #[64, 2],batch_size=N=64 for MNIST

    pred = pred_index.t()[0]
    for i in range(pred.shape[0]):
        pa = pred_count[i][0] / M
        pb = pred_count[i][1] / M
        shift = np.sqrt(np.log(1 / alpha) / (2 * batch_size))
        # print("pa=",pa,", pb=",pb," ,shift=",shift)
        pa = pa - shift
        pb = pb + shift
        # print("pa=",pa,", pb=",pb," ,shift=",shift)
        if pa <= pb:
            pred[i] = -1
    # print(pred)
    # print(target)

    # print('pred in device :',pred.device)
    # print('target in device :',target.device)
    pred = pred.cuda()
    target = target.cuda()
    correct = pred.eq(target.view(1, -1))
    # correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in (1,):
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy3(output, target):
    batch_size = target.size(0)
    correct = output.eq(target).sum()
    # print('correct', correct.sum())
    return correct * (100.0 / batch_size)


def vote(output, topk=(1,)):
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()[0]
    return pred


def create_exp_dir(path, scripts_to_save=None):
    os.makedirs(path, exist_ok=True)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.makedirs(os.path.join(path, 'scripts'), exist_ok=True)
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def save_checkpoint(state, ckpt_dir, is_best=False):
    os.makedirs(ckpt_dir, exist_ok=True)
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)


def ClipAndPerturb(vector, device, ro, sigma):
    _norm = np.linalg.norm(vector.cpu().detach().numpy(), ord=1)
    # print("L2 norm of parameter =",_norm)
    vector = vector / max(1, (_norm / ro))
    vector.to(device)
    vector += torch.normal(0.0, sigma * sigma, vector.shape).to(device)
    return vector
