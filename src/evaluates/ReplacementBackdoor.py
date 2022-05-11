import argparse
import copy
import glob
import logging
import os
import pickle
import random
import sys
import time
import utils

import torch.nn.functional as F
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
from tensorboardX import SummaryWriter

# from dataset.cifar10_dataset_vfl_replace_per_comm import Cifar10DatasetVFLPERROUND, \
#     need_poison_down_check_cifar10_vfl_per_round
# from dataset.cifar100_dataset_vfl_replace_per_comm import Cifar100DatasetVFLPERROUND, \
#     need_poison_down_check_cifar100_vfl_per_round
# from dataset.mnist_dataset_vfl_per_comm import MNISTDatasetVFLPERROUND, need_poison_down_check_mnist_vfl_per_round
# from dataset.nuswide_dataset_vfl_per_comm import NUSWIDEDatasetVFLPERROUND, need_poison_down_check_nuswide_vfl_per_round
from utils.dataset.ReplacementBackdoor.mnist_multi import MNISTDatasetVFLPERROUND, need_poison_down_check_mnist_vfl_per_round

from models.model_templates import Backdoor_ClassificationModelGuest, \
    MLP2, Backdoor_ClassificationModelHostHead, Backdoor_ClassificationModelHostTrainableHead, Backdoor_ClassificationModelHostHeadWithSoftmax, \
    SimpleCNN
from models.resnet_torch import resnet18, resnet50
# from models.vision import LeNetCIFAR2, LeNetMNIST, LeNetCIFAR3, LeNet5, LeNetCIFAR1, LeNet5_2
from utils.basic_functions import *
from utils.defense_functions import multistep_gradient


def transform_to_pred_labels(logits, encoder):
    enc_predict_prob = F.softmax(logits, dim=-1)
    dec_predict_prob = encoder.decoder(enc_predict_prob)
    return torch.argmax(dec_predict_prob, dim=-1)


# def main():
#     parser = argparse.ArgumentParser("backdoor")
#     parser.add_argument('--name', type=str, default='perround', help='experiment name')

#     parser.add_argument('--model', default='mlp2', help='resnet')
#     parser.add_argument('--input_size', type=int, default=28, help='resnet')
#     parser.add_argument('--backdoor', type=int, default=1)
    
#     parser.add_argument('--explicit_softmax', type=int, default=0)
#     parser.add_argument('--random_output', type=int, default=0)  

#     parser.add_argument("--certify", type=int, default=0, help="CertifyFLBaseline")
#     parser.add_argument("--M", type=int, default=1000, help="voting party count in CertifyFL")
#     parser.add_argument("--sigma", type=float, default=0, help='sigma for certify')
#     parser.add_argument("--adversarial_start_epoch", type=int, default=0, help="value of adversarial start epoch")
#     parser.add_argument("--certify_start_epoch", type=int, default=1, help="number of epoch when the cerfity ClipAndPerturb start")


class ReplacementBackdoor(object):

    def __init__(self, args):
        self.device = args.device
        self.gpu = args.gpu
        self.dataset = args.dataset
        self.epochs = args.epochs
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.models_dict = args.model_list
        self.num_classes = args.num_class_list[0]
        self.k = args.k
        self.seed = args.seed
        self.backdoor = 1

        self.report_freq = args.report_freq
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.gamma = args.learning_rate_decay_rate # learning rate decay rate
        self.decay_period = args.decay_period
        self.workers = args.worker_thread_number
        # self.grad_clip = args.grad_clip_value
        # self.label_smooth = args.label_smooth
        self.amplify_rate = args.amplify_rate
        # self.amplify_rate_output = args.amplify_rate_output
        self.use_project_head = args.use_project_head
        self.explicit_softmax = args.explicit_softmax
    
        # self.apply_trainable_layer = args.apply_trainable_layer
        self.apply_laplace = args.apply_laplace
        self.apply_gaussian = args.apply_gaussian
        self.dp_strength = args.dp_strength
        self.apply_grad_spar = args.apply_grad_spar
        self.grad_spars = args.grad_spars
        self.apply_encoder = args.apply_encoder
        self.ae_lambda = args.ae_lambda
        self.encoder = args.encoder
        self.apply_marvell = args.apply_marvell
        self.marvell_s = args.marvell_s
        self.apply_discrete_gradients = args.apply_discrete_gradients
        self.discrete_gradients_bins = args.discrete_gradients_bins
        # self.discrete_gradients_bound = args.discrete_gradients_bound
        self.discrete_gradients_bound = 1e-3

        # self.apply_ppdl = args.apply_ppdl
        # self.ppdl_theta_u = args.ppdl_theta_u
        # self.apply_gc = args.apply_gc
        # self.gc_preserved_percent = args.gc_preserved_percent
        # self.apply_lap_noise = args.apply_lap_noise
        # self.noise_scale = args.noise_scale
        # self.apply_discrete_gradients = args.apply_discrete_gradients

        self.name = args.exp_res_dir
        self.name = self.name + '{}-{}-{}-{}-{}'.format(args.epochs, args.batch_size, args.amplify_rate, args.seed, time.strftime("%Y%m%d-%H%M%S"))
        create_exp_dir(self.name)
        print(self.name)

    def train(self):
        amplify_rate = self.amplify_rate

        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(self.name, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)

        # tensorboard
        writer = SummaryWriter(log_dir=os.path.join(self.name, 'tb'))
        writer.add_text('experiment', self.name, 0)

        logging.info('***** USED DEVICE: {}'.format(self.device))


        if torch.cuda.is_available():
            torch.cuda.set_device(self.gpu)
            cudnn.benchmark = True
            cudnn.enabled = True
            # torch.cuda.manual_seed_all(self.seed)

        ##### set dataset
        input_dims = None

        if self.dataset == 'mnist':
            NUM_CLASSES = 10
            if self.k == 2:
                input_dims = [14 * 28, 14 * 28]
            else:
                assert self.k == 4, "only support 2 or 4 party"
                input_dims = [14 * 14, 14 * 14, 14 * 14, 14 * 14]
            self.input_size = 28
            DATA_DIR = './utils/dataset/MNIST'
            target_label = random.randint(0, NUM_CLASSES-1)
            logging.info('target label: {}'.format(target_label))

            train_dataset = MNISTDatasetVFLPERROUND(DATA_DIR, 'train', self.input_size, self.input_size, 600, 10, target_label)
            valid_dataset = MNISTDatasetVFLPERROUND(DATA_DIR, 'test', self.input_size, self.input_size, 100, 10, target_label)

            # set poison_check function
            need_poison_down_check = need_poison_down_check_mnist_vfl_per_round

        # elif self.dataset == 'nuswide':
        #     NUM_CLASSES = 5
        #     input_dims = [634, 1000]
        #     DATA_DIR = './utils/dataset/NUS_WIDE'
        #     # target_label = random.randint(0, NUM_CLASSES-1)
        #     # target_label = random.sample([0, 1, 3, 4], 1)[0]
        #     target_label = 1
        #     logging.info('target label: {}'.format(target_label))

        #     train_dataset = NUSWIDEDatasetVFLPERROUND(DATA_DIR, 'train', 10, target_label)
        #     valid_dataset = NUSWIDEDatasetVFLPERROUND(DATA_DIR, 'test', 10, target_label)

        #     # set poison_check function
        #     need_poison_down_check = need_poison_down_check_nuswide_vfl_per_round

        # elif self.dataset == 'cifar10':
        #     NUM_CLASSES = 10
        #     input_dims = [16 * 16, 16 * 16, 16 * 16, 16 * 16]
        #     self.input_size = 32
        #     num_ftrs = 1024

        #     DATA_DIR = './dataset/cifar-10-batches-py'

        #     # target_label = random.randint(0, NUM_CLASSES-1)
        #     target_label = 9
        #     logging.info('target label: {}'.format(target_label))

        #     train_dataset = Cifar10DatasetVFLPERROUND(DATA_DIR, 'train', self.input_size, self.input_size, 500, 10, target_label)
        #     valid_dataset = Cifar10DatasetVFLPERROUND(DATA_DIR, 'test', self.input_size, self.input_size, 100, 10, target_label)

        #     # set poison_check function
        #     need_poison_down_check = need_poison_down_check_cifar10_vfl_per_round

        # elif self.dataset == 'cifar100':
        #     # NUM_CLASSES = 100
        #     NUM_CLASSES = 20
        #     input_dims = [16 * 16, 16 * 16, 16 * 16, 16 * 16]
        #     self.input_size = 32
        #     num_ftrs = 1024

        #     DATA_DIR = './dataset/cifar-100-python'

        #     target_label = random.randint(0, NUM_CLASSES-1)
        #     logging.info('target label: {}'.format(target_label))

        #     train_dataset = Cifar100DatasetVFLPERROUND(DATA_DIR, 'train', self.input_size, self.input_size, 500, 10, target_label)
        #     valid_dataset = Cifar100DatasetVFLPERROUND(DATA_DIR, 'test', self.input_size, self.input_size, 100, 10, target_label)

        #     # set poison_check function
        #     need_poison_down_check = need_poison_down_check_cifar100_vfl_per_round
        else:
            raise Exception(f"does not support {self.dataset}")
            # TODO: this do not support cifar20 but dp and sparce do support

        n_train = len(train_dataset)
        n_valid = len(valid_dataset)

        train_indices = list(range(n_train))
        valid_indices = list(range(n_valid))
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=self.batch_size,
                                                # sampler=train_sampler,
                                                num_workers=self.workers,
                                                shuffle=False,
                                                pin_memory=True,
                                                drop_last=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                batch_size=self.batch_size,
                                                sampler=valid_sampler,
                                                num_workers=self.workers,
                                                pin_memory=True)

        # check poisoned samples
        print('train poison samples:', sum(need_poison_down_check(train_dataset.x)))
        print('test poison samples:', sum(need_poison_down_check(valid_dataset.x)))
        print(train_dataset.poison_list[:10])
        poison_list = train_dataset.poison_list

        ##### set model
        local_models = []
        if self.models_dict['0']['type'] == 'MLP2':
            for i in range(self.k):
                backbone = MLP2(input_dims[i], NUM_CLASSES)
                local_models.append(backbone)
        elif self.models_dict['0']['type'] == 'resnet18':
            for i in range(self.k):
                backbone = resnet18(NUM_CLASSES)
                local_models.append(backbone)
        elif self.models_dict['0']['type'] == 'resnet50':
            for i in range(self.k):
                backbone = resnet50(NUM_CLASSES)
                local_models.append(backbone)
        # elif self.models_dict['0']['type'] == 'simplecnn':
        #     for i in range(self.k):
        #         backbone = SimpleCNN(NUM_CLASSES)
        #         local_models.append(backbone)
        # elif self.models_dict['0']['type'] == 'lenet':
        #     print(f"[INFO] using LeNet")
        #     for i in range(self.k):
        #         backbone = LeNetCIFAR2(NUM_CLASSES)
        #         # backbone = LeNet5_2(NUM_CLASSES)
        #         local_models.append(backbone)

        criterion = nn.CrossEntropyLoss()

        apply_encoder = self.apply_encoder
        if apply_encoder == 1:
            print("[INFO] apply encoder for defense")
            encoder = self.encoder
        else:
            print("[INFO] does not apply encoder for defense")
            encoder = None

        model_list = []
        for i in range(self.k+1):
            if i == 0:
                if self.use_project_head == 1:
                    active_model = Backdoor_ClassificationModelHostTrainableHead(NUM_CLASSES * self.k, NUM_CLASSES).to(self.device)
                    logging.info('Trainable active party')
                else:
                    if self.explicit_softmax == 1:
                        active_model = Backdoor_ClassificationModelHostHeadWithSoftmax().to(self.device)
                        criterion = nn.NLLLoss()
                        logging.info('Non-trainable active party with softmax layer')
                    else:
                        active_model = Backdoor_ClassificationModelHostHead().to(self.device)
                    logging.info('Non-trainable active party')
            else:
                model_list.append(Backdoor_ClassificationModelGuest(local_models[i - 1]))

        local_models = None
        model_list = [model.to(self.device) for model in model_list]

        criterion = criterion.to(self.device)

        # weights optimizer
        optimizer_active_model = None
        optimizer_list = []
        if self.use_project_head == 1:
            optimizer_active_model = torch.optim.SGD(active_model.parameters(), self.learning_rate, momentum=self.momentum,
                                                    weight_decay=self.weight_decay)
            optimizer_list = [
                torch.optim.SGD(model.parameters(), self.learning_rate, momentum=self.momentum,
                                weight_decay=self.weight_decay)
                for model in model_list]
        else:
            optimizer_list = [
                torch.optim.SGD(model.parameters(), self.learning_rate, momentum=self.momentum,
                                weight_decay=self.weight_decay)
                for model in model_list]

        scheduler_list = []
        if self.learning_rate == 0.025:
            if optimizer_active_model is not None:
                scheduler_list.append(
                    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_active_model, float(self.epochs)))
            scheduler_list = scheduler_list + [
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(self.epochs))
                for optimizer in optimizer_list]
        else:
            if optimizer_active_model is not None:
                scheduler_list.append(
                    torch.optim.lr_scheduler.StepLR(optimizer_active_model, self.decay_period, gamma=self.gamma))
            scheduler_list = [torch.optim.lr_scheduler.StepLR(optimizer, self.decay_period, gamma=self.gamma) for optimizer
                            in optimizer_list]

        assert(len(model_list)==self.k)
        assert(len(optimizer_list)==self.k)
        assert(len(scheduler_list)==self.k)

        best_acc_top1 = 0.

        feat_need_copy = copy.deepcopy(train_dataset.x[1][train_dataset.target_list[0]])
        expand_factor = len(feat_need_copy.shape)

        # get train backdoor data
        train_backdoor_images, train_backdoor_true_labels = train_dataset.get_poison_data()

        # get train target data
        train_target_images, train_target_labels = train_dataset.get_target_data()
        print('train target data', train_target_images[0].shape, train_target_labels)
        print('train poison samples:', sum(need_poison_down_check(train_backdoor_images))) # should not be zero when using backdoor

        # get test backdoor data
        test_backdoor_images, test_backdoor_true_labels = valid_dataset.get_poison_data()

        # set test backdoor label
        test_backdoor_labels = copy.deepcopy(test_backdoor_true_labels)
        test_backdoor_labels[:] = valid_dataset.target_label

        target_label = train_dataset.target_label
        print('the label of the sample need copy = ', train_dataset.target_label, valid_dataset.target_label)

        # init some sample data for debug
        sample_list_finished = False
        while not sample_list_finished:
            sample_list = random.sample(range(valid_dataset.x[0].shape[0]), 100)
            common_list = [x for x in sample_list if x in valid_dataset.poison_list]
            if len(common_list) < 5:
                sample_list_finished = True

        test_sample_images, test_sample_labels = [train_dataset.x[i][sample_list] for i in range(self.k)], \
                                                train_dataset.y[sample_list]

        debug_log_list = [[], [], [], [], [], []]
        active_gradients_res_list = [None for _ in range(self.k)]

        # loop
        for epoch in range(self.epochs):

            output_replace_count = 0
            gradient_replace_count = 0

            ########### TRAIN ###########
            top1 = AverageMeter()
            losses = AverageMeter()

            cur_step = epoch * len(train_loader)
            cur_lr = optimizer_list[0].param_groups[0]['lr']
            # logging.info("Epoch {} LR {}".format(epoch, cur_lr))
            writer.add_scalar('train/lr', cur_lr, cur_step)
            
            for model in model_list:
                active_model.train()
                model.train()

            for step, (trn_X, trn_y) in enumerate(train_loader):

                # select one backdoor data
                id = random.randint(0, train_backdoor_images[0].shape[0] - 1)
                # backdoor_image_up = train_backdoor_images[0][id]
                # backdoor_image_down = train_backdoor_images[1][id]
                backdoor_image_list = [train_backdoor_images[il][id] for il in range(self.k)]
                backdoor_label = train_backdoor_true_labels[id]
                # select one target data
                id = random.randint(0, train_target_images[0].shape[0] - 1)
                target_image_list = [train_target_images[il][id] for il in range(self.k)]

                # merge normal train data with selected backdoor and target data
                trn_X_list = []
                for i in range(self.k):
                    trn_X_list.append(np.concatenate([trn_X[i].numpy(), np.expand_dims(backdoor_image_list[i], 0), np.expand_dims(target_image_list[i],0)]))
                trn_y = np.concatenate([trn_y.numpy(), np.array([[backdoor_label]]), np.array([[target_label]])])

                # trn_X_up = torch.from_numpy(trn_X_up).float().to(self.device)
                # trn_X_down = torch.from_numpy(trn_X_down).float().to(self.device)
                for i in range(self.k):
                    trn_X_list[i] = torch.from_numpy(trn_X_list[i]).float().to(self.device)
                target = torch.from_numpy(trn_y).view(-1).long().to(self.device)

                N = target.size(0)

                # # passive party 0 generate output
                # z_up = model_list[0](trn_X_up)
                # z_up_clone = z_up.detach().clone()
                # z_up_clone = torch.autograd.Variable(z_up_clone, requires_grad=True).to(self.device)

                # # passive party 1 generate output
                # z_down = model_list[1](trn_X_down)
                # z_down_clone = z_down.detach().clone()
                
                # passive party 0~3 generate output
                z_list = []
                z_list_clone = []
                for i in range(self.k):
                    z_list.append(model_list[i](trn_X_list[i]))
                    z_list_clone.append(z_list[i].detach().clone())

                # print('z_down_clone before', z_down_clone)

                ########### backdoor: replace output of passive party ##########
                if self.backdoor == 1:
                    with torch.no_grad():
                        for i in range(self.k-1):
                            # attack are 1,2,...,k-1; active party is 0
                            z_list_clone[i+1][-1] = z_list_clone[i+1][-2] # replace target data output using poisoned data output
                        output_replace_count = output_replace_count + 1
                ########### backdoor end here ##########

                # print('z_down_clone after', z_down_clone)

                # z_down_clone = torch.autograd.Variable(z_down_clone, requires_grad=True).to(self.device)
                for i in range(self.k):
                    z_list_clone[i] = torch.autograd.Variable(z_list_clone[i], requires_grad=True).to(self.device)

                # active party backward
                # logits = active_model(z_up_clone, z_down_clone)
                logits = active_model(z_list_clone)

                # TODO:
                if encoder:
                    target_one_hot = label_to_onehot(target, num_classes=NUM_CLASSES)
                    _, tr_target_one_hot = encoder(target_one_hot)
                    # print("target shape:", target.shape)
                    # print("target_one_hot shape:", target_one_hot.shape)
                    # print("tr_target_one_hot shape:", tr_target_one_hot.shape)
                    # print("logits shape:", logits.shape)
                    loss = cross_entropy_for_one_hot(logits, tr_target_one_hot)
                else:
                    loss = criterion(logits, target)

                # z_gradients_up = torch.autograd.grad(loss, z_up_clone, retain_graph=True)
                # z_gradients_down = torch.autograd.grad(loss, z_down_clone, retain_graph=True)
                z_gradients_list = [torch.autograd.grad(loss, z_list_clone[i], retain_graph=True) for i in range(self.k)]

                # z_gradients_up_clone = z_gradients_up[0].detach().clone()
                # z_gradients_down_clone = z_gradients_down[0].detach().clone()
                z_gradients_list_clone = [(z_gradients_list[i][0].detach().clone()) for i in range(self.k)]

                if self.apply_discrete_gradients:
                    z_gradients_list_clone = [multistep_gradient(z_gradients_list[i][0].detach().clone(), bins_num=self.discrete_gradients_bins, bound_abs=self.discrete_gradients_bound) for i in range(self.k)]

                # update active model
                if optimizer_active_model is not None:
                    optimizer_active_model.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer_active_model.step()

                ########### defense start here ##########
                location = 0.0
                threshold = 0.2
                if self.apply_laplace:
                    with torch.no_grad():
                        scale = self.dp_strength
                        # clip 2-norm per sample
                        # norm_factor_up = torch.div(torch.max(torch.norm(z_gradients_up_clone, dim=1)),
                        #                            threshold + 1e-6).clamp(min=1.0)
                        # norm_factor_down = torch.div(torch.max(torch.norm(z_gradients_down_clone, dim=1)),
                        #                            threshold + 1e-6).clamp(min=1.0)
                        norm_factor_list = [(torch.div(torch.max(torch.norm(z_gradients_list_clone[i], dim=1)),
                                                threshold + 1e-6).clamp(min=1.0)) for i in range(self.k)]

                        # add laplace noise
                        # dist_up = torch.distributions.laplace.Laplace(location, scale)
                        # dist_down = torch.distributions.laplace.Laplace(location, scale)
                        dist_list = [torch.distributions.laplace.Laplace(location, scale) for _ in range(self.k)]

                        # if self.defense_up == 1:
                        #     z_gradients_up_clone = torch.div(z_gradients_up_clone, norm_factor_up) + \
                        #                            dist_up.sample(z_gradients_up_clone.shape).to(self.device)
                        # z_gradients_down_clone = torch.div(z_gradients_down_clone, norm_factor_down) + \
                        #                          dist_down.sample(z_gradients_down_clone.shape).to(self.device)
                        # since we alwasys have self.defense_up==1
                        z_gradients_list_clone = [torch.div(z_gradients_list_clone[i], norm_factor_list[i]) + \
                                                dist_list[i].sample(z_gradients_list_clone[i].shape).to(self.device) for i in range(self.k)]
                        # z_gradients_list_clone.to(self.device)

                if self.apply_gaussian:
                    with torch.no_grad():
                        scale = self.dp_strength
                        # norm_factor_up = torch.div(torch.max(torch.norm(z_gradients_up_clone, dim=1)),
                        #                            threshold + 1e-6).clamp(min=1.0)
                        # norm_factor_down = torch.div(torch.max(torch.norm(z_gradients_down_clone, dim=1)),
                        #                            threshold + 1e-6).clamp(min=1.0)
                        norm_factor_list = [(torch.div(torch.max(torch.norm(z_gradients_list_clone[i], dim=1)),
                                                threshold + 1e-6).clamp(min=1.0)) for i in range(self.k)]
                        # if self.defense_up == 1:
                        #     z_gradients_up_clone = torch.div(z_gradients_up_clone, norm_factor_up) + \
                        #                            torch.normal(location, scale, z_gradients_up_clone.shape).to(self.device)
                        # z_gradients_down_clone = torch.div(z_gradients_down_clone, norm_factor_down) + \
                        #                          torch.normal(location, scale, z_gradients_down_clone.shape).to(self.device)
                        # since we alwasys have self.defense_up==1
                        z_gradients_list_clone = [torch.div(z_gradients_list_clone[i], norm_factor_list[i]) + \
                                                torch.normal(location, scale, z_gradients_list_clone[i].shape).to(self.device) for i in range(self.k)]
                        # z_gradients_list_clone.to(self.device)
                if self.apply_grad_spar != 0:
                    with torch.no_grad():
                        percent = self.grad_spars / 100.0
                        # if active_up_gradients_res is not None and \
                        #         z_gradients_up_clone.shape[0] == active_up_gradients_res.shape[0]:
                        #     z_gradients_up_clone = z_gradients_up_clone + active_up_gradients_res
                        # if active_down_gradients_res is not None and \
                        #         z_gradients_down_clone.shape[0] == active_down_gradients_res.shape[0]:
                        #     z_gradients_down_clone = z_gradients_down_clone + active_down_gradients_res
                        for i in range(self.k):
                            if active_gradients_res_list[i] is not None and \
                                    z_gradients_list_clone[i].shape[0] == active_gradients_res_list[i].shape[0]:
                                z_gradients_list_clone[i] = z_gradients_list_clone[i] + active_gradients_res_list[i]
                        
                        # up_thr = torch.quantile(torch.abs(z_gradients_up_clone), percent)
                        # down_thr = torch.quantile(torch.abs(z_gradients_down_clone), percent)
                        thr_list = []
                        for i in range(self.k):
                            thr_list.append(torch.quantile(torch.abs(z_gradients_list_clone[i]), percent))
                        
                        # active_up_gradients_res = torch.where(torch.abs(z_gradients_up_clone).double() < up_thr.item(),
                        #                                       z_gradients_up_clone.double(), float(0.)).to(self.device)
                        # active_down_gradients_res = torch.where(torch.abs(z_gradients_down_clone).double() < down_thr.item(),
                        #                                       z_gradients_down_clone.double(), float(0.)).to(self.device)
                        for i in range(self.k):
                            active_gradients_res_list[i] = torch.where(torch.abs(z_gradients_list_clone[i]).double() < thr_list[i].item(),
                                                            z_gradients_list_clone[i].double(), float(0.)).to(self.device)

                        # if self.defense_up == 1:
                        #     z_gradients_up_clone = z_gradients_up_clone - active_up_gradients_res
                        # z_gradients_down_clone = z_gradients_down_clone - active_down_gradients_res
                        # since we alwasys have self.defense_up==1
                        z_gradients_list_clone = [(z_gradients_list_clone[i] - active_gradients_res_list[i]) for i in range(self.k)]
                        # z_gradients_list_clone.to(self.device)
                ########### defense end here ##########

                # update passive model 0
                optimizer_list[0].zero_grad()
                weights_gradients_list = []
                weights_gradients_list.append(torch.autograd.grad(z_list[0], model_list[0].parameters(),
                                                        grad_outputs=z_gradients_list_clone[0]))

                for w, g in zip(model_list[0].parameters(), weights_gradients_list[0]):
                    if w.requires_grad:
                        w.grad = g.detach()
                optimizer_list[0].step()

                # print('z_gradients_down_clone before', z_gradients_down_clone)

                ########### backdoor: replace gradient for poisoned data ##########
                if self.backdoor == 1:
                    with torch.no_grad():
                        for i in range(self.k-1):
                            z_gradients_list_clone[i+1][-2] = z_gradients_list_clone[i+1][-1]*amplify_rate # replace the received poisoned gradient using target gradient,contradict with the paper??? 
                        gradient_replace_count = gradient_replace_count + 1
                ########### backdoor end here ##########

                # print('z_gradients_down_clone before', z_gradients_down_clone)

                # update passive model 1~3
                # optimizer_list[1].zero_grad()
                # weights_gradients_down = torch.autograd.grad(z_down[:-1], model_list[1].parameters(),
                #                                              grad_outputs=z_gradients_down_clone[
                #                                                           :-1])  # , retain_graph=True)

                # for w, g in zip(model_list[1].parameters(), weights_gradients_down):
                #     if w.requires_grad:
                #         w.grad = g.detach()
                # optimizer_list[1].step()
                for i in range(self.k-1):
                    optimizer_list[i+1].zero_grad()
                    if self.backdoor == 1:
                        weights_gradients_list.append(torch.autograd.grad(z_list[i+1][:-1], model_list[i+1].parameters(),
                                                                            grad_outputs=z_gradients_list_clone[i+1][:-1]))
                    else:
                        weights_gradients_list.append(torch.autograd.grad(z_list[i+1], model_list[i+1].parameters(),
                                                                            grad_outputs=z_gradients_list_clone[i+1]))
                    for w,g in zip(model_list[i+1].parameters(), weights_gradients_list[i+1]):
                        if w.requires_grad:
                            w.grad = g.detach()
                    optimizer_list[i+1].step()

                # train metrics
                prec1 = accuracy(logits, target, topk=(1,))
                losses.update(loss.item(), N)
                top1.update(prec1[0].item(), N)

                writer.add_scalar('train/loss', losses.avg, cur_step)
                writer.add_scalar('train/top1', top1.avg, cur_step)
                cur_step += 1

            # validation
            cur_step = (epoch + 1) * len(train_loader)

            ########### VALIDATION ###########

            top1_valid = AverageMeter()
            losses_valid = AverageMeter()

            for model in model_list:
                active_model.eval()
                model.eval()

            with torch.no_grad():
                # test accuracy
                for step, (val_X, val_y) in enumerate(valid_loader):
                    val_X = [x.float().to(self.device) for x in val_X]
                    target = val_y.view(-1).long().to(self.device)
                    N = target.size(0)

                    # z_up = model_list[0](val_X[0])
                    # z_down = model_list[1](val_X[1])
                    z_list = [model_list[i](val_X[i]) for i in range(self.k)]

                    # logits = active_model(z_up, z_down)
                    logits = active_model(z_list)

                    if encoder:
                        enc_predict_prob = F.softmax(logits, dim=-1)
                        dec_predict_prob = encoder.decoder(enc_predict_prob)
                        predict_label = torch.argmax(dec_predict_prob, dim=-1)
                        prec1 = accuracy3(predict_label, target)
                        # print("logits shape:", logits.shape)
                        # print("target shape:", target.shape)
                        # print("enc_predict_prob shape:", enc_predict_prob.shape)
                        # print("dec_predict_prob shape:", dec_predict_prob.shape)
                        # print("predict_label shape:", predict_label.shape)
                        # print("prec1 shape:", prec1.shape)

                        top1_valid.update(prec1, N)

                    else:
                        # TODO:
                        loss = criterion(logits, target)

                        prec1 = accuracy(logits, target, topk=(1,))

                        losses_valid.update(loss.item(), N)
                        top1_valid.update(prec1[0].item(), N)

                # backdoor related accuracy
                # backdoor_X_up = torch.from_numpy(test_backdoor_images[0]).float().to(self.device)
                # backdoor_X_down = torch.from_numpy(test_backdoor_images[1]).float().to(self.device)
                backdoor_X_list = [torch.from_numpy(test_backdoor_images[i]).float().to(self.device) for i in range(self.k)]
                backdoor_labels = torch.from_numpy(test_backdoor_labels).long().to(self.device)
                backdoor_true_labels = torch.from_numpy(test_backdoor_true_labels).long().to(self.device)

                # sample_X_up = torch.from_numpy(test_sample_images[0]).float().to(self.device)
                # sample_X_down = torch.from_numpy(test_sample_images[1]).float().to(self.device)
                sample_X_list = [torch.from_numpy(test_sample_images[i]).float().to(self.device) for i in range(self.k)]
                sample_true_labels = torch.from_numpy(test_sample_labels).float().to(self.device)

                backdoor_X_down_target_list = []
                for i in range(self.k-1):
                    if expand_factor == 1:
                        backdoor_X_down_target_list.append(torch.from_numpy(feat_need_copy).repeat(backdoor_X_list[i+1].shape[0],
                                                                                        1).float().to(self.device))
                    elif expand_factor == 2:
                        backdoor_X_down_target_list.append(torch.from_numpy(feat_need_copy).repeat(backdoor_X_list[i+1].shape[0], 1,
                                                                                        1).float().to(self.device))
                    else:
                        backdoor_X_down_target_list.append(torch.from_numpy(feat_need_copy).repeat(backdoor_X_list[i+1].shape[0], 1, 1,
                                                                                    1).float().to(self.device))

                N = backdoor_labels.shape[0]

                # z_up = model_list[0](backdoor_X_up)
                # z_down = model_list[1](backdoor_X_down)
                z_list = [model_list[i](backdoor_X_list[i]) for i in range(self.k)]

                # z_up_sample = model_list[0](sample_X_up)
                # z_down_sample = model_list[1](sample_X_down)
                z_list_sample = [model_list[i](sample_X_list[i]) for i in range(self.k)]

                # # random up output
                # std_z_up, mean_z_up = torch.std_mean(z_up)
                # z_up_random = torch.normal(mean_z_up.item(), std_z_up.item(), z_up.shape).float().to(self.device)

                # # random down output
                # std_z_down, mean_z_down = torch.std_mean(z_down)
                # z_down_random = torch.normal(mean_z_down.item(), std_z_down.item(), z_down.shape).float().to(self.device)

                # # target down output
                # z_down_target = model_list[1](backdoor_X_down_target)
                # std_z_down_target, mean_z_down_target = torch.std_mean(z_down_target)

                ########## backdoor metric

                if encoder:
                    logits_backdoor = active_model(z_list)
                    pre_backdoor_label = transform_to_pred_labels(encoder=encoder, logits=logits_backdoor)
                    acc = accuracy3(pre_backdoor_label, backdoor_labels)

                    losses_backdoor = 0.0
                    top1_backdoor = acc

                    # logits_random_up = active_model(z_up_random, z_down)
                    # pre_random_up = transform_to_pred_labels(encoder=encoder, logits=logits_random_up)
                    # acc = accuracy3(pre_random_up, backdoor_labels)
                    # losses_backdoor_random_up = 0.0
                    # top1_backdoor_random_up = acc

                    # logits_randon_down = active_model(z_up, z_down_random)
                    # pre_random_down = transform_to_pred_labels(encoder=encoder, logits=logits_randon_down)
                    # acc = accuracy3(pre_random_down, backdoor_true_labels)
                    # losses_backdoor_random_down = 0.0
                    # top1_backdoor_random_down = acc

                    # logits_target_down = active_model(z_up, z_down_target)
                    # pre_target_down = transform_to_pred_labels(encoder=encoder, logits=logits_target_down)
                    # acc = accuracy3(pre_target_down, backdoor_labels)

                    # losses_backdoor_target_down = 0.0
                    # top1_backdoor_target_down = acc

                else:
                    logits_backdoor = active_model(z_list)
                    loss_backdoor = criterion(logits_backdoor, backdoor_labels)
                    prec1 = accuracy(logits_backdoor, backdoor_labels, topk=(1,))

                    losses_backdoor = loss_backdoor.item()
                    top1_backdoor = prec1[0]

                    # ########## backdoor metric with random up output
                    # logits_random_up = active_model(z_up_random, z_down)
                    # loss_random_up = criterion(logits_random_up, backdoor_labels)

                    # prec1 = accuracy(logits_random_up, backdoor_labels, topk=(1,))

                    # losses_backdoor_random_up = loss_random_up.item()
                    # top1_backdoor_random_up = prec1[0]

                    # ########## test_accuracy with random down output
                    # logits_randon_down = active_model(z_up, z_down_random)
                    # loss_random_down = criterion(logits_randon_down, backdoor_true_labels)

                    # prec1 = accuracy(logits_randon_down, backdoor_true_labels, topk=(1,))

                    # losses_backdoor_random_down = loss_random_down.item()
                    # top1_backdoor_random_down = prec1[0]

                    # ########## test_accuracy with target down output
                    # logits_target_down = active_model(z_up, z_down_target)
                    # loss_target_down = criterion(logits_target_down, backdoor_labels)

                    # prec1 = accuracy(logits_target_down, backdoor_labels, topk=(1,))

                    # losses_backdoor_target_down = loss_target_down.item()
                    # top1_backdoor_target_down = prec1[0]

                # debug_log_list[0].append(z_up_sample.cpu().numpy())
                # debug_log_list[1].append(z_down_sample.cpu().numpy())
                # debug_log_list[2].append(z_up.cpu().numpy())
                # debug_log_list[3].append(z_down.cpu().numpy())
                # debug_log_list[4].append(sample_true_labels.cpu().numpy())
                # debug_log_list[5].append(backdoor_true_labels.cpu().numpy())

            writer.add_scalar('val/loss', losses_valid.avg, cur_step)
            writer.add_scalar('val/top1_valid', top1_valid.avg, cur_step)
            writer.add_scalar('backdoor/loss', losses_backdoor, cur_step)
            writer.add_scalar('backdoor/top1_valid', top1_backdoor, cur_step)

            template = 'Epoch {}, Poisoned {}/{}, Loss: {:.4f}, Accuracy: {:.2f}, ' \
                    'Test Loss: {:.4f}, Test Accuracy: {:.2f}, ' \
                    'Backdoor Loss: {:.4f}, Backdoor Accuracy: {:.2f}\n'

            logging.info(template.format(epoch + 1,
                                        output_replace_count,
                                        gradient_replace_count,
                                        losses.avg,
                                        top1.avg,
                                        losses_valid.avg,
                                        top1_valid.avg,
                                        losses_backdoor,
                                        top1_backdoor.item()
                                        ))

            if losses_valid.avg > 1e8 or np.isnan(losses_valid.avg):
                logging.info('********* INSTABLE TRAINING, BREAK **********')
                break

            valid_acc_top1 = top1_valid.avg
            # save
            if valid_acc_top1 > best_acc_top1:
                best_acc_top1 = valid_acc_top1
            logging.info('best_acc_top1 %f', best_acc_top1)

            # update scheduler
            for scheduler in scheduler_list:
                scheduler.step()

            with open('{}'.format(os.path.join(self.name, '_debug.pickle')), 'wb') as handle:
                pickle.dump(debug_log_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_poisoned_matrix(passive_matrix, need_poison, poison_grad, amplify_rate):
    poisoned_matrix = passive_matrix
    poisoned_matrix[need_poison] = poison_grad * amplify_rate
    return poisoned_matrix


def copy_grad(passive_matrix, need_copy):
    poison_grad = passive_matrix[need_copy]
    return poison_grad
