import sys, os
sys.path.append(os.pardir)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tensorflow as tf

from tqdm import tqdm
# from utils import cross_entropy_for_one_hot, sharpen
import numpy as np
import time

from models.vision import resnet18, MLP2
from utils.basic_functions import cross_entropy_for_onehot, sharpen
from utils.defense_functions import *

from utils.constants import *
from utils.defense_functions import *

import utils.constants as shared_var
from utils.marvell_functions import KL_gradient_perturb

tf.compat.v1.enable_eager_execution() 


class MainTaskVFL(object):

    def __init__(self, args):
        self.device = args.device
        self.dataset_name = args.dataset
        self.train_dataset = args.train_dst
        self.val_dataset = args.test_dst
        self.half_dim = args.half_dim
        self.epochs = args.main_epochs
        self.lr = args.main_lr
        self.batch_size = args.batch_size
        self.models_dict = args.model_list
        # self.num_classes = args.num_classes
        # self.num_class_list = args.num_class_list
        self.num_classes = args.num_class_list[0]
        self.gradients_res_a = None
        self.gradients_res_b = None

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


    def fetch_parties_data(self, data):
        if self.dataset_name == 'nuswide':
            data_a = data[0]
            data_b = data[1]
        else:
            data_a = data[:, :, :self.half_dim, :]
            data_b = data[:, :, self.half_dim:, :]
        return data_a.to(self.device), data_b.to(self.device)

    def build_models(self, num_classes):
        # if self.dataset_name == 'cifar100' or self.dataset_name == 'cifar10':
        #     net_a = self.models_dict[self.dataset_name](num_classes).to(self.device)
        #     net_b = self.models_dict[self.dataset_name](num_classes).to(self.device)
        # elif self.dataset_name == 'mnist':
        #     net_a = self.models_dict[self.dataset_name](self.half_dim * self.half_dim * 2, num_classes).to(self.device)
        #     net_b = self.models_dict[self.dataset_name](self.half_dim * self.half_dim * 2, num_classes).to(self.device)
        # elif self.dataset_name == 'nuswide':
        #     net_a = self.models_dict[self.dataset_name](self.half_dim[0], num_classes).to(self.device)
        #     net_b = self.models_dict[self.dataset_name](self.half_dim[1], num_classes).to(self.device)
        # return net_a, net_b
        if self.dataset_name == 'cifar100' or self.dataset_name == 'cifar10':
            net_a = globals()[self.models_dict['0']['type']](num_classes).to(self.device)
            net_b = globals()[self.models_dict['0']['type']](num_classes).to(self.device)
        elif self.dataset_name == 'mnist':
            net_a = globals()[self.models_dict['0']['type']](self.half_dim * self.half_dim * 2, num_classes).to(self.device)
            net_b = globals()[self.models_dict['0']['type']](self.half_dim * self.half_dim * 2, num_classes).to(self.device)
        elif self.dataset_name == 'nuswide':
            net_a = globals()[self.models_dict['0']['type']](self.half_dim[0], num_classes).to(self.device)
            net_b = globals()[self.models_dict['0']['type']](self.half_dim[1], num_classes).to(self.device)
        return net_a, net_b

    def label_to_one_hot(self, target, num_classes=10):
        try:
            _ = target.size()[1]
            print("use target itself", target.size())
            onehot_target = target.type(torch.float32).to(self.device)
        except:
            target = torch.unsqueeze(target, 1).to(self.device)
            print("use unsqueezed target", target.size())
            onehot_target = torch.zeros(target.size(0), num_classes, device=self.device)
            onehot_target.scatter_(1, target, 1)
        return onehot_target

    def get_loader(self, dst, batch_size):
        # return torch.utils.data.DataLoader(dst, batch_size=batch_size, sampler=train_sampler, num_workers=4)
        return DataLoader(dst, batch_size=batch_size)

    def get_random_softmax_onehot_label(self, gt_onehot_label):
        _random = torch.randn(gt_onehot_label.size()).to(self.device)
        for i in range(len(gt_onehot_label)):
            # print("random[i] and onehot[i]:", _random[i], "|", gt_onehot_label[i])
            max_index, = torch.where(_random[i] == _random[i].max())
            # print("max_index:", max_index)
            max_label, = torch.where(gt_onehot_label[i] == gt_onehot_label[i].max())
            while len(max_index) > 1:
                temp = torch.randn(gt_onehot_label[i].size()).to(self.device)
                # temp = torch.randn(gt_onehot_label[i].size())
                # print("temp:", temp)
                max_index, = torch.where(temp == temp.max())
                # print("max_index:", max_index)
                _random[i] = temp.clone()
            assert(len(max_label)==1)
            # print("max_label:", max_label)
            max_index = max_index.item()
            max_label = max_label.item()
            # print(max_index, max_label)
            if max_index != max_label:
                temp = _random[i][int(max_index)].clone()
                _random[i][int(max_index)] = _random[i][int(max_label)].clone()
                _random[i][int(max_label)] = temp.clone()
            _random[i] = F.softmax(_random[i], dim=-1)
            # print("after softmax: _random[i]", _random[i])
        return self.encoder(_random)

    def train_batch(self, batch_data_a, batch_data_b, batch_label, net_a, net_b, encoder, model_optimizer, criterion):
        if self.apply_encoder:
            if encoder:
                _, gt_one_hot_label = encoder(batch_label)
            else:
                assert(encoder != None)
        else:
            gt_one_hot_label = batch_label
        # print('current_label:', gt_one_hot_label)

        # ====== normal vertical federated learning ======

        # compute logits of clients
        pred_a = net_a(batch_data_a)
        pred_b = net_b(batch_data_b)

        # aggregate logits of clients
        pred = pred_a + pred_b
        loss = criterion(pred, gt_one_hot_label)

        ######################## defense start ############################
        ######################## defense1: dp ############################
        pred_a_gradients = torch.autograd.grad(loss, pred_a, retain_graph=True)
        pred_a_gradients_clone = pred_a_gradients[0].detach().clone()
        pred_b_gradients = torch.autograd.grad(loss, pred_b, retain_graph=True)
        pred_b_gradients_clone = pred_b_gradients[0].detach().clone()
        if self.apply_laplace and self.dp_strength != 0.0 or self.apply_gaussian and self.dp_strength != 0.0:
            location = 0.0
            threshold = 0.2  # 1e9
            if self.apply_laplace:
                with torch.no_grad():
                    scale = self.dp_strength
                    # clip 2-norm per sample
                    print("norm of gradients:", torch.norm(pred_a_gradients_clone, dim=1), torch.max(torch.norm(pred_a_gradients_clone, dim=1)))
                    norm_factor_a = torch.div(torch.max(torch.norm(pred_a_gradients_clone, dim=1)),
                                              threshold + 1e-6).clamp(min=1.0)
                    # add laplace noise
                    dist_a = torch.distributions.laplace.Laplace(location, scale)
                    pred_a_gradients_clone = torch.div(pred_a_gradients_clone, norm_factor_a) + \
                                             dist_a.sample(pred_a_gradients_clone.shape).to(self.device)
                    print("norm of gradients after laplace:", torch.norm(pred_a_gradients_clone, dim=1), torch.max(torch.norm(pred_a_gradients_clone, dim=1)))
            elif self.apply_gaussian:
                with torch.no_grad():
                    scale = self.dp_strength

                    print("norm of gradients:", torch.norm(pred_a_gradients_clone, dim=1), torch.max(torch.norm(pred_a_gradients_clone, dim=1)))
                    norm_factor_a = torch.div(torch.max(torch.norm(pred_a_gradients_clone, dim=1)),
                                              threshold + 1e-6).clamp(min=1.0)
                    pred_a_gradients_clone = torch.div(pred_a_gradients_clone, norm_factor_a) + \
                                             torch.normal(location, scale, pred_a_gradients_clone.shape).to(self.device)
                    print("norm of gradients after gaussian:", torch.norm(pred_a_gradients_clone, dim=1), torch.max(torch.norm(pred_a_gradients_clone, dim=1)))
        ######################## defense2: gradient sparsification ############################
        elif self.apply_grad_spar:
            with torch.no_grad():
                percent = self.grad_spars / 100.0
                if self.gradients_res_a is not None and \
                        pred_a_gradients_clone.shape[0] == self.gradients_res_a.shape[0]:
                    pred_a_gradients_clone = pred_a_gradients_clone + self.gradients_res_a
                a_thr = torch.quantile(torch.abs(pred_a_gradients_clone), percent)
                self.gradients_res_a = torch.where(torch.abs(pred_a_gradients_clone).double() < a_thr.item(),
                                                      pred_a_gradients_clone.double(), float(0.)).to(self.device)
                pred_a_gradients_clone = pred_a_gradients_clone - self.gradients_res_a
        ######################## defense3: marvell ############################
        elif self.apply_marvell and self.marvell_s != 0 and self.num_classes == 2:
            # for marvell, change label to [0,1]
            marvell_y = []
            for i in range(len(gt_one_hot_label)):
                marvell_y.append(int(gt_one_hot_label[i][1]))
            marvell_y = np.array(marvell_y)
            shared_var.batch_y = np.asarray(marvell_y)
            logdir = 'marvell_logs/main_task/{}_logs/{}'.format(self.dataset_name, time.strftime("%Y%m%d-%H%M%S"))
            writer = tf.summary.create_file_writer(logdir)
            shared_var.writer = writer
            with torch.no_grad():
                pred_a_gradients_clone = KL_gradient_perturb(pred_a_gradients_clone, self.marvell_s)
                pred_a_gradients_clone = pred_a_gradients_clone.to(self.device)
        ######################## defense5: ppdl, GradientCompression, laplace_noise, DiscreteSGD ############################
        # elif self.apply_ppdl:
        #     dp_gc_ppdl(epsilon=1.8, sensitivity=1, layer_grad_list=[pred_a_gradients_clone], theta_u=self.ppdl_theta_u, gamma=0.001, tau=0.0001)
        #     dp_gc_ppdl(epsilon=1.8, sensitivity=1, layer_grad_list=[pred_b_gradients_clone], theta_u=self.ppdl_theta_u, gamma=0.001, tau=0.0001)
        # elif self.apply_gc:
        #     tensor_pruner = TensorPruner(zip_percent=self.gc_preserved_percent)
        #     tensor_pruner.update_thresh_hold(pred_a_gradients_clone)
        #     pred_a_gradients_clone = tensor_pruner.prune_tensor(pred_a_gradients_clone)
        #     tensor_pruner.update_thresh_hold(pred_b_gradients_clone)
        #     pred_b_gradients_clone = tensor_pruner.prune_tensor(pred_b_gradients_clone)
        # elif self.apply_lap_noise:
        #     dp = DPLaplacianNoiseApplyer(beta=self.noise_scale)
        #     pred_a_gradients_clone = dp.laplace_mech(pred_a_gradients_clone)
        #     pred_b_gradients_clone = dp.laplace_mech(pred_b_gradients_clone)
        elif self.apply_discrete_gradients:
            # print(pred_a_gradients_clone)
            pred_a_gradients_clone = multistep_gradient(pred_a_gradients_clone, bins_num=self.discrete_gradients_bins, bound_abs=self.discrete_gradients_bound)
            pred_b_gradients_clone = multistep_gradient(pred_b_gradients_clone, bins_num=self.discrete_gradients_bins, bound_abs=self.discrete_gradients_bound)
        ######################## defense end ############################
        model_optimizer.zero_grad()

        # update passive party(attacker) model
        weights_grad_a = torch.autograd.grad(pred_a, net_a.parameters(), grad_outputs=pred_a_gradients_clone)
        for w, g in zip(net_a.parameters(), weights_grad_a):
            if w.requires_grad:
                w.grad = g.detach()
        # update active party(defenser) model
        weights_grad_b = torch.autograd.grad(pred_b, net_b.parameters(), grad_outputs=pred_b_gradients_clone)
        for w, g in zip(net_b.parameters(), weights_grad_b):
            if w.requires_grad:
                w.grad = g.detach()
        # print("weights_grad_a,b:",weights_grad_a,weights_grad_b)
        model_optimizer.step()

        predict_prob = F.softmax(pred, dim=-1)
        suc_cnt = torch.sum(torch.argmax(predict_prob, dim=-1) == torch.argmax(gt_one_hot_label, dim=-1)).item()
        train_acc = suc_cnt / predict_prob.shape[0]
        return loss.item(), train_acc

    def train(self):

        train_loader = self.get_loader(self.train_dataset, batch_size=self.batch_size)
        val_loader = self.get_loader(self.val_dataset, batch_size=self.batch_size)

        # for num_classes in self.num_class_list:
        # n_minibatches = len(train_loader)
        if self.dataset_name == 'cifar100' or self.dataset_name == 'cifar10':
            print_every = 1
        elif self.dataset_name == 'mnist':
            print_every = 1
        elif self.dataset_name == 'nuswide':
            print_every = 1
        # net_a refers to passive model, net_b refers to active model
        net_a, net_b = self.build_models(self.num_classes)
        model_optimizer = torch.optim.Adam(list(net_a.parameters()) + list(net_b.parameters()),
                                           lr=self.lr)
        criterion = cross_entropy_for_onehot

        test_acc = 0.0
        for i_epoch in range(self.epochs):
            tqdm_train = tqdm(train_loader, desc='Training (epoch #{})'.format(i_epoch + 1))
            postfix = {'train_loss': 0.0, 'train_acc': 0.0, 'test_acc': 0.0}
            for i, (gt_data, gt_label) in enumerate(tqdm_train):
                net_a.train()
                net_b.train()
                gt_data_a, gt_data_b = self.fetch_parties_data(gt_data)
                gt_one_hot_label = self.label_to_one_hot(gt_label, self.num_classes)
                # gt_one_hot_label = gt_label.to(self.device)
                gt_one_hot_label = gt_one_hot_label.to(self.device)
                # print('before batch, gt_one_hot_label:', gt_one_hot_label)
                # ====== train batch ======
                loss, train_acc = self.train_batch(gt_data_a, gt_data_b, gt_one_hot_label,
                                              net_a, net_b, self.encoder, model_optimizer, criterion)
                # validation
                if (i + 1) % print_every == 0:
                    # print("validate and test")
                    net_a.eval()
                    net_b.eval()
                    suc_cnt = 0
                    sample_cnt = 0

                    with torch.no_grad():
                        # enc_result_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
                        # result_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
                        for gt_val_data, gt_val_label in val_loader:
                            gt_val_one_hot_label = self.label_to_one_hot(gt_val_label, self.num_classes)
                            # gt_val_one_hot_label = gt_val_label.to(self.device)
                            test_data_a, test_data_b = self.fetch_parties_data(gt_val_data)

                            test_logit_a = net_a(test_data_a)
                            test_logit_b = net_b(test_data_b)
                            test_logit = test_logit_a + test_logit_b

                            enc_predict_prob = F.softmax(test_logit, dim=-1)
                            if self.apply_encoder:
                                dec_predict_prob = self.encoder.decoder(enc_predict_prob)
                                predict_label = torch.argmax(dec_predict_prob, dim=-1)
                            else:
                                predict_label = torch.argmax(enc_predict_prob, dim=-1)

                            # enc_predict_label = torch.argmax(enc_predict_prob, dim=-1)
                            actual_label = torch.argmax(gt_val_one_hot_label, dim=-1)
                            sample_cnt += predict_label.shape[0]
                            suc_cnt += torch.sum(predict_label == actual_label).item()
                        test_acc = suc_cnt / float(sample_cnt)
                        postfix['train_loss'] = loss
                        postfix['train_acc'] = '{:.2f}%'.format(train_acc * 100)
                        postfix['test_acc'] = '{:.2f}%'.format(test_acc * 100)
                        tqdm_train.set_postfix(postfix)
                        print('Epoch {}% \t train_loss:{:.2f} train_acc:{:.2f} test_acc:{:.2f}'.format(
                            i_epoch, loss, train_acc, test_acc))
        
        parameter = 'none'
        if self.apply_laplace or self.apply_gaussian:
            parameter = str(self.dp_strength)
        elif self.apply_grad_spar:
            parameter = str(self.grad_spars)
        elif self.apply_encoder:
            parameter = str(self.ae_lambda)
        elif self.apply_discrete_gradients:
            parameter = str(self.discrete_gradients_bins)
        elif self.apply_marvell:
            parameter = str(self.marvell_s)
        return test_acc, parameter
