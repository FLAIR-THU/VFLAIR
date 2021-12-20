import sys, os
sys.path.append(os.pardir)

# import logging
import pprint
import time

import tensorflow as tf
import torch

from models.vision import *
from utils.basic_functions import *
from utils.constants import *

import utils.constants as shared_var
from utils.marvell_functions import KL_gradient_perturb


tf.compat.v1.enable_eager_execution() 



class BatchLabelReconstruction(object):
    def __init__(self, args):
        '''
        :param args:  contains all the necessary parameters
        '''
        self.dataset = args.dataset
        self.model = args.model
        self.num_exp = args.num_exp
        self.epochs = args.epochs
        self.lr = args.lr
        self.early_stop = args.early_stop
        self.early_stop_param = args.early_stop_param
        self.device = args.device
        self.batch_size_list = args.batch_size_list
        self.num_class_list = args.num_class_list
        self.dst = args.dst
        self.exp_res_dir = args.exp_res_dir
        self.exp_res_path = args.exp_res_path
        self.net_a = args.net_a
        self.net_b = args.net_b
        self.gt_data_a = args.gt_data_a
        self.gt_data_b = args.gt_data_b
        self.gt_label = args.gt_label
        self.gt_onehot_label = torch.stack(args.gt_onehot_label).to(self.device)
        # defending parameters
        self.apply_trainable_layer = args.apply_trainable_layer
        self.apply_laplace = args.apply_laplace
        self.apply_gaussian = args.apply_gaussian
        self.dp_strength = args.dp_strength
        self.apply_grad_spar = args.apply_grad_spar
        self.grad_spars = args.grad_spars
        self.apply_encoder = args.apply_encoder
        self.apply_adversarial_encoder = args.apply_adversarial_encoder
        self.ae_lambda = args.ae_lambda
        self.encoder = args.encoder
        self.apply_marvell = args.apply_marvell
        self.marvell_s = args.marvell_s
        self.show_param()

    def show_param(self):
        print(f'********** config dict **********')
        pprint.pprint(self.__dict__)

    def calc_label_recovery_rate(self, dummy_label, gt_label):
        success = torch.sum(torch.argmax(dummy_label, dim=-1) == torch.argmax(gt_label, dim=-1)).item()
        total = dummy_label.shape[0]
        return success / total

    def train(self):
        '''
        execute the label inference algorithm
        :return: recovery rate
        '''

        print(f"Running on %s{torch.cuda.current_device()}" % self.device)
        # if self.dataset == 'nuswide':
        #     all_nuswide_labels = []
        #     for line in os.listdir('./data/NUS_WIDE/Groundtruth/AllLabels'):
        #         all_nuswide_labels.append(line.split('_')[1][:-4])
        for batch_size in self.batch_size_list:
            for num_classes in self.num_class_list:
        #         classes = [None] * num_classes
        #         if self.dataset == 'cifar100':
        #             # if apply the defense, we only use cifar20
        #             if self.apply_laplace or self.apply_gaussian or self.apply_grad_spar:
        #                 classes = [i for i in range(num_classes)]
        #             else:
        #                 # classes = random.sample(list(range(100)), num_classes)
        #                 classes = random.sample(list(range(10)), num_classes)

        #             all_data, all_label = get_class_i(self.dst, classes)
        #         elif self.dataset == 'mnist':
        #             classes = random.sample(list(range(10)), num_classes)
        #             all_data, all_label = get_class_i(self.dst, classes)
        #         elif self.dataset == 'nuswide':
        #             classes = random.sample(all_nuswide_labels, num_classes)
        #             x_image, x_text, Y = get_labeled_data('./data/NUS_WIDE', classes, None, 'Train')
        #         elif self.dataset == 'cifar10':
        #             classes = random.sample(list(range(10)), num_classes)
        #             all_data, all_label = get_class_i(self.dst, classes)

                recovery_rate_history = []
                for i_run in range(1, self.num_exp + 1):
                    start_time = time.time()
        #             # randomly sample
        #             if self.dataset == 'mnist' or self.dataset == 'cifar100' or self.dataset == 'cifar10':
        #                 gt_data = []
        #                 gt_label = []
        #                 for i in range(0, batch_size):
        #                     sample_idx = torch.randint(len(all_data), size=(1,)).item()
        #                     gt_data.append(all_data[sample_idx])
        #                     gt_label.append(all_label[sample_idx])
        #                 gt_data = torch.stack(gt_data).to(self.device)
        #                 half_size = list(gt_data.size())[-1] // 2
        #                 gt_data_a = gt_data[:, :, :half_size, :]
        #                 gt_data_b = gt_data[:, :, half_size:, :]
        #                 gt_label = torch.stack(gt_label).to(self.device)
        #                 gt_onehot_label = gt_label  # label_to_onehot(gt_label)
        #             elif self.dataset == 'nuswide':
        #                 gt_data_a, gt_data_b, gt_label = [], [], []
        #                 for i in range(0, batch_size):
        #                     sample_idx = torch.randint(len(x_image), size=(1,)).item()
        #                     gt_data_a.append(torch.tensor(x_text[sample_idx], dtype=torch.float32))
        #                     gt_data_b.append(torch.tensor(x_image[sample_idx], dtype=torch.float32))
        #                     gt_label.append(torch.tensor(Y[sample_idx], dtype=torch.float32))
        #                 gt_data_a = torch.stack(gt_data_a).to(self.device)
        #                 gt_data_b = torch.stack(gt_data_b).to(self.device)
        #                 gt_label = torch.stack(gt_label).to(self.device)
        #                 gt_onehot_label = gt_label  # label_to_onehot(gt_label)
        #             if self.apply_encoder:
        #                 _, gt_onehot_label = self.encoder(gt_onehot_label) # get the result given by AutoEncoder.forward
        #             if self.apply_adversarial_encoder:
        #                 _, gt_onehot_label = self.encoder(gt_data_a)
        #             # set model
        #             if self.model == 'MLP2':
        #                 self.net_a = MLP2(np.prod(list(gt_data_a.size())[1:]), num_classes).to(self.device)
        #                 self.net_b = MLP2(np.prod(list(gt_data_b.size())[1:]), num_classes).to(self.device)
        #             elif self.model == 'resnet18':
        #                 self.net_a = resnet18(num_classes).to(self.device)
        #                 self.net_b = resnet18(num_classes).to(self.device)
                    
                    # ......if args.apply_certify != 0 and epoch >= args.certify_start_epoch:
                    #     .....

                    criterion = cross_entropy_for_onehot
                    pred_a = self.net_a(self.gt_data_a)
                    pred_b = self.net_b(self.gt_data_b)
                    ######################## defense start ############################
                    ######################## defense1: trainable layer ############################
                    if self.apply_trainable_layer:
                        active_aggregate_model = ActivePartyWithTrainableLayer(input_dim=num_classes * 2, output_dim=num_classes)
                        dummy_active_aggregate_model = ActivePartyWithTrainableLayer(input_dim=num_classes * 2, output_dim=num_classes)
                    else:
                        active_aggregate_model = ActivePartyWithoutTrainableLayer()
                        dummy_active_aggregate_model = ActivePartyWithoutTrainableLayer()
                    pred = active_aggregate_model(pred_a, pred_b)
                    loss = criterion(pred, self.gt_onehot_label)
                    ######################## for defense2~4, calculate pred_a_gradients ############################
                    pred_a_gradients = torch.autograd.grad(loss, pred_a, retain_graph=True)
                    pred_a_gradients_clone = pred_a_gradients[0].detach().clone()
                    ######################## defense2: dp ############################
                    if self.apply_laplace and self.dp_strength != 0 or self.apply_gaussian and self.dp_strength != 0:
                        location = 0.0
                        threshold = 0.2  # 1e9
                        if self.apply_laplace:
                            with torch.no_grad():
                                scale = self.dp_strength
                                # clip 2-norm per sample
                                norm_factor_a = torch.div(torch.max(torch.norm(pred_a_gradients_clone, dim=1)),threshold + 1e-6).clamp(min=1.0)
                                # add laplace noise
                                dist_a = torch.distributions.laplace.Laplace(location, scale)
                                pred_a_gradients_clone = torch.div(pred_a_gradients_clone, norm_factor_a) + \
                                           dist_a.sample(pred_a_gradients_clone.shape).to(self.device)
                        elif self.apply_gaussian:
                            with torch.no_grad():
                                scale = self.dp_strength
                                norm_factor_a = torch.div(torch.max(torch.norm(pred_a_gradients_clone, dim=1)),
                                                           threshold + 1e-6).clamp(min=1.0)
                                pred_a_gradients_clone = torch.div(pred_a_gradients_clone, norm_factor_a) + \
                                                       torch.normal(location, scale, pred_a_gradients_clone.shape).to(self.device)
                    ######################## defense3: gradient sparsification ############################
                    elif self.apply_grad_spar and self.grad_spars != 0:
                        with torch.no_grad():
                            percent = self.grad_spars / 100.0
                            up_thr = torch.quantile(torch.abs(pred_a_gradients_clone), percent)
                            active_up_gradients_res = torch.where(
                                torch.abs(pred_a_gradients_clone).double() < up_thr.item(),
                                pred_a_gradients_clone.double(), float(0.)).to(self.device)
                            pred_a_gradients_clone = pred_a_gradients_clone - active_up_gradients_res
                    ######################## defense4: marvell ############################
                    elif self.apply_marvell and self.marvell_s != 0 and num_classes == 2:
                        # for marvell, change label to [0,1]
                        marvell_y = []
                        for i in range(len(self.gt_label)):
                            marvell_y.append(int(self.gt_label[i][1]))
                        marvell_y = np.array(marvell_y)
                        shared_var.batch_y = np.asarray(marvell_y)
                        logdir = 'marvell_logs/dlg_task/{}_logs/{}'.format(self.dataset, time.strftime("%Y%m%d-%H%M%S"))
                        writer = tf.summary.create_file_writer(logdir)
                        shared_var.writer = writer
                        with torch.no_grad():
                            pred_a_gradients_clone = KL_gradient_perturb(pred_a_gradients_clone, self.classes, self.marvell_s)
                            pred_a_gradients_clone = pred_a_gradients_clone.to(self.device)
                    original_dy_dx = torch.autograd.grad(pred_a, self.net_a.parameters(), grad_outputs=pred_a_gradients_clone)
                    ######################## defense end ############################

                    dummy_pred_b = torch.randn(pred_b.size()).to(self.device).requires_grad_(True)
                    dummy_label = torch.randn(self.gt_onehot_label.size()).to(self.device).requires_grad_(True)

                    if self.apply_trainable_layer:
                        optimizer = torch.optim.Adam([dummy_pred_b, dummy_label] + list(dummy_active_aggregate_model.parameters()), lr=self.lr)
                    else:
                        optimizer = torch.optim.Adam([dummy_pred_b, dummy_label], lr=self.lr)

                    for iters in range(1, self.epochs + 1):
                        def closure():
                            optimizer.zero_grad()
                            dummy_pred = dummy_active_aggregate_model(self.net_a(self.gt_data_a), dummy_pred_b)

                            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
                            dummy_loss = criterion(dummy_pred, dummy_onehot_label)
                            dummy_dy_dx_a = torch.autograd.grad(dummy_loss, self.net_a.parameters(), create_graph=True)
                            grad_diff = 0
                            for (gx, gy) in zip(dummy_dy_dx_a, original_dy_dx):
                                grad_diff += ((gx - gy) ** 2).sum()
                            grad_diff.backward()
                            return grad_diff

                        rec_rate = self.calc_label_recovery_rate(dummy_label, self.gt_label)
                        # if iters == 1:
                        #     append_exp_res(f'exp_result/{self.dataset}/exp_on_{self.dataset}_rec_rate_change.txt',
                        #                    f'{batch_size} 0 {rec_rate} {closure()}')
                        optimizer.step(closure)
                        
                        if self.early_stop == True:
                            if closure().item() < self.early_stop_param:
                                break

                    rec_rate = self.calc_label_recovery_rate(dummy_label, self.gt_label)
                    recovery_rate_history.append(rec_rate)
                    end_time = time.time()
                    # output the rec_info of this exp
                    if self.apply_laplace or self.apply_gaussian:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,dp_strength=%lf,recovery_rate=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.dp_strength,rec_rate, end_time - start_time))
                    elif self.apply_grad_spar:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,grad_spars=%lf,recovery_rate=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.grad_spars,rec_rate, end_time - start_time))
                    elif self.apply_marvell:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,marvel_s=%lf,recovery_rate=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.marvell_s,rec_rate, end_time - start_time))
                    else:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,recovery_rate=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, rec_rate, end_time - start_time))
                avg_rec_rate = np.mean(recovery_rate_history)
                if self.apply_laplace or self.apply_gaussian:
                    exp_result = str(self.dp_strength) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                elif self.apply_grad_spar:
                    exp_result = str(self.grad_spars) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                elif self.apply_encoder or self.apply_adversarial_encoder:
                    exp_result = str(self.ae_lambda) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                elif self.apply_marvell:
                    exp_result = str(self.marvell_s) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                else:
                    exp_result = f"bs|num_class|recovery_rate,%d|%d|%lf|%s|%lf" % (batch_size, num_classes, avg_rec_rate, str(recovery_rate_history), np.max(recovery_rate_history))

                append_exp_res(self.exp_res_path, exp_result)
                print(exp_result)

if __name__ == '__main__':
    pass