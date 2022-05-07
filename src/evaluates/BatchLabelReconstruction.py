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
from utils.defense_functions import *

import utils.constants as shared_var
from utils.marvell_functions import KL_gradient_perturb


tf.compat.v1.enable_eager_execution() 



class BatchLabelReconstruction(object):
    def __init__(self, args):
        '''
        :param args:  contains all the necessary parameters
        '''
        self.device = args.device
        self.dataset = args.dataset
        # self.model = args.model_list
        self.num_exp = args.num_exp
        self.epochs = args.epochs
        self.lr = args.lr
        self.early_stop = args.early_stop
        self.early_stop_param = args.early_stop_param
        self.batch_size_list = args.batch_size_list
        self.num_class_list = args.num_class_list
        self.dst = args.dst
        self.exp_res_dir = args.exp_res_dir
        self.exp_res_path = args.exp_res_path
        self.net_a = args.net_list[0]
        self.net_b = args.net_list[1]
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
        self.ae_lambda = args.ae_lambda
        self.encoder = args.encoder
        self.apply_marvell = args.apply_marvell
        self.marvell_s = args.marvell_s
        self.apply_discrete_gradients = args.apply_discrete_gradients
        self.discrete_gradients_bins = args.discrete_gradients_bins
        # self.discrete_gradients_bound = args.discrete_gradients_bound
        self.discrete_gradients_bound = 1e-3
        # self.show_param()

    def show_param(self):
        print(f'********** config dict **********')
        pprint.pprint(self.__dict__)

    def calc_label_recovery_rate(self, dummy_label, gt_label):
        success = torch.sum(torch.argmax(dummy_label, dim=-1) == torch.argmax(gt_label, dim=-1)).item()
        total = dummy_label.shape[0]
        return success / total

    def get_random_softmax_onehot_label(self, gt_onehot_label):
        _random = torch.randn(gt_onehot_label.size()).to(self.device)
        for i in range(len(gt_onehot_label)):
            max_index, = torch.where(_random[i] == _random[i].max())
            max_label, = torch.where(gt_onehot_label[i] == gt_onehot_label[i].max())
            while len(max_index) > 1:
                temp = torch.randn(gt_onehot_label[i].size()).to(self.device)
                # temp = torch.randn(gt_onehot_label[i].size())
                max_index, = torch.where(temp == temp.max())
                _random[i] = temp.clone()
            assert(len(max_label)==1)
            max_index = max_index.item()
            max_label = max_label.item()
            if max_index != max_label:
                temp = _random[i][int(max_index)].clone()
                _random[i][int(max_index)] = _random[i][int(max_label)].clone()
                _random[i][int(max_label)] = temp.clone()
            _random[i] = F.softmax(_random[i], dim=-1)
        return self.encoder(_random)

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

                recovery_rate_history = []
                for i_run in range(1, self.num_exp + 1):
                    start_time = time.time()

                    if self.apply_encoder:
                        _, self.gt_onehot_label = self.encoder(self.gt_onehot_label) # get the result given by AutoEncoder.forward
                        # if not self.apply_random_encoder:
                        #     _, self.gt_onehot_label = self.encoder(self.gt_onehot_label) # get the result given by AutoEncoder.forward
                        # else:
                        #     _, self.gt_onehot_label = self.get_random_softmax_onehot_label(self.gt_onehot_label)

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
                            print("use defense laplace in attacker")
                            with torch.no_grad():
                                scale = self.dp_strength
                                # clip 2-norm per sample
                                norm_factor_a = torch.div(torch.max(torch.norm(pred_a_gradients_clone, dim=1)),threshold + 1e-6).clamp(min=1.0)
                                # add laplace noise
                                dist_a = torch.distributions.laplace.Laplace(location, scale)
                                pred_a_gradients_clone = torch.div(pred_a_gradients_clone, norm_factor_a) + \
                                           dist_a.sample(pred_a_gradients_clone.shape).to(self.device)
                        elif self.apply_gaussian:
                            print("gaussian in attacker")
                            with torch.no_grad():
                                scale = self.dp_strength
                                norm_factor_a = torch.div(torch.max(torch.norm(pred_a_gradients_clone, dim=1)),
                                                           threshold + 1e-6).clamp(min=1.0)
                                pred_a_gradients_clone = torch.div(pred_a_gradients_clone, norm_factor_a) + \
                                                       torch.normal(location, scale, pred_a_gradients_clone.shape).to(self.device)
                    ######################## defense3: gradient sparsification ############################
                    elif self.apply_grad_spar and self.grad_spars != 0:
                        print("gradient sparcification in attacker")
                        with torch.no_grad():
                            percent = self.grad_spars / 100.0
                            up_thr = torch.quantile(torch.abs(pred_a_gradients_clone), percent)
                            active_up_gradients_res = torch.where(
                                torch.abs(pred_a_gradients_clone).double() < up_thr.item(),
                                pred_a_gradients_clone.double(), float(0.)).to(self.device)
                            pred_a_gradients_clone = pred_a_gradients_clone - active_up_gradients_res
                    ######################## defense4: marvell ############################
                    elif self.apply_marvell and self.marvell_s != 0:
                        assert num_classes == 2, "Marvell defense requires binary-classification"             
                        # for marvell, change label to [0,1]
                        print("marvell in attacker")
                        marvell_y = []
                        for i in range(len(self.gt_label)):
                            marvell_y.append(int(self.gt_label[i][1]))
                        marvell_y = np.array(marvell_y)
                        shared_var.batch_y = np.asarray(marvell_y)
                        # logdir = 'marvell_logs/dlg_task/{}_logs/{}'.format(self.dataset, time.strftime("%Y%m%d-%H%M%S"))
                        # writer = tf.summary.create_file_writer(logdir)
                        # shared_var.writer = writer
                        with torch.no_grad():
                            pred_a_gradients_clone = KL_gradient_perturb(pred_a_gradients_clone, self.classes, self.marvell_s)
                            pred_a_gradients_clone = pred_a_gradients_clone.to(self.device)
                    ######################## defense5: ppdl, GradientCompression, laplace_noise, DiscreteSGD ############################
                    # elif self.apply_ppdl:
                    #     dp_gc_ppdl(epsilon=1.8, sensitivity=1, layer_grad_list=[pred_a_gradients_clone], theta_u=self.ppdl_theta_u, gamma=0.001, tau=0.0001)
                    #     # dp_gc_ppdl(epsilon=1.8, sensitivity=1, layer_grad_list=[pred_b_gradients_clone], theta_u=self.ppdl_theta_u, gamma=0.001, tau=0.0001)
                    # elif self.apply_gc:
                    #     tensor_pruner = TensorPruner(zip_percent=self.gc_preserved_percent)
                    #     tensor_pruner.update_thresh_hold(pred_a_gradients_clone)
                    #     pred_a_gradients_clone = tensor_pruner.prune_tensor(pred_a_gradients_clone)
                    #     # tensor_pruner.update_thresh_hold(pred_b_gradients_clone)
                    #     # pred_b_gradients_clone = tensor_pruner.prune_tensor(pred_b_gradients_clone)
                    # elif self.apply_lap_noise:
                    #     dp = DPLaplacianNoiseApplyer(beta=self.noise_scale)
                    #     pred_a_gradients_clone = dp.laplace_mech(pred_a_gradients_clone)
                    #     # pred_b_gradients_clone = dp.laplace_mech(pred_b_gradients_clone)
                    elif self.apply_discrete_gradients:
                        pred_a_gradients_clone = multistep_gradient(pred_a_gradients_clone, bins_num=self.discrete_gradients_bins, bound_abs=self.discrete_gradients_bound)
                        # pred_b_gradients_clone = multistep_gradient(pred_b_gradients_clone, bins_num=self.discrete_gradients_bins, bound_abs=self.discrete_gradients_bound)
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
                elif self.apply_encoder:
                    exp_result = str(self.ae_lambda) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                elif self.apply_marvell:
                    exp_result = str(self.marvell_s) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                else:
                    exp_result = f"bs|num_class|recovery_rate,%d|%d|%lf|%s|%lf" % (batch_size, num_classes, avg_rec_rate, str(recovery_rate_history), np.max(recovery_rate_history))

                append_exp_res(self.exp_res_path, exp_result)
                print(exp_result)

if __name__ == '__main__':
    pass