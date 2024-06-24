import sys, os

sys.path.append(os.pardir)

import torch
import torch.nn.functional as F
import time
import numpy as np
import copy
import pickle
import matplotlib.pyplot as plt
import itertools

from evaluates.attacks.attacker import Attacker
from models.global_models import *  # ClassificationModelHostHead, ClassificationModelHostTrainableHead
from utils.basic_functions import cross_entropy_for_onehot, append_exp_res

torch.backends.cudnn.enabled = False


def label_to_one_hot(target, num_classes=10):
    # print('label_to_one_hot:', target, type(target))
    try:
        _ = target.size()[1]
        # print("use target itself", target.size())
        onehot_target = target.type(torch.float32)
    except:
        target = torch.unsqueeze(target, 1)
        # print("use unsqueezed target", target.size())
        onehot_target = torch.zeros(target.size(0), num_classes)
        onehot_target.scatter_(1, target, 1)
    return onehot_target


class BatchLabelReconstruction(Attacker):
    def __init__(self, top_vfl, args):
        super().__init__(args)
        self.args = args
        # get information for launching BLI attack
        self.vfl_info = top_vfl.first_epoch_state
        # prepare parameters
        self.device = args.device
        self.num_classes = args.num_classes
        self.k = args.k
        self.party = args.attack_configs['party']  # parties that launch attacks
        self.lr = args.attack_configs['lr']
        self.epochs = args.attack_configs['epochs']
        self.early_stop = args.attack_configs['early_stop'] if 'early_stop' in args.attack_configs else 0
        self.early_stop_threshold = args.attack_configs[
            'early_stop_threshold'] if 'early_stop_threshold' in args.attack_configs else 1e-7
        self.label_size = args.num_classes
        self.dummy_active_top_trainable_model = None
        self.optimizer_trainable = None  # construct later
        self.dummy_active_top_non_trainable_model = None
        self.optimizer_non_trainable = None  # construct later
        self.criterion = cross_entropy_for_onehot
        self.file_name = 'attack_result.txt'
        self.exp_res_dir = f'exp_result/main/{args.dataset}/attack/BLR/'
        self.exp_res_path = ''

    def set_seed(self, seed=0):
        # random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def calc_label_recovery_rate(self, dummy_label, gt_label):
        success = torch.sum(torch.argmax(dummy_label, dim=-1) == torch.argmax(gt_label, dim=-1)).item()
        total = dummy_label.shape[0]
        return success / total

    def attack(self):
        self.set_seed(123)
        for ik in self.party:  # attacker party #ik
            index = ik
            # self.exp_res_dir = self.exp_res_dir + f'{index}/'
            # if not os.path.exists(self.exp_res_dir):
            #     os.makedirs(self.exp_res_dir)
            # self.exp_res_path = self.exp_res_dir + self.file_name

            # collect necessary information
            pred_list = self.vfl_info['predict']
            pred_a = self.vfl_info['predict'][ik]  # [copy.deepcopy(self.parties[ik].local_pred_clone) for ik in range(self.k)]
            pred_b = self.vfl_info['predict'][1]  # active party

            self_data = self.vfl_info['data'][ik][0]  # copy.deepcopy(self.parties_data)
            active_data = self.vfl_info['data'][1][0]  # Active party data

            local_gradient = self.vfl_info['gradient'][ik]
            original_dy_dx = self.vfl_info['local_model_gradient'][ik]  # gradient calculated for local model update

            net_a = self.vfl_info['model'][0].to(self.device)
            net_b = self.vfl_info['model'][1].to(self.device)
            global_model = self.vfl_info['global_model'].to(self.device)

            global_pred = self.vfl_info['global_pred'].to(self.device)
            pred_a = net_a(self_data).to(self.device).requires_grad_(True)
            pred_b = net_b(active_data).to(self.device).requires_grad_(True)  # real pred_b   fake:dummy pred_b
            # local_model = self.vfl_info['model'][ik]
            # local_model_copy = copy.deepcopy(local_model)
            # local_model = local_model.to(self.device)
            # local_model_copy.eval()
            global_model.eval()
            net_a.eval()

            true_label = self.vfl_info['label'].to(self.device)  # copy.deepcopy(self.gt_one_hot_label)

            # ################## debug: for checking if saved results are right (start) ##################
            print(f"sample_count = {pred_a.size()[0]}, number of classes = {pred_a.size()[1]}, {self.label_size}")
            # pickle.dump(self.vfl_info, open('./vfl_info.pkl','wb'))
            # original_dy = self.vfl_info['gradient'][ik]
            # new_pred_a = local_model_copy(self_data)
            # new_original_dy_dx = torch.autograd.grad(new_pred_a, local_model_copy.parameters(), grad_outputs=original_dy, retain_graph=True)
            # print(f"predict_error:{torch.nonzero(new_pred_a-pred_a)}")
            # for new_w, w in zip(new_original_dy_dx, original_dy_dx):
            #     print(f"model_weight_error:{torch.nonzero(new_w-w)}")
            # ################## debug: for checking if saved results are right (end) ##################

            sample_count = pred_a.size()[0]
            recovery_history = []
            recovery_rate_history = [[], []]

            for i in range(2):
                # set fake pred_b
                dummy_pred_b = torch.randn(pred_a.size()).to(self.device)
                dummy_label = torch.randn((sample_count, self.label_size)).to(self.device)
                dummy_pred_b.requires_grad = True
                dummy_label.requires_grad = True

                # Load Top Model
                if i == 0:
                    # ############# Problem real top model should be global_model?? #######
                    active_aggregate_model = global_model  # ClassificationModelHostHead()
                    dummy_active_aggregate_model = ClassificationModelHostHead()
                    # para = torch.tensor[]
                    # print(para.is_leaf)
                    optimizer = torch.optim.Adam([dummy_label, dummy_pred_b], lr=self.lr, betas=(0.9, 0.999), eps=1e-08,
                                                 weight_decay=0, amsgrad=False)
                else:
                    assert i == 1
                    active_aggregate_model = global_model  # ClassificationModelHostTrainableHead(self.k*self.num_classes, self.num_classes)
                    dummy_active_aggregate_model = ClassificationModelHostTrainableHead(self.k * self.num_classes,
                                                                                        self.num_classes)
                    optimizer = torch.optim.Adam(
                        itertools.chain([dummy_pred_b, dummy_label], list(dummy_active_aggregate_model.parameters())),
                        lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)  #
                active_aggregate_model = active_aggregate_model.to(self.device)  # real top model
                dummy_active_aggregate_model = dummy_active_aggregate_model.to(self.device)  # dummy top model

                # # In Active Party: calculate Real Pred and L
                # pred = active_aggregate_model([pred_a, pred_b]) # real pred
                # loss = self.criterion(pred, true_label) # real loss
                # pred_a_gradients = torch.autograd.grad(loss, pred_a, retain_graph=True, create_graph=True) # gradient given to a
                # pred_a_gradients_clone = pred_a_gradients[0].detach().clone()
                # # real L
                # original_dy_dx = torch.autograd.grad(pred_a, net_a.parameters(), grad_outputs=pred_a_gradients_clone,retain_graph=True,allow_unused=True)

                # === Begin Attack ===
                print(
                    f"BLI iteration for type{i}, self.device={self.device}, {dummy_pred_b.device}, {dummy_label.device}")
                start_time = time.time()

                for iters in range(1, self.epochs + 1):
                    # s_time = time.time()

                    def closure():
                        optimizer.zero_grad()

                        # fake pred/loss using fake top model/fake label
                        dummy_pred = dummy_active_aggregate_model([pred_a, dummy_pred_b])
                        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
                        dummy_loss = self.criterion(dummy_pred, dummy_onehot_label)
                        # dummy L
                        dummy_dy_dx_a = torch.autograd.grad(dummy_loss, net_a.parameters(), create_graph=True)

                        # loss: L-L'
                        grad_diff = 0
                        for (gx, gy) in zip(dummy_dy_dx_a, original_dy_dx):
                            grad_diff += ((gx - gy) ** 2).sum()
                        grad_diff.backward(retain_graph=True)

                        # if iters%200==0:
                        #     print('Iters',iters,' grad_diff:',grad_diff.item())
                        return grad_diff

                    # rec_rate = self.calc_label_recovery_rate(dummy_label, true_label)
                    # print(f"iter={iters}::rec_rate={rec_rate}")
                    optimizer.step(closure)
                    e_time = time.time()
                    # print(f"in BLR, i={i}, iter={iters}, time={s_time-e_time}")

                    if self.early_stop == 1:
                        if closure().item() < self.early_stop_threshold:
                            break

                    rec_rate = self.calc_label_recovery_rate(dummy_label, true_label)
                    # if iters%100==0:
                    #     print('Iters',iters,' rec_rate:',rec_rate)
                    recovery_rate_history[i].append(rec_rate)
                    end_time = time.time()

                ########## Clean #########
                del (dummy_active_aggregate_model)
                del (dummy_pred_b)
                del (dummy_label)

                print(f'batch_size=%d,class_num=%d,party_index=%d,recovery_rate=%lf,time_used=%lf' % (
                sample_count, self.label_size, index, rec_rate, end_time - start_time))

            final_rec_rate_trainable = recovery_rate_history[0][
                -1]  # sum(recovery_rate_history[0])/len(recovery_rate_history[0])
            final_rec_rate_non_trainable = recovery_rate_history[1][
                -1]  # sum(recovery_rate_history[1])/len(recovery_rate_history[1])
            best_rec_rate = max(final_rec_rate_trainable, final_rec_rate_non_trainable)

            print(f"BLI, if self.args.apply_defense={self.args.apply_defense}")
            # if self.args.apply_defense == True:
            #     exp_result = f"bs|num_class|attack_party_index|Q|top_trainable|acc,%d|%d|%d|%d|%d|%lf|%s (AttackConfig: %s) (Defense: %s %s)" % (sample_count, self.label_size, index, self.args.Q, self.args.apply_trainable_layer, best_rec_rate, str(self.args.attack_configs), self.args.defense_name, str(self.args.defense_configs))

            # else:
            #     exp_result = f"bs|num_class|attack_party_index|Q|top_trainable|acc,%d|%d|%d|%d|%d|%lf" % (sample_count, self.label_size, index, self.args.Q, self.args.apply_trainable_layer, best_rec_rate)# str(recovery_rate_history)

            # append_exp_res(self.exp_res_path, exp_result)

        # xx = [i for i in range(len(recovery_rate_history[0]))]
        # plt.figure()
        # plt.plot(xx,recovery_rate_history[0],'o-', color='b', alpha=0.8, linewidth=1, label='trainable')
        # plt.plot(xx,recovery_rate_history[1],'o-', color='r', alpha=0.8, linewidth=1, label='non-trainable')
        # plt.legend()
        # plt.savefig('./exp_result/BLI_Recovery_history.png')

        print("returning from BLI")
        return best_rec_rate
        # return recovery_history
