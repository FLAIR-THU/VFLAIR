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
import copy

from models.vision import resnet18, MLP2
from utils.basic_functions import cross_entropy_for_onehot, append_exp_res
# from evaluates.attacks.attack_api import apply_attack
from evaluates.defenses.defense_api import apply_defense
from evaluates.defenses.defense_functions import *
from utils.constants import *
import utils.constants as shared_var
from utils.marvell_functions import KL_gradient_perturb
from evaluates.attacks.attack_api import AttackerLoader

tf.compat.v1.enable_eager_execution() 


class MainTaskVFL(object):

    def __init__(self, args):
        self.args = args
        self.k = args.k
        self.device = args.device
        self.dataset_name = args.dataset
        # self.train_dataset = args.train_dst
        # self.val_dataset = args.test_dst
        # self.half_dim = args.half_dim
        self.epochs = args.main_epochs
        self.lr = args.main_lr
        self.batch_size = args.batch_size
        self.models_dict = args.model_list
        # self.num_classes = args.num_classes
        # self.num_class_list = args.num_class_list
        self.num_classes = args.num_classes
        self.exp_res_dir = args.exp_res_dir

        self.exp_res_path = args.exp_res_path
        self.parties = args.parties

        self.parties_data = None
        self.gt_one_hot_label = None
        self.pred_list = []
        self.pred_list_clone = []
        self.pred_gradients_list = []
        self.pred_gradients_list_clone = []
        self.loss = None
        self.train_acc = None

        # some state of VFL throughout training process
        self.first_epoch_state = None
        self.middle_epoch_state = None
        # self.final_epoch_state = None # <-- this is save in the above parameters

    def label_to_one_hot(self, target, num_classes=10):
        try:
            _ = target.size()[1]
            # print("use target itself", target.size())
            onehot_target = target.type(torch.float32).to(self.device)
        except:
            target = torch.unsqueeze(target, 1).to(self.device)
            # print("use unsqueezed target", target.size())
            onehot_target = torch.zeros(target.size(0), num_classes, device=self.device)
            onehot_target.scatter_(1, target, 1)
        return onehot_target

    def train_batch(self, parties_data, batch_label):
        # encoder = self.encoder
        # if self.apply_encoder:
        #     if encoder:
        #         _, gt_one_hot_label = encoder(batch_label)
        #     else:
        #         assert(encoder != None)
        # else:
        #     gt_one_hot_label = batch_label
        # print('current_label:', gt_one_hot_label)
        gt_one_hot_label = batch_label

        # ====== normal vertical federated learning ======

        # compute logits of clients
        self.pred_list = []
        self.pred_list_clone = []
        torch.autograd.set_detect_anomaly(True)
        for ik in range(self.k):
            self.parties[ik].obtain_local_data(parties_data[ik][0])
            self.pred_list.append(self.parties[ik].local_model(parties_data[ik][0]))
            # pred_list[ik] = torch.autograd.Variable(pred_list[ik], requires_grad=True).to(self.device)
            self.pred_list_clone.append(self.pred_list[ik].detach().clone())
            self.pred_list_clone[ik] = torch.autograd.Variable(self.pred_list_clone[ik], requires_grad=True).to(self.device)

        # ######################## aggregate logits of clients ########################
        pred, loss = self.parties[self.k-1].aggregate(self.pred_list_clone, gt_one_hot_label)
        # _gradients = torch.autograd.grad(loss, pred, retain_graph=True)
        # _gradients = torch.autograd.grad(loss, pred_list[ik], retain_graph=True)
        # pred = self.parties[self.k-1].global_model(pred_list)
        # loss = self.parties[self.k-1].criterion(pred, gt_one_hot_label)
        self.pred_gradients_list, self.pred_gradients_list_clone = self.parties[self.k-1].gradient_calculation(pred, self.pred_list_clone, loss)
        # pred, loss, pred_gradients_list, pred_gradients_list_clone = self.parties[self.k-1].aggregate_and_gradient_calculation(pred_list, gt_one_hot_label)
        # ######################## aggregate logits of clients ########################

        # defense applied on gradients
        if self.args.apply_defense == True and self.args.apply_mid == False:
            self.pred_gradients_list_clone = self.launch_defense(self.pred_gradients_list_clone, "gradients")        

        # print("gradients_clone[ik] have size:", pred_gradients_list_clone[0].size())
        # _g = torch.autograd.grad(pred_list[ik], self.parties[ik].local_model.parameters(), grad_outputs=torch.tensor([[1.]*10]*2048).to(self.device), retain_graph=True)
        # _g = torch.autograd.grad(pred_list[ik], self.parties[ik].local_model.parameters(), grad_outputs=pred_gradients_list_clone[ik], retain_graph=True)
        for ik in range(self.k):
            self.parties[ik].local_backward(self.pred_gradients_list_clone[ik], self.pred_list[ik])
        self.parties[self.k-1].global_backward(pred, loss)

        predict_prob = F.softmax(pred, dim=-1)
        suc_cnt = torch.sum(torch.argmax(predict_prob, dim=-1) == torch.argmax(batch_label, dim=-1)).item()
        train_acc = suc_cnt / predict_prob.shape[0]
        return loss.item(), train_acc

    def train(self):

        print_every = 1

        for ik in range(self.k):
            self.parties[ik].prepare_data_loader(batch_size=self.batch_size)

        test_acc = 0.0
        for i_epoch in range(self.epochs):
            # tqdm_train = tqdm(self.parties[self.k-1].train_loader, desc='Training (epoch #{})'.format(i_epoch + 1))
            postfix = {'train_loss': 0.0, 'train_acc': 0.0, 'test_acc': 0.0}
            i = -1
            data_loader_list = [self.parties[ik].train_loader for ik in range(self.k)]
            # data_loader_list.append(tqdm_train)
            # for parties_data in zip(self.parties[0].train_loader, self.parties[self.k-1].train_loader, tqdm_train): ## TODO: what to de for 4 party?
            for parties_data in zip(*data_loader_list):
                self.parties_data = parties_data
                i += 1
                for ik in range(self.k):
                    self.parties[ik].local_model.train()
                self.parties[self.k-1].global_model.train()

                # print("train", parties_data[0][0].size(),parties_data[self.k-1][0].size(),parties_data[self.k-1][1].size())
                self.gt_one_hot_label = self.label_to_one_hot(parties_data[self.k-1][1], self.num_classes)
                self.gt_one_hot_label = self.gt_one_hot_label.to(self.device)
                # print("parties' data have size:", parties_data[0][0].size(), parties_data[self.k-1][0].size(), parties_data[self.k-1][1].size())
                # ====== train batch ======
                self.loss, self.train_acc = self.train_batch(parties_data, self.gt_one_hot_label)
            
                if i == 0 and i_epoch == 0:
                    # self.launch_attack(self.pred_gradients_list_clone, self.pred_list_clone, "gradients_label")
                    self.first_epoch_state = self.save_state()
                elif i_epoch == self.epochs//2 and i == 0:
                    self.middle_epoch_state = self.save_state()

            # validation
            if (i + 1) % print_every == 0:
                print("validate and test")
                for ik in range(self.k):
                    self.parties[ik].local_model.eval()
                self.parties[self.k-1].global_model.eval()
                
                suc_cnt = 0
                sample_cnt = 0

                with torch.no_grad():
                    # enc_result_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
                    # result_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
                    data_loader_list = [self.parties[ik].test_loader for ik in range(self.k)]
                    # for parties_data in zip(self.parties[0].test_loader, self.parties[self.k-1].test_loader):
                    for parties_data in zip(*data_loader_list):
                        # print("test", parties_data[0][0].size(),parties_data[self.k-1][0].size(),parties_data[self.k-1][1].size())
                        gt_val_one_hot_label = self.label_to_one_hot(parties_data[self.k-1][1], self.num_classes)
                        gt_val_one_hot_label = gt_val_one_hot_label.to(self.device)

                        pred_list = []
                        for ik in range(self.k):
                            pred_list.append(self.parties[ik].local_model(parties_data[ik][0]))
                        test_logit, test_loss = self.parties[self.k-1].aggregate(pred_list, gt_val_one_hot_label)

                        enc_predict_prob = F.softmax(test_logit, dim=-1)
                        # if self.apply_encoder:
                        #     dec_predict_prob = self.encoder.decoder(enc_predict_prob)
                        #     predict_label = torch.argmax(dec_predict_prob, dim=-1)
                        # else:
                        #     predict_label = torch.argmax(enc_predict_prob, dim=-1)
                        predict_label = torch.argmax(enc_predict_prob, dim=-1)

                        # enc_predict_label = torch.argmax(enc_predict_prob, dim=-1)
                        actual_label = torch.argmax(gt_val_one_hot_label, dim=-1)
                        sample_cnt += predict_label.shape[0]
                        suc_cnt += torch.sum(predict_label == actual_label).item()
                    self.test_acc = suc_cnt / float(sample_cnt)
                    postfix['train_loss'] = self.loss
                    postfix['train_acc'] = '{:.2f}%'.format(self.train_acc * 100)
                    postfix['test_acc'] = '{:.2f}%'.format(self.test_acc * 100)
                    # tqdm_train.set_postfix(postfix)
                    print('Epoch {}% \t train_loss:{:.2f} train_acc:{:.2f} test_acc:{:.2f}'.format(
                        i_epoch, self.loss, self.train_acc, self.test_acc))
        
        # parameter = 'none'
        # if self.apply_laplace or self.apply_gaussian:
        #     parameter = str(self.dp_strength)
        # elif self.apply_grad_spar:
        #     parameter = str(self.grad_spars)
        # elif self.apply_encoder:
        #     parameter = str(self.ae_lambda)
        # elif self.apply_discrete_gradients:
        #     parameter = str(self.discrete_gradients_bins)
        # elif self.apply_marvell:
        #     parameter = str(self.marvell_s)

        # if self.apply_laplace or self.apply_gaussian or self.apply_grad_spar or self.apply_encoder or self.apply_marvell:
        #     exp_result = str(parameter) + ' ' + str(test_acc) + ' bs=' + str(self.batch_size) + '|num_class=' + str(self.num_classes)
        # else:
        #     exp_result = f"bs|num_class|epochs|recovery_rate,%d|%d|%d| %lf" % (self.batch_size, self.num_classes, self.self.epochs, test_acc)
        
        exp_result = f"bs|num_class|epochs|recovery_rate,%d|%d|%d| %lf" % (self.batch_size, self.num_classes, self.epochs, self.test_acc)
        append_exp_res(self.exp_res_path, exp_result)
        print(exp_result)
        
        return test_acc

    def save_state(self):
        return {
            "model": [copy.deepcopy(self.parties[ik].local_model) for ik in range(self.args.k)]+[self.parties[self.args.k-1].global_model],
            "data": copy.deepcopy(self.parties_data), 
            "label": copy.deepcopy(self.gt_one_hot_label),
            "predict": copy.deepcopy(self.pred_list_clone),
            "gradient": copy.deepcopy(self.pred_gradients_list_clone),
            "train_acc": copy.deepcopy(self.train_acc),
            "loss": copy.deepcopy(self.loss)
        }

    def evaluate_attack(self):
        self.attacker = AttackerLoader(self, self.args)
        if self.attacker != None:
            self.attacker.attack()

    def launch_attack(self, gradients_list, pred_list, type):
        if type == 'gradients_label':
            for ik in range(self.k):
                start_time = time.time()
                recovery_history = self.parties[ik].party_attack(self.args, gradients_list[ik], pred_list[ik])
                end_time = time.time()
                if recovery_history != None:
                    recovery_rate_history = []
                    for dummy_label in recovery_history:
                        rec_rate = self.calc_label_recovery_rate(dummy_label, self.gt_one_hot_label)
                        recovery_rate_history.append(rec_rate)
                        print(f'batch_size=%d,class_num=%d,party_index=%d,recovery_rate=%lf,time_used=%lf' % (dummy_label.size()[0], self.num_classes, ik, rec_rate, end_time - start_time))
                    best_rec_rate = max(recovery_rate_history)
                    exp_result = f"bs|num_class|attack_party_index|recovery_rate,%d|%d|%d|%lf|%s" % (dummy_label.size()[0], self.num_classes, ik, best_rec_rate, str(recovery_rate_history))
                    append_exp_res(self.parties[ik].attacker.exp_res_path, exp_result)
        else:
            # further extention
            pass

    def launch_defense(self, gradients_list, type):
        if type == 'gradients':
            return apply_defense(self.args, gradients_list)
        else:
            # further extention
            return gradients_list
            pass

    def calc_label_recovery_rate(self, dummy_label, gt_label):
        success = torch.sum(torch.argmax(dummy_label, dim=-1) == torch.argmax(gt_label, dim=-1)).item()
        total = dummy_label.shape[0]
        return success / total