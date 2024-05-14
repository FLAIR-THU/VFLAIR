import sys, os

sys.path.append(os.pardir)

import torch
import torch.nn.functional as F
from torch import autograd
import time
import numpy as np
import copy
import matplotlib.pyplot as plt
import random

from evaluates.attacks.attacker import Attacker
from models.global_models import *
from models.mlp import MLP2_scalable
from load.LoadModels import load_basic_models
from utils.basic_functions import cross_entropy_for_onehot, append_exp_res, label_to_one_hot
from dataset.party_dataset import PassiveDataset
from dataset.party_dataset import ActiveDataset
from torch.utils.data import DataLoader

import sys, os

sys.path.append(os.pardir)

import torch
import torch.nn.functional as F
import torch.nn.init as init
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import time
import numpy as np
import copy
import pickle
import matplotlib.pyplot as plt
import itertools

from evaluates.attacks.attacker import Attacker
from models.global_models import *
from utils.basic_functions import cross_entropy_for_onehot, append_exp_res, label_to_one_hot, roc_auc_score, \
    multiclass_auc
from utils.pmc_functions import precision_recall, interleave_offsets, interleave, BottomModelPlus, SemiLoss, WeightEMA, \
    AverageMeter, InferenceHead, accuracy
from dataset.party_dataset import ActiveDataset


class AttributeInference(Attacker):
    def __init__(self, top_vfl, args):
        super().__init__(args)
        self.args = args
        # get information for launching BLI attack
        self.vfl_info = top_vfl.final_state
        # prepare parameters
        self.device = args.device
        self.num_classes = args.num_classes
        self.num_attributes = -1
        self.k = args.k  # party number
        self.party = args.attack_configs['party']  # parties that launch attacks

        self.lr = args.attack_configs['lr']
        self.epochs = args.attack_configs['epochs']
        self.label_size = args.num_classes
        self.batch_size = args.attack_configs['batch_size']

        self.criterion = cross_entropy_for_onehot
        self.l2_loss = nn.MSELoss()

        # self.file_name = 'attack_result.txt'
        # self.exp_res_dir = f'exp_result/main/{args.dataset}/attack/PMC/'
        # self.exp_res_path = ''

    def set_seed(self, seed=0):
        # random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def calc_attribute_recovery_rate(self, dummy_label, gt_label):
        success = torch.sum(torch.argmax(dummy_label, dim=-1) == torch.argmax(gt_label, dim=-1)).item()
        total = dummy_label.shape[0]
        return success / total

    def train_auxiliary_model_for_attribute(self, train_data_loader_list, test_data_loader_list, victim_party_index,
                                            attacker_index, ):
        # print(f"[debug] training auxiliary model on auxiliary data from total data to attribute with model trained from scratch, victim={victim_party_index}, attacker={attacker_index}")
        assert attacker_index == self.k - 1, "Attack of attribute inference should be launched at the active party side"
        num_included_parties = 2  # [vinctim_party, attacker]
        # for loader in data_loader_list:
        #     print(f"[debug] data loader in list {loader}")

        model_list = []
        optimizer_list = []
        for party_index in [victim_party_index, attacker_index]:
            _, local_model, local_optimizer, _, _ = load_basic_models(self.args, party_index)
            local_optimizer.lr = self.lr  # force the learning rate to be the one for attack
            model_list.append(local_model)
            optimizer_list.append(local_optimizer)
            # if global_model != None and global_optimizer != None:
            #     model_list.append(global_model)
            #     optimizer_list.append(global_optimizer)
        global_model = globals()[self.args.global_model](self.z_dim, self.num_attributes)
        global_model = global_model.to(self.args.device)
        global_optimizer = torch.optim.Adam(list(global_model.parameters()), lr=self.lr)
        model_list.append(global_model)
        optimizer_list.append(global_optimizer)
        assert len(model_list) == len(optimizer_list) == (
                    num_included_parties + 1), f"[Error] There should be 3 models for Attribute Inference Attack, but now have {len(model_list)}"

        for i_epoch in range(self.epochs):
            # print('\nAuxiliary Model Training Epoch: [%d | %d]' % (i_epoch + 1, self.epochs))
            train_acc = 0.
            test_acc = 0.
            # train
            for model in model_list:
                model.train()
            epoch_suc_cnt = 0
            epoch_total_sample_cnt = 0
            train_epoch_total_loss = 0.
            for parties_data in zip(*train_data_loader_list):
                gt_one_hot_label = parties_data[-1][1]
                # print("in training auxiliary model", gt_one_hot_label)
                pred_list = []
                pred_list_clone = []
                for ik in range(num_included_parties):
                    _local_pred = model_list[ik](parties_data[ik][0])
                    pred_list.append(_local_pred)
                    pred_list_clone.append(_local_pred.detach().clone())
                    pred_list_clone[ik] = torch.autograd.Variable(pred_list_clone[ik], requires_grad=True).to(
                        self.device)
                # ################## debug ##################
                pred_list[0] = torch.zeros(pred_list[0].shape).to(self.args.device)
                # ################## debug ##################
                pred = model_list[-1](pred_list)
                loss = self.criterion(pred, gt_one_hot_label)

                for i_optimizer in range(len(optimizer_list)):
                    optimizer_list[i_optimizer].zero_grad()
                # update local model
                pred_gradients_list = []
                pred_gradients_list_clone = []
                for ik in range(num_included_parties):
                    # ################## debug ##################
                    if ik == 0:
                        pred_gradients_list.append(torch.zeros((1,)).to(self.args.device))
                        pred_gradients_list_clone.append(torch.zeros((1,)).to(self.args.device))
                        continue
                    # ################## debug ##################
                    pred_gradients_list.append(
                        torch.autograd.grad(loss, pred_list[ik], retain_graph=True, create_graph=True))
                    pred_gradients_list_clone.append(pred_gradients_list[ik][0].detach().clone())
                    weights_grad_a = torch.autograd.grad(pred_list[ik], model_list[ik].parameters(),
                                                         grad_outputs=pred_gradients_list_clone[ik], retain_graph=True)
                    for w, g in zip(model_list[ik].parameters(), weights_grad_a):
                        if w.requires_grad:
                            w.grad = g.detach()
                    optimizer_list[ik].step()
                # update global model
                _gradients = torch.autograd.grad(loss, pred, retain_graph=True)
                _gradients_clone = _gradients[0].detach().clone()
                weights_grad_a = torch.autograd.grad(pred, model_list[-1].parameters(), grad_outputs=_gradients_clone,
                                                     retain_graph=True)
                for w, g in zip(model_list[-1].parameters(), weights_grad_a):
                    if w.requires_grad:
                        w.grad = g.detach()
                optimizer_list[-1].step()

                predict_prob = F.softmax(pred, dim=-1)
                suc_cnt = torch.sum(torch.argmax(predict_prob, dim=-1) == torch.argmax(gt_one_hot_label, dim=-1)).item()
                train_acc = suc_cnt / predict_prob.shape[0]
                epoch_suc_cnt += suc_cnt
                epoch_total_sample_cnt += predict_prob.shape[0]
                train_epoch_total_loss += loss * predict_prob.shape[0]
            train_acc = epoch_suc_cnt / epoch_total_sample_cnt
            train_epoch_total_loss = train_epoch_total_loss / epoch_total_sample_cnt

            # test
            for model in model_list:
                model.eval()
            epoch_suc_cnt = 0
            epoch_total_sample_cnt = 0
            test_preds = []
            test_targets = []
            for parties_data in zip(*test_data_loader_list):
                gt_one_hot_label = parties_data[-1][1]
                pred_list = []
                for ik in range(num_included_parties):
                    _local_pred = model_list[ik](parties_data[ik][0])
                    pred_list.append(_local_pred)
                pred = model_list[-1](pred_list)

                test_preds.append(list(pred.detach().cpu().numpy()))
                test_targets.append(list(gt_one_hot_label.detach().cpu().numpy()))

                predict_prob = F.softmax(pred, dim=-1)
                suc_cnt = torch.sum(torch.argmax(predict_prob, dim=-1) == torch.argmax(gt_one_hot_label, dim=-1)).item()
                train_acc = suc_cnt / predict_prob.shape[0]
                epoch_suc_cnt += suc_cnt
                epoch_total_sample_cnt += predict_prob.shape[0]
            test_acc = epoch_suc_cnt / epoch_total_sample_cnt
            test_preds = np.vstack(test_preds)
            test_targets = np.vstack(test_targets)
            test_auc = np.mean(multiclass_auc(test_targets, test_preds))
            # print(f"Auxiliary Model Training, Epoch {i_epoch+1}/{self.epochs}, attribute inference on aux_data only, train_loss={train_epoch_total_loss}, train_acc={train_acc}, test_acc={test_acc}")
            print(
                f"Auxiliary Model Training, Epoch {i_epoch + 1}/{self.epochs}, attribute inference on aux_data only, train_loss={train_epoch_total_loss}, train_acc={train_acc}, test_acc={test_acc}. test_auc={test_auc}")

        return model_list

    def train_mapper_attacker(self, train_data_loader_list, test_data_loader_list, original_model_list, aux_model_list,
                              model_T, model_attack):
        for model in original_model_list:
            model.eval()
        for model in aux_model_list:
            model.eval()
        optimizer_T = torch.optim.Adam(list(model_T.parameters()), lr=self.lr)
        optimizer_attack = torch.optim.Adam(list(model_attack.parameters()), lr=self.lr)
        model_T.train()
        model_attack.train()

        num_included_parties = 2  # [vinctim_party, attacker]

        ############################## v1: train model_T and model_attack simutaneously ##############################
        best_acc = 0.0
        for i_epoch in range(self.epochs):
            print('\nMapping and Attack Model Training Epoch: [%d | %d]' % (i_epoch + 1, self.epochs))
            train_acc = 0.0
            test_acc = 0.0

            # training within the epoch
            epoch_suc_cnt = 0
            epoch_total_sample_cnt = 0
            train_epoch_l2_loss = 0.
            train_epoch_ce_loss = 0.
            model_T.train()
            model_attack.train()
            for parties_data in zip(*train_data_loader_list):
                gt_one_hot_label = parties_data[-1][1]

                # get original pred (z), z.shape=(batch_size, num_classes*2)
                pred_list = []
                pred_list_clone = []
                for ik in range(num_included_parties):
                    _local_pred = original_model_list[ik](parties_data[ik][0])
                    pred_list.append(_local_pred)
                    pred_list_clone.append(_local_pred.detach().clone())
                    pred_list_clone[ik] = torch.autograd.Variable(pred_list_clone[ik], requires_grad=True).to(
                        self.device)
                # ################## debug ##################
                pred_list[0] = torch.zeros(pred_list[0].shape).to(self.args.device)
                # ################## debug ##################
                # pred = original_model_list[-1](pred_list)
                z = torch.cat(pred_list, dim=1)
                # print(f"[debug] z has shape: {z.shape}")

                # get aux pred (z_aux), z_aux.shape=(batch_size, num_attributes*2)
                aux_pred_list = []
                aux_pred_list_clone = []
                for ik in range(num_included_parties):
                    _local_pred = aux_model_list[ik](parties_data[ik][0])
                    aux_pred_list.append(_local_pred)
                    aux_pred_list_clone.append(_local_pred.detach().clone())
                    aux_pred_list_clone[ik] = torch.autograd.Variable(aux_pred_list_clone[ik], requires_grad=True).to(
                        self.device)
                # ################## debug ##################
                aux_pred_list[0] = torch.zeros(aux_pred_list[0].shape).to(self.args.device)
                # ################## debug ##################
                # aux_pred = aux_model_list[-1](aux_pred_list)
                aux_z = torch.cat(aux_pred_list, dim=1)
                # print(f"[debug] aux_z has shape: {aux_z.shape}")

                # update model_T with ||model_T(z)-aux_z||_2^2
                optimizer_T.zero_grad()
                transfered_z = model_T(z)
                # print(f"[debug] transfered_z has shape: {transfered_z.shape}")
                mapping_l2_loss = self.l2_loss(transfered_z, aux_z)
                mapping_l2_loss.backward(retain_graph=True)
                optimizer_T.step()

                # update model_attack with T(z) and attribute(label here)
                optimizer_attack.zero_grad()
                # attribute_pred = model_attack(model_T(z))
                # ################## debug ##################
                attribute_pred = model_attack(z)
                # ################## debug ##################
                # print(f"[debug] attribute_pred has shape {attribute_pred.shape}")
                loss = self.criterion(attribute_pred, gt_one_hot_label)
                loss.backward(retain_graph=True)
                optimizer_attack.step()

                predict_prob = F.softmax(attribute_pred, dim=-1)
                suc_cnt = torch.sum(torch.argmax(predict_prob, dim=-1) == torch.argmax(gt_one_hot_label, dim=-1)).item()
                # train_acc = suc_cnt / predict_prob.shape[0]
                epoch_suc_cnt += suc_cnt
                epoch_total_sample_cnt += predict_prob.shape[0]
                train_epoch_l2_loss += mapping_l2_loss.item() * predict_prob.shape[0]
                train_epoch_ce_loss += loss.item() * predict_prob.shape[0]
            train_acc = epoch_suc_cnt / epoch_total_sample_cnt
            train_epoch_l2_loss = train_epoch_l2_loss / epoch_total_sample_cnt
            train_epoch_ce_loss = train_epoch_ce_loss / epoch_total_sample_cnt

            # testing within the epoch
            epoch_suc_cnt = 0
            epoch_total_sample_cnt = 0
            test_attribute_pred_list = []
            test_one_hot_attribute_list = []
            model_T.eval()
            model_attack.eval()
            for parties_data in zip(*test_data_loader_list):
                gt_one_hot_label = parties_data[-1][1]

                # get original pred (z), z.shape=(batch_size, num_classes*2)
                pred_list = []
                pred_list_clone = []
                for ik in range(num_included_parties):
                    _local_pred = original_model_list[ik](parties_data[ik][0])
                    pred_list.append(_local_pred)
                    pred_list_clone.append(_local_pred.detach().clone())
                    pred_list_clone[ik] = torch.autograd.Variable(pred_list_clone[ik], requires_grad=True).to(
                        self.device)
                # ################## debug ##################
                aux_pred_list[0] = torch.zeros(aux_pred_list[0].shape).to(self.args.device)
                # ################## debug ##################
                # pred = original_model_list[-1](pred_list)
                z = torch.cat(pred_list, dim=1)
                # print(f"[debug] z has shape: {z.shape}")

                # # get aux pred (z_aux), z_aux.shape=(batch_size, num_attributes*2)
                # aux_pred_list = []
                # aux_pred_list_clone = []
                # for ik in range(num_included_parties):
                #     _local_pred = original_model_list[ik](parties_data[ik][0])
                #     aux_pred_list.append(_local_pred)
                #     aux_pred_list_clone.append(_local_pred.detach().clone())
                #     aux_pred_list_clone[ik] = torch.autograd.Variable(aux_pred_list_clone[ik], requires_grad=True).to(self.device)
                # # aux_pred = original_model_list[-1](aux_pred_list)
                # aux_z = torch.cat(aux_pred_list, dim=1)
                # print(f"[debug] aux_z has shape: {aux_z.shape}")

                # attribute_pred = model_attack(model_T(z))
                # ################## debug ##################
                attribute_pred = model_attack(z)
                # ################## debug ##################
                test_attribute_pred_list.append(list(attribute_pred.detach().cpu().numpy()))
                test_one_hot_attribute_list.append(list(gt_one_hot_label.detach().cpu().numpy()))

                predict_prob = F.softmax(attribute_pred, dim=-1)
                suc_cnt = torch.sum(torch.argmax(predict_prob, dim=-1) == torch.argmax(gt_one_hot_label, dim=-1)).item()
                # train_acc = suc_cnt / predict_prob.shape[0]
                epoch_suc_cnt += suc_cnt
                epoch_total_sample_cnt += predict_prob.shape[0]
            test_acc = epoch_suc_cnt / epoch_total_sample_cnt

            test_attribute_pred_list = np.vstack(test_attribute_pred_list)
            test_one_hot_attribute_list = np.vstack(test_one_hot_attribute_list)
            test_auc = np.mean(multiclass_auc(test_one_hot_attribute_list, test_attribute_pred_list))
            best_acc = max(test_acc, best_acc)
            # print(f"Mapping and Attack Model Training Epoch {i_epoch+1}/{self.epochs}, attribute inference with attack model, train_l2_loss={train_epoch_l2_loss}, train_ce_loss={train_epoch_ce_loss}, train_acc={train_acc}, test_acc={test_acc}")
            print(
                f"Mapping and Attack Model Training Epoch {i_epoch + 1}/{self.epochs}, attribute inference with attack model, train_l2_loss={train_epoch_l2_loss}, train_ce_loss={train_epoch_ce_loss}, train_acc={train_acc}, test_acc={test_acc}, test_auc={test_auc}")
            print(f"best_acc={best_acc}")
        ############################## v1: train model_T and model_attack simutaneously ##############################

        # ############################## v2: train model_T and model_attack separately ##############################
        # for i_epoch in range(self.epochs):
        #     print('\nMapping and Attack Model Training Epoch: [%d | %d]' % (i_epoch + 1, self.epochs))
        #     train_acc = 0.0
        #     test_acc = 0.0

        #     # training within the epoch
        #     epoch_suc_cnt = 0
        #     epoch_total_sample_cnt = 0
        #     model_T.train()
        #     for parties_data in zip(*train_data_loader_list):
        #         gt_one_hot_label = parties_data[-1][1]

        #         # get original pred (z), z.shape=(batch_size, num_classes*2)
        #         pred_list = []
        #         pred_list_clone = []
        #         for ik in range(num_included_parties):
        #             _local_pred = original_model_list[ik](parties_data[ik][0])
        #             pred_list.append(_local_pred)
        #             pred_list_clone.append(_local_pred.detach().clone())
        #             pred_list_clone[ik] = torch.autograd.Variable(pred_list_clone[ik], requires_grad=True).to(self.device)
        #         # pred = original_model_list[-1](pred_list)
        #         z = torch.cat(pred_list, dim=1)
        #         # print(f"[debug] z has shape: {z.shape}")

        #         # get aux pred (z_aux), z_aux.shape=(batch_size, num_attributes*2)
        #         aux_pred_list = []
        #         aux_pred_list_clone = []
        #         for ik in range(num_included_parties):
        #             _local_pred = aux_model_list[ik](parties_data[ik][0])
        #             aux_pred_list.append(_local_pred)
        #             aux_pred_list_clone.append(_local_pred.detach().clone())
        #             aux_pred_list_clone[ik] = torch.autograd.Variable(aux_pred_list_clone[ik], requires_grad=True).to(self.device)
        #         # aux_pred = aux_model_list[-1](aux_pred_list)
        #         aux_z = torch.cat(aux_pred_list, dim=1)
        #         # print(f"[debug] aux_z has shape: {aux_z.shape}")

        #         # update model_T with ||model_T(z)-aux_z||_2^2
        #         optimizer_T.zero_grad()
        #         transfered_z = model_T(z)
        #         # print(f"[debug] transfered_z has shape: {transfered_z.shape}")
        #         mapping_l2_loss = self.l2_loss(transfered_z, aux_z)
        #         mapping_l2_loss.backward(retain_graph=True)
        #         optimizer_T.step()

        # model_T.eval()
        # for i_epoch in range(self.epochs):
        #     print('\nMapping and Attack Model Training Epoch: [%d | %d]' % (i_epoch + 1, self.epochs))
        #     train_acc = 0.0
        #     test_acc = 0.0

        #     # training within the epoch
        #     epoch_suc_cnt = 0
        #     epoch_total_sample_cnt = 0
        #     model_attack.train()
        #     for parties_data in zip(*train_data_loader_list):
        #         gt_one_hot_label = parties_data[-1][1]

        #         # get original pred (z), z.shape=(batch_size, num_classes*2)
        #         pred_list = []
        #         pred_list_clone = []
        #         for ik in range(num_included_parties):
        #             _local_pred = original_model_list[ik](parties_data[ik][0])
        #             pred_list.append(_local_pred)
        #             pred_list_clone.append(_local_pred.detach().clone())
        #             pred_list_clone[ik] = torch.autograd.Variable(pred_list_clone[ik], requires_grad=True).to(self.device)
        #         # pred = original_model_list[-1](pred_list)
        #         z = torch.cat(pred_list, dim=1)
        #         # print(f"[debug] z has shape: {z.shape}")

        #         # update model_attack with T(z) and attribute(label here)
        #         optimizer_attack.zero_grad()
        #         attribute_pred = model_attack(model_T(z))
        #         # print(f"[debug] attribute_pred has shape {attribute_pred.shape}")
        #         loss = self.criterion(attribute_pred, gt_one_hot_label)
        #         loss.backward(retain_graph=True)
        #         optimizer_attack.step()

        #         predict_prob = F.softmax(attribute_pred, dim=-1)
        #         suc_cnt = torch.sum(torch.argmax(predict_prob, dim=-1) == torch.argmax(gt_one_hot_label, dim=-1)).item()
        #         # train_acc = suc_cnt / predict_prob.shape[0]
        #         epoch_suc_cnt += suc_cnt
        #         epoch_total_sample_cnt += predict_prob.shape[0]
        #     train_acc = epoch_suc_cnt / epoch_total_sample_cnt

        #     # testing within the epoch
        #     epoch_suc_cnt = 0
        #     epoch_total_sample_cnt = 0
        #     model_attack.eval()
        #     for parties_data in zip(*test_data_loader_list):
        #         gt_one_hot_label = parties_data[-1][1]

        #         # get original pred (z), z.shape=(batch_size, num_classes*2)
        #         pred_list = []
        #         pred_list_clone = []
        #         for ik in range(num_included_parties):
        #             _local_pred = original_model_list[ik](parties_data[ik][0])
        #             pred_list.append(_local_pred)
        #             pred_list_clone.append(_local_pred.detach().clone())
        #             pred_list_clone[ik] = torch.autograd.Variable(pred_list_clone[ik], requires_grad=True).to(self.device)
        #         # pred = original_model_list[-1](pred_list)
        #         z = torch.cat(pred_list, dim=1)
        #         # print(f"[debug] z has shape: {z.shape}")

        #         attribute_pred = model_attack(model_T(z))

        #         predict_prob = F.softmax(attribute_pred, dim=-1)
        #         suc_cnt = torch.sum(torch.argmax(predict_prob, dim=-1) == torch.argmax(gt_one_hot_label, dim=-1)).item()
        #         # train_acc = suc_cnt / predict_prob.shape[0]
        #         epoch_suc_cnt += suc_cnt
        #         epoch_total_sample_cnt += predict_prob.shape[0]
        #     test_acc = epoch_suc_cnt / epoch_total_sample_cnt
        #     # best_acc = max(test_acc, best_acc)   
        #     print(f"attribute inference with attack model, train_acc={train_acc}, test_acc={test_acc}")
        # ############################## v2: train model_T and model_attack separately ##############################

        # return train_acc, test_acc
        return train_acc, best_acc, test_auc

    def attack(self):
        self.set_seed(self.args.current_seed)
        for ik in self.party:  # attacker party #ik
            index = ik
            victim_party_list = [ik for ik in range(self.k)]
            victim_party_list.remove(index)
            # randomly select one party which is not the attacker as victim party
            victim_party = victim_party_list[0]

            batch_size = self.batch_size
            num_classes = self.label_size

            # get full data, train with aux, test with test
            train_victim_data = self.vfl_info["aux_data"][victim_party]
            train_local_data = self.vfl_info["aux_data"][index]
            # train_label = self.vfl_info["aux_label"][-1]
            train_attribute = self.vfl_info["aux_attribute"][-1].long()
            test_victim_data = self.vfl_info["test_data"][victim_party]
            test_local_data = self.vfl_info["test_data"][index]
            # test_label = self.vfl_info["test_label"][-1] # only active party have label
            test_attribute = self.vfl_info["test_attribute"][-1].long()  # only active party have attribute
            assert train_attribute != None, "[Error] No auxiliary attribute for Attribute Inference"
            assert test_attribute != None, "[Error] No test attribute for Attribute Inference"

            # actually, the training data and testing data should be M_p(X_p) not X_p
            bottom_victim_model = copy.deepcopy(self.vfl_info['model'][victim_party]).to(
                self.device)  # local bottom model for attacker
            bottom_local_model = copy.deepcopy(self.vfl_info['model'][index]).to(
                self.device)  # local bottom model for attacker
            global_model = copy.deepcopy(self.vfl_info['global_model']).to(self.device)
            train_victim_intermediate = bottom_victim_model(train_victim_data).to(self.device)
            train_local_intermediate = bottom_local_model(train_local_data).to(self.device)
            test_victim_intermediate = bottom_victim_model(test_victim_data).to(self.device)
            test_local_intermediate = bottom_local_model(test_local_data).to(self.device)

            # change attribute to "one-hot label"
            self.num_attributes = len(np.unique(train_attribute.cpu().numpy()))
            num_attributes = self.num_attributes
            print(
                f"[debug] in Attribute Inference, attribute has #class={num_attributes}, label has #class={self.num_classes}")
            train_label = label_to_one_hot(train_attribute, self.num_attributes)
            test_label = label_to_one_hot(test_attribute, self.num_attributes)

            self.z_dim = self.args.model_list[str(victim_party)]['output_dim'] + self.args.model_list[str(index)][
                'output_dim']

            print('all_train_data:', train_victim_data.size(), train_local_data.size(), train_victim_intermediate.shape,
                  train_local_intermediate.shape)
            print('all_train_label:', train_label.size())

            cudnn.benchmark = True
            train_data_loader_list = [
                DataLoader(ActiveDataset(train_victim_data, train_label), batch_size=batch_size, shuffle=False),
                DataLoader(ActiveDataset(train_local_data, train_label), batch_size=batch_size, shuffle=False)]
            test_data_loader_list = [
                DataLoader(ActiveDataset(test_victim_data, test_label), batch_size=batch_size, shuffle=False),
                DataLoader(ActiveDataset(test_local_data, test_label), batch_size=batch_size, shuffle=False)]

            # step1: train M_aux with D_aux (train here), which maps X_aux to A_aux
            # aux_model_list = [copy.deepcopy(self.vfl_info['model'][victim_party]).to(self.device), 
            #               copy.deepcopy(self.vfl_info['model'][index]).to(self.device), 
            #               copy.deepcopy(self.vfl_info['global_model']).to(self.device)]
            aux_model_list = self.train_auxiliary_model_for_attribute(train_data_loader_list, test_data_loader_list,
                                                                      victim_party_index=victim_party,
                                                                      attacker_index=index)

            # step2: initialize transfer-mapping model T (2 layer mlp) and attack model to map intermediate-results to attibute
            model_T = MLP2_scalable(self.z_dim, self.z_dim, hidden_dim=min(128, int(self.z_dim * 2))).to(self.device)
            model_attack = MLP2_scalable(self.z_dim, self.num_attributes, hidden_dim=min(128, int(self.z_dim * 2))).to(
                self.device)
            original_model_list = [bottom_victim_model, bottom_local_model, global_model]
            attribute_train_acc, attribute_test_acc, attribute_test_auc = self.train_mapper_attacker(
                train_data_loader_list, test_data_loader_list, original_model_list, aux_model_list, model_T,
                model_attack)

            ####### Clean ######
            for model in aux_model_list:
                del (model)
            for model in original_model_list:
                del (model)
            del (train_victim_data)
            del (train_local_data)
            del (train_attribute)
            del (test_victim_data)
            del (test_local_data)
            del (test_attribute)

        print(f"returning from Attribute Attack, test_auc={attribute_test_auc}")
        return attribute_test_acc
        # return recovery_history
