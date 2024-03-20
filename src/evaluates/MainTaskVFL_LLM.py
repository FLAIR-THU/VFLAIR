import json
import sys, os

sys.path.append(os.pardir)
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import time
import copy
import collections
import threading
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import scipy.stats as stats
import torch.nn as nn
import torch
import warnings
import collections

from transformers import top_k_top_p_filtering

# from models.vision import resnet18, MLP2
from utils.basic_functions import cross_entropy_for_onehot, append_exp_res, multiclass_auc

from utils.communication_protocol_funcs import get_size_of

# from evaluates.attacks.attack_api import apply_attack
from evaluates.defenses.defense_api import apply_defense
from evaluates.defenses.defense_functions import *
from utils.constants import *
import utils.constants as shared_var
from utils.marvell_functions import KL_gradient_perturb
from utils.noisy_label_functions import add_noise
from utils.noisy_sample_functions import noisy_sample
from utils.communication_protocol_funcs import compress_pred, Cache, ins_weight
from utils.squad_utils import normalize_answer, _get_best_indexes, get_tokens, compute_exact, compute_f1

from loguru import logger
from evaluates.attacks.attack_api import AttackerLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

from load.LoadModels import MODEL_PATH
from models.llm_models.qwen2 import E2EModel

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True

STOPPING_ACC = {'mnist': 0.977, 'cifar10': 0.80, 'cifar100': 0.40, 'diabetes': 0.69, \
                'nuswide': 0.88, 'breast_cancer_diagnose': 0.88, 'adult_income': 0.84, 'cora': 0.72, \
                'avazu': 0.83, 'criteo': 0.74, 'nursery': 0.99, 'credit': 0.82, 'news20': 0.8, \
                'cola_public': 0.8,
                'SST-2': 0.9}  # add more about stopping accuracy for different datasets when calculating the #communication-rounds needed


class MainTaskVFL_LLM(object):

    def __init__(self, args, job_id=None):
        self.job_id = job_id
        self.args = args
        self.k = args.k
        # self.k_server = args.k_server
        self.device = args.device
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
        # self.exp_res_dir = args.exp_res_dir

        # self.exp_res_path = args.exp_res_path
        self.parties = args.parties
        # self.servers = args.servers

        self.Q = args.Q  # FedBCD

        self.parties_data = None
        self.gt_one_hot_label = None
        self.clean_one_hot_label = None
        self.pred_list = []
        self.pred_list_clone = []
        self.pred_gradients_list = []
        self.pred_gradients_list_clone = []

        # FedBCD related
        self.local_pred_list = []
        self.local_pred_list_clone = []
        self.local_pred_gradients_list = []
        self.local_pred_gradients_list_clone = []

        self.loss = None
        self.train_acc = None
        self.flag = 1
        self.stopping_iter = 0
        self.stopping_time = 0.0
        self.stopping_commu_cost = 0
        self.communication_cost = 0

        # Early Stop
        self.early_stop_threshold = args.early_stop_threshold
        self.final_epoch = 0
        self.current_epoch = 0
        self.current_step = 0

        # some state of VFL throughout training process
        self.first_epoch_state = None
        self.middle_epoch_state = None
        self.final_state = None
        # self.final_epoch_state = None # <-- this is save in the above parameters

        self.num_update_per_batch = args.num_update_per_batch
        self.num_batch_per_workset = args.Q  # args.num_batch_per_workset
        self.max_staleness = self.num_update_per_batch * self.num_batch_per_workset
        self._last_result = None
        self._barrier = threading.Barrier(args.k)
        self.e2e_model = None  # type:E2EModel
        self._init_e2e_model()

    def set_last_result(self, result, run):
        if run == 'predict':
            self._last_result = json.loads(result)
            self._barrier.wait()
        else:
            self._last_result = result

    def get_last_result(self):
        return self._last_result

    def get_active_party(self):
        return self.parties[self.k - 1]

    def label_to_one_hot(self, target, num_classes=10):
        target = target.long()
        # print('label_to_one_hot:', target, type(target),type(target[0]))
        try:
            _ = target.size()[1]
            # print("use target itself", target.size())
            onehot_target = target.type(torch.float32).to(self.device)
        except:
            target = torch.unsqueeze(target, 1).to(self.device)
            # print("use unsqueezed target", target.size(),type(target))

            onehot_target = torch.zeros(target.size(0), num_classes, device=self.device)
            onehot_target.scatter_(1, target, 1)
        return onehot_target

    def LR_Decay(self, i_epoch):
        # for ik in range(self.k):
        #     self.parties[ik].LR_decay(i_epoch)
        self.parties[self.k - 1].global_LR_decay(i_epoch)

    def apply_defense_on_transmission(self, pred_detach):
        ########### Defense applied on pred transmit ###########
        if self.args.apply_defense == True and self.args.apply_dp == True:
            # print('pre pred_detach:',type(pred_detach),pred_detach.shape) # torch.size bs,12,768 intermediate
            pred_detach = torch.stack(self.launch_defense(pred_detach, "pred"))
            # print('after pred_detach:',type(pred_detach),pred_detach.shape) # torch.size bs,12,768 intermediate
        return pred_detach

    def apply_communication_protocol_on_transmission(self, pred_detach):
        ########### communication_protocols ###########
        if self.args.communication_protocol in ['Quantization', 'Topk']:
            pred_detach = compress_pred(self.args, pred_detach, self.parties[ik].local_gradient, \
                                        self.current_epoch, self.current_step).to(self.args.device)
        return pred_detach

    def pred_transmit(self):
        '''
        Active party gets pred from passive parties
        '''
        all_pred_list = []
        for ik in range(self.k - 1):
            # give pred
            if self.args.model_type == 'Bert':
                if self.args.task_type == 'SequenceClassification':
                    pred, pred_detach, attention_mask = self.parties[ik].give_pred()  # , _input_shape
                elif self.args.task_type == 'QuestionAnswering':
                    pred, pred_detach, attention_mask = self.parties[ik].give_pred()  # , _input_shape
                else:
                    assert 1 > 2, "task type not supported for finetune"
            elif self.args.model_type == 'Llama':
                if self.args.task_type == 'SequenceClassification':
                    pred, pred_detach, sequence_lengths, attention_mask = self.parties[ik].give_pred()  # , _input_shape
                elif self.args.task_type == 'CausalLM':
                    pred, pred_detach, attention_mask = self.parties[ik].give_pred()  # , _input_shape
                elif self.args.task_type == 'QuestionAnswering':
                    pred, pred_detach, attention_mask = self.parties[ik].give_pred()  # , _input_shape
                else:
                    assert 1 > 2, "task type not supported for finetune"

                # ########### Defense applied on pred transmit ###########
                # if self.args.apply_defense == True and self.args.apply_dp == True :
                #     if (ik in self.args.defense_configs['party']):
                #         pred_detach = torch.tensor(self.launch_defense(pred_detach, "pred")) 
                # ########### communication_protocols ###########
                # if self.args.communication_protocol in ['Quantization','Topk']:
                #     pred_detach = compress_pred( self.args ,pred_detach , self.parties[ik].local_gradient,\
                #                     self.current_epoch, self.current_step).to(self.args.device)

                # pred_clone = torch.autograd.Variable(pred_detach, requires_grad=True).to(self.args.device)
                # attention_mask = torch.autograd.Variable(attention_mask).to(self.args.device)
            elif self.args.model_type == 'GPT2':
                if self.args.task_type == 'SequenceClassification':
                    pred, pred_detach, sequence_lengths, attention_mask = self.parties[ik].give_pred()
                elif self.args.task_type == 'CausalLM':
                    pred, pred_detach, attention_mask, past_key_values = self.parties[ik].give_pred()
                elif self.args.task_type == 'QuestionAnswering':
                    pred, pred_detach, attention_mask = self.parties[ik].give_pred()
                else:
                    assert 1 > 2, "task type not supported for finetune"

            # Defense
            if self.args.apply_defense:
                if (ik in self.args.defense_configs['party']):
                    # print('Apply DP')
                    pred_detach = self.apply_defense_on_transmission(pred_detach)
            # Communication Process
            pred_detach = self.apply_communication_protocol_on_transmission(pred_detach)

            pred_clone = torch.autograd.Variable(pred_detach, requires_grad=True).to(self.args.device)
            attention_mask = torch.autograd.Variable(attention_mask).to(self.args.device)

            # receive pred
            if self.args.model_type == 'Bert':
                if self.args.task_type == 'SequenceClassification':
                    pred_list = [pred_clone, attention_mask]
                    self.parties[self.k - 1].receive_pred(pred_list, ik)
                    self.parties[ik].update_local_pred(pred_clone)
                    self.communication_cost += get_size_of(pred_clone) + \
                                               get_size_of(attention_mask)  # MB
                elif self.args.task_type == 'QuestionAnswering':
                    pred_list = [pred_clone, attention_mask]
                    self.parties[self.k - 1].receive_pred(pred_list, ik)
                    self.parties[ik].update_local_pred(pred_clone)
                    self.communication_cost += get_size_of(pred_clone) + get_size_of(attention_mask)  # MB
            elif self.args.model_type == 'Llama':
                if self.args.task_type == 'SequenceClassification':
                    pred_list = [pred_clone, sequence_lengths, attention_mask]
                    self.parties[self.k - 1].receive_pred(pred_list, ik)
                    self.parties[ik].update_local_pred(pred_clone)
                    self.communication_cost += get_size_of(pred_clone) + \
                                               get_size_of(sequence_lengths) + \
                                               get_size_of(attention_mask)  # MB
                elif self.args.task_type == 'CausalLM':
                    pred_list = [pred_clone, attention_mask]
                    self.parties[self.k - 1].receive_pred(pred_list, ik)
                    self.parties[ik].update_local_pred(pred_clone)
                    self.communication_cost += get_size_of(pred_clone) + \
                                               get_size_of(attention_mask)  # MB
                elif self.args.task_type == 'QuestionAnswering':
                    pred_list = [pred_clone, attention_mask]
                    self.parties[self.k - 1].receive_pred(pred_list, ik)
                    self.parties[ik].update_local_pred(pred_clone)
                    self.communication_cost += get_size_of(pred_clone) + get_size_of(attention_mask)  # MB
            elif self.args.model_type == 'GPT2':
                if self.args.task_type == 'SequenceClassification':
                    pred_list = [pred_clone, sequence_lengths, attention_mask]
                    self.parties[self.k - 1].receive_pred(pred_list, ik)
                    self.parties[ik].update_local_pred(pred_clone)
                    self.communication_cost += get_size_of(pred_clone) + get_size_of(attention_mask) + get_size_of(
                        sequence_lengths)  # MB
                elif self.args.task_type == 'CausalLM':
                    pred_list = [pred_clone, attention_mask, past_key_values]
                    self.parties[self.k - 1].receive_pred(pred_list, ik)
                    self.parties[ik].update_local_pred(pred_clone)
                    self.communication_cost += get_size_of(pred_clone) + get_size_of(attention_mask)  # MB
                elif self.args.task_type == 'QuestionAnswering':
                    pred_list = [pred_clone, attention_mask]
                    self.parties[self.k - 1].receive_pred(pred_list, ik)
                    self.parties[ik].update_local_pred(pred_clone)
                    self.communication_cost += get_size_of(pred_clone) + get_size_of(attention_mask)  # MB

            all_pred_list.append(pred_list)

        self.all_pred_list = all_pred_list
        return all_pred_list

    def global_pred_transmit(self):
        # active party give global pred to passive party
        final_pred = self.parties[self.k - 1].aggregate(self.parties[self.k - 1].pred_received, test="True")

        for ik in range(self.k - 1):
            # self.communication_cost += get_size_of(final_pred)
            self.parties[ik].global_pred = final_pred

        return final_pred

    def local_gradient_transmit(self):
        for ik in range(self.k - 1):
            if self.parties[ik].local_model_optimizer != None:
                passive_local_gradient = self.parties[self.k - 1].cal_passive_local_gradient(ik)
                self.parties[ik].local_gradient = passive_local_gradient

    def global_gradient_transmit(self, final_pred):
        global_loss = self.parties[0].cal_loss(final_pred)

        global_gradients = self.parties[0].cal_global_gradient(global_loss, final_pred)

        self.communication_cost += get_size_of(global_gradients)

        self.parties[self.k - 1].receive_loss_and_gradients(self.parties[0].global_loss,
                                                            self.parties[0].global_gradients)
        # self.parties[self.k-1].global_loss = self.parties[0].global_loss
        # self.parties[self.k-1].global_gradients = self.parties[0].global_gradients

    def qa_inference(self):
        # QA
        exact_score_list = []
        f1_list = []
        for ik in range(self.k - 1):
            # Passive data local predict
            result = self.parties[ik].predict()

            nbest_list, gold_ans_list, total_sample_cnt = result

        total_scores = []
        best_non_null_entry = None

        for nbest, gold_ans in zip(nbest_list, gold_ans_list):
            print('nbest:', nbest)
            print('gold_ans:', gold_ans)

            exact_score = 0
            f1 = 0
            if self.args.metric_type == "best_pred":
                for entry in nbest:
                    total_scores.append(entry.start_logit + entry.end_logit)
                    if not best_non_null_entry:
                        if entry.text:
                            best_non_null_entry = entry
                pred_ans_text = best_non_null_entry.text if (best_non_null_entry != None) else ""
                print('pred_ans_text:', pred_ans_text)
                print('gold_ans:', gold_ans)

                exact_score = max(compute_exact(a, pred_ans_text) for a in gold_ans)
                f1 = max(compute_f1(a, pred_ans_text) for a in gold_ans)
                print('exact_score:', exact_score)
                print('-' * 100)

                exact_score_list.append(exact_score)
                f1_list.append(f1)
            elif self.args.metric_type == "n_best":
                for entry in nbest:
                    total_scores.append(entry.start_logit + entry.end_logit)
                    if not best_non_null_entry:
                        if entry.text:
                            best_non_null_entry = entry
                    pred_ans_text = entry.text
                    exact_score = max(exact_score, max(compute_exact(a, pred_ans_text) for a in gold_ans))
                    f1 = max(f1, max(compute_f1(a, pred_ans_text) for a in gold_ans))
                exact_score_list.append(exact_score)
                f1_list.append(f1)
            else:
                assert 1 > 2, f"{self.args.metric_type} not provided!"

        exact_score = np.mean(exact_score_list)
        f1 = np.mean(f1_list)
        exp_result = 'exact_score:{:.4f} f1:{:.4f}'.format(exact_score, f1)

        self.test_acc = exact_score
        # self.final_state = self.save_state(False)
        # self.final_state.update(self.save_party_data())
        print(exp_result)
        return exp_result, self.test_acc

    def vfl_forward(self, input_ids, attention_mask, token_type_ids, past_key_values):

        # print('=== vfl forward ===')
        # print('input_ids:',input_ids)
        # print('attention_mask:',attention_mask.shape)
        # print('token_type_ids:',token_type_ids)

        if past_key_values != None:
            self.local_past_key_values = past_key_values[:self.args.local_encoders_num]
            self.global_past_key_values = past_key_values[self.args.local_encoders_num:]
            self.all_past_key_values = past_key_values
        else:
            self.local_past_key_values = None
            self.global_past_key_values = None
            self.all_past_key_values = None

        input_shape = input_ids.shape[:2]  # batchsize, seq_length
        self.input_shape = input_shape

        self.parties[0].obtain_local_data(input_ids, attention_mask, token_type_ids, self.local_past_key_values)
        self.parties[1].obtain_local_data(None, None, None, past_key_values)

        # self.gt_one_hot_label = gt_one_hot_label

        # self.parties[1].encoder_attention_mask = self.parties[0].local_model.encoder_attention_mask
        # self.parties[1].encoder_hidden_states = self.parties[0].local_model.encoder_hidden_states

        # passive party do local pred
        pred_list = self.parties[0].pred_transmit(use_cache=True)

        # local_past_key_values = self.parties[0].local_model.past_key_values

        # passive party inform active party to do global pred
        test_logit = self.parties[0]._send_pred_message(pred_list, use_cache=False)

        final_output = self.parties[1].global_output
        return final_output

    def generate_result(self, test_logit, gt_one_hot_label, parties_data):
        test_preds = []
        test_targets = []
        test_predict_labels = []
        test_actual_labels = []
        target_word_list = []
        predict_word_list = []
        suc_cnt = 0
        sample_cnt = 0

        if self.args.task_type == "SequenceClassification":
            if self.args.num_classes == 1:  # regression
                predict_label = test_logit.detach().cpu()
                actual_label = gt_one_hot_label.detach().cpu()

                predict_label = torch.tensor([_.item() for _ in predict_label])
                actual_label = torch.tensor([_.item() for _ in actual_label])

                sample_cnt = predict_label.shape[0]

                return list(predict_label), list(actual_label), sample_cnt

            else:  # Classification
                enc_predict_prob = test_logit

                predict_label = torch.argmax(enc_predict_prob, dim=-1)
                actual_label = torch.argmax(gt_one_hot_label, dim=-1)

                test_preds.append(list(enc_predict_prob.detach().cpu().numpy()))
                test_targets.append(list(gt_one_hot_label.detach().cpu().numpy()))

                sample_cnt = predict_label.shape[0]
                suc_cnt += torch.sum(predict_label == actual_label).item()
                return list(predict_label.detach().cpu()), list(actual_label.detach().cpu()), sample_cnt

        elif self.args.task_type == "CausalLM":
            # get logits of last hidden state
            # print('test_logit:',test_logit.shape) # [batchsize, maxlength512, vocab_size32000]
            next_token_logits = test_logit[:, -1]  # [bs, 32000]

            if self.args.dataset == "Lambada":
                # print('gt_one_hot_label:',type(gt_one_hot_label),gt_one_hot_label) # list of target tokens
                target_word = [normalize_answer(_p) for _p in gt_one_hot_label]  # list of normalized tokens
                target_word_list.extend(target_word)

                enc_predict_prob = nn.functional.softmax(next_token_logits, dim=-1)

                if self.args.metric_type == "best_pred":
                    predict_label = torch.argmax(enc_predict_prob, dim=-1)  # [bs]
                    predict_word = [self.args.tokenizer.decode([_best_id]) for _best_id in predict_label.tolist()]
                    # print('predict_label:',predict_label,' predict_word:',predict_word)
                    predict_word = [normalize_answer(_p) for _p in predict_word]
                    predict_word_list.extend(predict_word)  # predict_word: bs * best_pred
                elif self.args.metric_type == "n_best":
                    logit_list, index_list = torch.sort(enc_predict_prob, descending=True)
                    predict_label = index_list[:, :self.args.n_best_size]
                    for _bs in range(predict_label.shape[0]):  # each batch
                        predict_word = [self.args.tokenizer.decode([_label]) for _label in predict_label[_bs].tolist()]
                        predict_word = [normalize_answer(_p) for _p in predict_word]
                        predict_word_list.append(predict_word)  # predict_word: list of n best for this batch

                return target_word_list, predict_word_list, None

            else:  # MMLU
                choice_id_list = []
                for choice in self.args.label_dict.keys():
                    choice_id_list.append(self.args.tokenizer(choice).input_ids[-1])
                    _id = self.args.tokenizer(choice).input_ids[-1]
                enc = next_token_logits[:, choice_id_list]  # [bs, num_choice]
                enc_predict_prob = nn.functional.softmax(enc, dim=-1)  # [bs, num_choice]

                predict_label = torch.argmax(enc_predict_prob, dim=-1)  # [bs]
                actual_label = gt_one_hot_label  # torch.argmax(gt_one_hot_label, dim=-1)

                test_predict_labels.extend(predict_label.detach().cpu().tolist())
                test_actual_labels.extend(actual_label.detach().cpu().tolist())

                sample_cnt += predict_label.shape[0]
                suc_cnt += torch.sum(predict_label == actual_label).item()

                return [], [], 0

        elif self.args.task_type == "QuestionAnswering":
            start_logits = test_logit.start_logits
            end_logits = test_logit.end_logits
            sample_cnt = start_logits.shape[0]

            n_best_size = self.args.n_best_size
            start_indexes = [_get_best_indexes(_logits, n_best_size) for _logits in start_logits]
            end_indexes = [_get_best_indexes(_logits, n_best_size) for _logits in end_logits]

            exact_score_list = []
            f1_list = []
            batch_nbest_list = []
            batch_gold_ans_list = []
            for i in range(start_logits.shape[0]):  # for each sample in this batch
                _start_logits = start_logits[i]
                _end_logits = end_logits[i]
                _start_indexes = start_indexes[i]
                _end_indexes = end_indexes[i]

                ############ Gold ################
                feature = parties_data[0][4][
                    i]  # print('parties_data[0][4]:',type(parties_data[0][4]),'feature:',type(feature))
                feature_tokens = [_token for _token in feature["tokens"]]  # [_token[0] for _token in feature["tokens"]]

                gold_start_indexs, gold_end_indexs = gt_one_hot_label[i]  # the i'th sample in a batch
                if len(gold_start_indexs.shape) == 0:
                    gold_start_indexs = gold_start_indexs.unsqueeze(0)
                if len(gold_end_indexs.shape) == 0:
                    gold_end_indexs = gold_end_indexs.unsqueeze(0)

                gold_ans = []  # gold answers for this sample
                for _i in range(len(gold_start_indexs)):
                    gold_start_index = int(gold_start_indexs[_i])
                    gold_end_index = int(gold_end_indexs[_i])
                    if gold_start_index == -1:
                        continue
                    gold_ans_text = " ".join(feature_tokens[gold_start_index:(gold_end_index + 1)])
                    gold_ans_text = normalize_answer(gold_ans_text)
                    gold_ans.append(gold_ans_text)
                batch_gold_ans_list.append(gold_ans)

                ############ Pred ################
                _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                    "PrelimPrediction",
                    ["start_index", "end_index", "start_logit", "end_logit"])
                _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                    "NbestPrediction", ["text", "start_logit", "end_logit"])

                # iterate through all possible start-end pairs
                prelim_predictions = []
                for start_index in _start_indexes:
                    for end_index in _end_indexes:
                        # We could hypothetically create invalid predictions, e.g., predict
                        # that the start of the span is in the question. We throw out all
                        # invalid predictions.
                        if start_index >= len(feature["tokens"]):
                            continue
                        if end_index >= len(feature["tokens"]):
                            continue
                        if start_index not in feature["token_to_orig_map"]:
                            continue
                        if end_index not in feature["token_to_orig_map"]:
                            continue
                        if not feature["token_is_max_context"].get(start_index, False):
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > self.args.max_answer_length:
                            continue

                        prelim_predictions.append(
                            _PrelimPrediction(
                                start_index=start_index,
                                end_index=end_index,
                                start_logit=_start_logits[start_index],
                                end_logit=_end_logits[end_index]))

                # Iterate through Sorted Predictions
                prelim_predictions = sorted(
                    prelim_predictions,
                    key=lambda x: (x.start_logit + x.end_logit),
                    reverse=True)

                exact_score = 0
                f1 = 0
                # Get n best prediction text
                nbest = []
                n_best_size = min(n_best_size, len(prelim_predictions))
                for _id in range(n_best_size):
                    start_index = prelim_predictions[_id].start_index
                    end_index = prelim_predictions[_id].end_index

                    pred_ans_text = " ".join(feature_tokens[start_index:(end_index + 1)])
                    pred_ans_text = normalize_answer(pred_ans_text)

                    nbest.append(
                        _NbestPrediction(
                            text=pred_ans_text,
                            start_logit=prelim_predictions[_id].start_logit,
                            end_logit=prelim_predictions[_id].end_logit))
                batch_nbest_list.append(nbest)

            return batch_nbest_list, batch_gold_ans_list, sample_cnt

            # # Get best predicted answer
            # total_scores = []
            # best_non_null_entry = None

            # if self.args.metric_type == "best_pred":
            #     for entry in nbest:
            #         total_scores.append(entry.start_logit + entry.end_logit)
            #         if not best_non_null_entry:
            #             if entry.text:
            #                 best_non_null_entry = entry
            #     pred_ans_text = best_non_null_entry.text if (best_non_null_entry != None) else ""
            #     # Calculate exact_score/f1
            #     # print('best pred:',pred_ans_text)
            #     exact_score = max(compute_exact(a, pred_ans_text) for a in gold_ans)
            #     f1 = max(compute_f1(a, pred_ans_text) for a in gold_ans)
            #     # print('this batch:',exact_score,f1)
            #     exact_score_list.append(exact_score)
            #     f1_list.append(f1)
            # elif self.args.metric_type == "n_best":
            #     for entry in nbest:
            #         total_scores.append(entry.start_logit + entry.end_logit)
            #         if not best_non_null_entry:
            #             if entry.text:
            #                 best_non_null_entry = entry
            #         pred_ans_text = entry.text  # print('best pred:',pred_ans_text)

            #         # Calculate exact_score/f1
            #         exact_score = max(exact_score, max(compute_exact(a, pred_ans_text) for a in gold_ans))
            #         f1 = max(f1, max(compute_f1(a, pred_ans_text) for a in gold_ans))
            #     # print('this batch:',exact_score,f1)
            #     exact_score_list.append(exact_score)
            #     f1_list.append(f1)
            # else:
            #     assert 1 > 2, f"{self.args.metric_type} not provided!"

            # return exact_score_list, f1_list, sample_cnt #None

        else:
            assert 1 > 2, "task_type not supported"

    def predict(self):
        # passive party dataloader list
        data_loader_list = [self.parties[ik].test_loader for ik in range(self.args.k - 1)]

        exact_score_list = []
        f1_list = []

        nbest_list = []
        gold_ans_list = []

        target_word_list = []
        predict_word_list = []

        predict_label_list = []
        actual_label_list = []

        total_sample_cnt = 0
        with torch.no_grad():
            for parties_data in zip(*data_loader_list):
                _parties_data = []
                for party_id in range(len(parties_data)):  # iter through each passive party
                    batch_input_ids = []
                    batch_label = []
                    batch_attention_mask = []
                    batch_token_type_ids = []
                    batch_feature = []
                    for bs_id in range(len(parties_data[party_id])):
                        # Input_ids
                        batch_input_ids.append(parties_data[party_id][bs_id][0].tolist())
                        # Attention Mask
                        batch_attention_mask.append(parties_data[party_id][bs_id][2].tolist())

                        # ptoken_type_ids
                        if parties_data[party_id][bs_id][3] == []:
                            batch_token_type_ids = None
                        else:
                            batch_token_type_ids.append(parties_data[party_id][bs_id][3].tolist())

                        # feature (for QuestionAnswering only)
                        if parties_data[party_id][bs_id][4] == []:
                            batch_feature = None
                        else:
                            batch_feature.append(parties_data[party_id][bs_id][4])

                        # Label
                        if type(parties_data[party_id][bs_id][1]) != str:
                            batch_label.append(parties_data[party_id][bs_id][1].tolist())
                        else:
                            batch_label.append(parties_data[party_id][bs_id][1])

                    batch_input_ids = torch.tensor(batch_input_ids).to(self.device)
                    batch_attention_mask = torch.tensor(batch_attention_mask).to(self.device)
                    if batch_token_type_ids != None:
                        batch_token_type_ids = torch.tensor(batch_token_type_ids).to(self.device)

                    if type(batch_label[0]) != str:
                        if self.args.task_type == 'QuestionAnswering':
                            # origin_batch_label_shape [bs, 2, num_of_answers]
                            # 2*bs, num_of_answers
                            # print('batch_label:',batch_label)
                            if type(batch_label[0][0]) == list:
                                batch_label = pad_sequence([torch.tensor(position_list) for sample_label in batch_label \
                                                            for position_list in sample_label], batch_first=True,
                                                           padding_value=-1).to(self.device)
                                origin_batch_label_shape = [int(batch_label.shape[0] / 2), 2, batch_label.shape[1]]
                                batch_label = batch_label.reshape(origin_batch_label_shape)
                            else:
                                batch_label = torch.tensor(batch_label).to(self.device)
                        else:
                            batch_label = torch.tensor(batch_label).to(self.device)

                    _parties_data.append \
                        ([batch_input_ids, batch_label, batch_attention_mask, batch_token_type_ids, batch_feature])
                parties_data = _parties_data

                if self.args.task_type == "SequenceClassification" and self.args.num_classes > 1:  # classification
                    gt_one_hot_label = self.label_to_one_hot(parties_data[0][1], self.args.num_classes)
                elif self.args.task_type == "QuestionAnswering":
                    gt_one_hot_label = list(parties_data[0][1])
                else:
                    gt_one_hot_label = parties_data[0][1]

                self.input_shape = parties_data[0][0].shape[:2]  # batchsize, seq_length
                for ik in range(self.args.k - 1):
                    # input_ids, local_batch_attention_mask, local_batch_token_type_ids, past_key_values = None
                    self.parties[ik].obtain_local_data(parties_data[0][0], parties_data[0][2], parties_data[0][3])
                self.gt_one_hot_label = gt_one_hot_label

                # print( 'batch_label:',parties_data[0][1])
                # print( '== batch_feature ==')
                # print( 'tokens', parties_data[0][4][0]['tokens'])

                # start_position = parties_data[0][4][0]['start_position']
                # end_position = parties_data[0][4][0]['end_position']
                # print(start_position, end_position)
                # print( 'tokens start_end', parties_data[0][4][0]['tokens'][start_position[0]: end_position[0]+1])
                # print( 'tokens start_end', parties_data[0][4][0]['tokens'][start_position[1]: end_position[1]+1])

                # print( 'orig_answer_text', parties_data[0][4][0]['orig_answer_text'])

                # Passive Party -> pred_list[local pred]
                pred_list = self.pred_transmit()
                # local pred list -> Active Party -> test_logit[final pred]
                test_logit = self.global_pred_transmit(pred_list)

                # test_logit -> standard output for each task
                if self.args.task_type == "SequenceClassification":  # and self.args.num_classes > 1: # classification
                    batch_predict_label, batch_actual_label, sample_cnt = self.generate_result(test_logit,
                                                                                               gt_one_hot_label,
                                                                                               parties_data)
                    predict_label_list.extend(batch_predict_label)
                    actual_label_list.extend(batch_actual_label)
                    if sample_cnt is not None:
                        total_sample_cnt += sample_cnt
                elif self.args.task_type == "QuestionAnswering":
                    batch_nbest, batch_gold_ans, sample_cnt = self.generate_result(test_logit, gt_one_hot_label,
                                                                                   parties_data)
                    nbest_list.extend(batch_nbest)
                    gold_ans_list.extend(batch_gold_ans)

                    # exact_scores, f1s, sample_cnt = self.generate_result(test_logit, gt_one_hot_label, parties_data)
                    # exact_scores_list.extend(exact_scores)
                    # f1_list.extend(f1s)
                    if sample_cnt is not None:
                        total_sample_cnt += sample_cnt
                elif self.args.task_type == "CausalLM":
                    batch_target_word, batch_predict_word, sample_cnt = self.generate_result(test_logit,
                                                                                             gt_one_hot_label,
                                                                                             parties_data)
                    target_word_list.extend(batch_target_word)
                    predict_word_list.extend(batch_predict_word)
                    if sample_cnt is not None:
                        total_sample_cnt += sample_cnt
                else:
                    assert 1 > 2, 'Task type not supported'

                del parties_data

        if self.args.task_type == "SequenceClassification":  # and self.args.num_classes > 1: # classification
            return predict_label_list, actual_label_list, total_sample_cnt
        elif self.args.task_type == "QuestionAnswering":
            # return exact_score_list, f1_list, total_sample_cnt
            return nbest_list, gold_ans_list, total_sample_cnt
        elif self.args.task_type == "CausalLM":
            return target_word_list, predict_word_list, total_sample_cnt
        else:
            assert 1 > 2, 'Task type not supported'

    def seq_inference(self):
        # SequenceClassification
        test_predict_labels = []
        test_actual_labels = []
        postfix = {'test_acc': 0.0}
        suc_cnt = 0
        total_sample_cnt = 0
        for ik in range(self.k - 1):
            # Passive data local predict
            predict_labels, actual_labels, sample_cnt = self.parties[ik].predict()
            test_predict_labels.extend(predict_labels)
            test_actual_labels.extend(actual_labels)
            total_sample_cnt += sample_cnt

        if self.num_classes == 1:
            self.test_mse = torch.mean(
                (torch.tensor(test_predict_labels) - torch.tensor(test_actual_labels)) ** 2).item()
            # torch.nn.MSELoss()(  torch.tensor(test_predict_labels), torch.tensor(test_actual_labels) ).item()

            # print('test_predict_labels:',test_predict_labels[:5])
            # print('test_actual_labels:',test_actual_labels[:5])
            self.test_pearson_corr = \
                stats.pearsonr(torch.tensor(test_predict_labels), torch.tensor(test_actual_labels))[0]
            # full_test_pearson_corr = pearsonr( torch.tensor(test_full_predict_labels), torch.tensor(test_actual_labels) )[0]
            self.test_spearmanr_corr = \
                stats.spearmanr(torch.tensor(test_predict_labels), torch.tensor(test_actual_labels))[0]
            postfix['test_mse'] = '{:.4f}%'.format(self.test_mse * 100)
            postfix['test_pearson_corr'] = '{:.4f}%'.format(self.test_pearson_corr * 100)

            exp_result = 'test_mse:{:.4f} test_pearson_corr:{:.4f} test_spearmanr_corr:{:.4f}'.format(self.test_mse,
                                                                                                      self.test_pearson_corr,
                                                                                                      self.test_spearmanr_corr)
            print(exp_result)
            # print('Full pred pearson:',full_test_pearson_corr)
            return exp_result, self.test_mse
        else:
            suc_cnt = torch.sum(torch.tensor(test_predict_labels) == \
                                torch.tensor(test_actual_labels)).item()
            print('test_predict_labels:', test_predict_labels[:10])
            print('test_actual_labels:', test_actual_labels[:10])

            self.test_acc = suc_cnt / float(total_sample_cnt)  # ACC
            self.test_mcc = matthews_corrcoef(np.array(test_predict_labels), np.array(test_actual_labels))  # MCC

            postfix['test_acc'] = '{:.2f}%'.format(self.test_acc * 100)
            postfix['test_mcc'] = '{:.2f}%'.format(self.test_mcc * 100)

            exp_result = 'test_acc:{:.2f} test_mcc:{:.2f}'.format(self.test_acc, self.test_mcc)
            print(exp_result)

            # self.final_state = self.save_state(False)
            # self.final_state.update(self.save_party_data())

            return exp_result, self.test_acc

    def _llm_inference(self, **kwargs):
        if self.k > 2:
            raise ValueError('llm_inference only supports k=2')

        format_kwargs = self._format_forward_kwargs(**kwargs)
        generate_ids = self.e2e_model.generate(format_kwargs.get('input_ids'), max_new_tokens=20)
        generate_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(format_kwargs['input_ids'], generate_ids)
        ]
        resp = self.args.tokenizer.batch_decode(generate_ids, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)
        logger.debug(f"text generation: {resp}")
        return '', ''

    def _format_forward_kwargs(self, **kwargs):
        if not kwargs:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "You are a python programmer, what can you do?"}
            ]
        else:
            messages = kwargs['messages']
        tokenizer = self.args.tokenizer
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt")
        kwargs.update({'input_ids': model_inputs.input_ids.to(self.parties[0].device),
                       'output_hidden_states': True})
        logger.debug(f"default inference, kwargs.keys: {kwargs.keys()}")
        base_dict = {'input_ids': None,
                     'attention_mask': None,
                     'position_ids': None,
                     'past_key_values': None,
                     'inputs_embeds': None,
                     'use_cache': False,
                     'output_attentions': None,
                     'output_hidden_states': True,
                     'return_dict': None, }
        for k in base_dict:
            if k in kwargs:
                base_dict.update({k: kwargs.get(k)})
        return base_dict

    def causal_llm_inference(self):
        postfix = {'test_acc': 0.0}
        target_word_list = []
        predict_word_list = []
        for ik in range(self.k - 1):
            # Passive data local predict
            result = self.parties[ik].predict()
            target_words, predict_words, _ = result
            target_word_list.extend(target_words)
            predict_word_list.extend(predict_words)

        if self.args.metric_type == "best_pred":
            suc_cnt = 0
            for i in range(len(target_word_list)):
                if target_word_list[i] == predict_word_list[i]:
                    suc_cnt += 1
            self.test_acc = suc_cnt / float(len(target_word_list))  # ACC
        elif self.args.metric_type == "n_best":
            suc_cnt = 0
            for i in range(len(target_word_list)):
                if target_word_list[i] in predict_word_list[i]:
                    suc_cnt += 1
            self.test_acc = suc_cnt / float(len(target_word_list))  # ACC
        else:
            assert 1 > 2, 'metric type not supported'

        postfix['test_acc'] = '{:.2f}%'.format(self.test_acc * 100)

        exp_result = 'test_acc:{:.2f}'.format(self.test_acc)
        print(exp_result)

        # self.final_state = self.save_state(False)
        # self.final_state.update(self.save_party_data())

        return exp_result, self.test_acc

    def inference(self, **kwargs):
        # print(' ========= Inference ==========')
        for ik in range(self.k - 1):
            self.parties[ik].prepare_data_loader()
            self.parties[ik].eval()
        self.parties[self.k - 1].eval()

        if self.args.task_type == "QuestionAnswering":
            exp_result, main_task_result = self.qa_inference()
            return exp_result, main_task_result

        elif self.args.task_type == "SequenceClassification":
            # exp_result, self.test_acc =
            exp_result, main_task_result = self.seq_inference()
            return exp_result, main_task_result

        elif self.args.task_type == "CausalLM":
            # exp_result, self.test_acc =
            exp_result, main_task_result = self.causal_llm_inference()
            return exp_result, main_task_result

        elif self.args.task_type == "DevLLMInference":
            # LLM推理入口，兼容forward格式入参
            result = self._llm_inference(**kwargs)
            return result

    def train_batch(self, parties_data, batch_label):
        '''
        batch_label: self.gt_one_hot_label   may be noisy
            QA: bs * [start_position, end_position]
        '''
        ############### allocate data ###############
        gt_one_hot_label = batch_label
        self.gt_one_hot_label = gt_one_hot_label
        # allocate data/attention mask to passive party
        for ik in range(self.k - 1):
            # allocate data (data/label/attention_mask/token_type_ids)
            input_shape = parties_data[ik][0].shape[:2]  # parties_data[ik][0].size()
            self.parties[ik].input_shape = input_shape
            self.parties[ik].obtain_local_data(parties_data[ik][0], parties_data[ik][2], parties_data[ik][3])
            self.parties[ik].gt_one_hot_label = gt_one_hot_label
        ############### allocate data ###############

        ################ normal vertical federated learning ################
        torch.autograd.set_detect_anomaly(True)

        # =================== Commu ===================
        # exchange info between party: local_pred/global_pred
        all_pred_list = self.pred_transmit()  # [ pred of this party ]
        final_pred = self.parties[self.args.k - 1].aggregate(all_pred_list,
                                                             test="True")  # self._send_pred_message(all_pred_list)

        # passive party -> global gradient -> active party
        self.global_gradient_transmit(final_pred)

        # active party -> local gradient -> passive party
        self.local_gradient_transmit()
        # =================== Commu ===================

        # ============= Model Update =============
        # update parameters for global trainable part
        self.args.parties[self.args.k - 1].global_backward()  # self._send_global_backward_message()
        for ik in range(self.k - 1):
            self.parties[ik].local_backward()
        # ============= Model Update =============

        ################ normal vertical federated learning ################

        # print train_acc each batch
        if self.args.task_type == 'QuestionAnswering':
            pred = self.parties[self.k - 1].global_pred  # QuestionAnsweringModelOutput
            loss = self.parties[self.k - 1].global_loss

            start_logits = pred.start_logits
            end_logits = pred.end_logits

            n_best_size = self.args.n_best_size
            start_indexes = [_get_best_indexes(_logits, n_best_size) for _logits in start_logits]
            end_indexes = [_get_best_indexes(_logits, n_best_size) for _logits in end_logits]

            exact_score_list = []
            f1_list = []
            # for each sample in this batch
            for i in range(start_logits.shape[0]):
                _start_logits = start_logits[i]
                _end_logits = end_logits[i]
                _start_indexes = start_indexes[i]
                _end_indexes = end_indexes[i]

                ############ Gold ################
                feature = parties_data[0][4][i]
                feature_tokens = [_token[0] for _token in feature["tokens"]]

                gold_start_indexs, gold_end_indexs = gt_one_hot_label[i]  # the i'th sample in a batch
                if len(gold_start_indexs.shape) == 0:
                    gold_start_indexs = gold_start_indexs.unsqueeze(0)
                if len(gold_end_indexs.shape) == 0:
                    gold_end_indexs = gold_end_indexs.unsqueeze(0)
                gold_ans = []  # gold answers for this sample
                for _i in range(len(gold_start_indexs)):
                    gold_start_index = int(gold_start_indexs[_i])
                    gold_end_index = int(gold_end_indexs[_i])
                    gold_ans_text = " ".join(feature_tokens[gold_start_index:(gold_end_index + 1)])
                    gold_ans_text = normalize_answer(gold_ans_text)
                    gold_ans.append(gold_ans_text)
                # print('gold_ans:',gold_ans,feature["orig_answer_text"])

                ############ Pred ################
                _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                    "PrelimPrediction",
                    ["start_index", "end_index", "start_logit", "end_logit"])
                _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                    "NbestPrediction", ["text", "start_logit", "end_logit"])

                # iterate through all possible start-end pairs
                prelim_predictions = []
                for start_index in _start_indexes:
                    for end_index in _end_indexes:
                        # We could hypothetically create invalid predictions, e.g., predict
                        # that the start of the span is in the question. We throw out all
                        # invalid predictions.
                        if start_index >= len(feature["tokens"]):
                            continue
                        if end_index >= len(feature["tokens"]):
                            continue
                        if start_index not in feature["token_to_orig_map"]:
                            continue
                        if end_index not in feature["token_to_orig_map"]:
                            continue
                        if not feature["token_is_max_context"].get(start_index, False):
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > self.args.max_answer_length:
                            continue

                        prelim_predictions.append(
                            _PrelimPrediction(
                                start_index=start_index,
                                end_index=end_index,
                                start_logit=_start_logits[start_index],
                                end_logit=_end_logits[end_index]))

                # Iterate through Sorted Predictions
                prelim_predictions = sorted(
                    prelim_predictions,
                    key=lambda x: (x.start_logit + x.end_logit),
                    reverse=True)
                exact_score = 0
                f1 = 0
                # Get n best prediction text
                nbest = []
                n_best_size = min(n_best_size, len(prelim_predictions))
                for _id in range(n_best_size):
                    start_index = prelim_predictions[_id].start_index
                    end_index = prelim_predictions[_id].end_index

                    pred_ans_text = " ".join(feature_tokens[start_index:(end_index + 1)])
                    pred_ans_text = normalize_answer(pred_ans_text)

                    nbest.append(
                        _NbestPrediction(
                            text=pred_ans_text,
                            start_logit=prelim_predictions[_id].start_logit,
                            end_logit=prelim_predictions[_id].end_logit))

                # Get best predicted answer
                total_scores = []
                best_non_null_entry = None

                if self.args.metric_type == "best_pred":
                    for entry in nbest:
                        total_scores.append(entry.start_logit + entry.end_logit)
                        if not best_non_null_entry:
                            if entry.text:
                                best_non_null_entry = entry
                    pred_ans_text = best_non_null_entry.text if (best_non_null_entry != None) else ""
                    # Calculate exact_score/f1 for best pred
                    # print('best pred:',pred_ans_text)
                    exact_score = max(compute_exact(a, pred_ans_text) for a in gold_ans)
                    f1 = max(compute_f1(a, pred_ans_text) for a in gold_ans)
                    # print('this batch:',exact_score,f1)
                    exact_score_list.append(exact_score)
                    f1_list.append(f1)
                elif self.args.metric_type == "n_best":
                    for entry in nbest:
                        total_scores.append(entry.start_logit + entry.end_logit)
                        if not best_non_null_entry:
                            if entry.text:
                                best_non_null_entry = entry
                        pred_ans_text = entry.text  # print('best pred:',pred_ans_text)
                        # Calculate best exact_score/f1 among n best preds
                        exact_score = max(exact_score, max(compute_exact(a, pred_ans_text) for a in gold_ans))
                        f1 = max(f1, max(compute_f1(a, pred_ans_text) for a in gold_ans))
                    # print('this batch:',exact_score,f1)
                    exact_score_list.append(exact_score)
                    f1_list.append(f1)
                else:
                    assert 1 > 2, f"{self.args.metric_type} not provided!"

            exact_score = np.mean(exact_score_list)
            f1 = np.mean(f1_list)

            return loss.item(), exact_score

        elif self.args.task_type == 'SequenceClassification':
            real_batch_label = batch_label

            pred = final_pred
            predict_prob = F.softmax(pred, dim=-1)

            suc_cnt = torch.sum(torch.argmax(predict_prob, dim=-1) == torch.argmax(real_batch_label, dim=-1)).item()
            batch_train_acc = suc_cnt / predict_prob.shape[0]

            loss = self.parties[self.k - 1].global_loss

            return loss.item(), batch_train_acc

        elif self.args.task_type == 'CausalLM':
            pred = self.parties[self.k - 1].global_pred  # logits
            loss = self.parties[self.k - 1].global_loss
            test_logit = pred
            next_token_logits = test_logit[:,
                                -1]  # [bs, 32000] # print('next_token_logits:',next_token_logits.shape,next_token_logits)

            if self.args.dataset == "Lambada":
                # print('gt_one_hot_label:',type(gt_one_hot_label),gt_one_hot_label)
                target_word_list = [normalize_answer(_p) for _p in gt_one_hot_label]
                # print('target_word_list:',type(target_word_list),len(target_word_list),target_word_list)

                # predict_word_list : bs * predicted words
                enc_predict_prob = nn.functional.softmax(next_token_logits, dim=-1)
                if self.args.metric_type == "best_pred":
                    predict_label = torch.argmax(enc_predict_prob, dim=-1)  # [bs]
                    predict_word = [self.args.tokenizer.decode([_best_id]) for _best_id in predict_label.tolist()]
                    predict_word_list = [normalize_answer(_p) for _p in predict_word]
                elif self.args.metric_type == "n_best":
                    logit_list, index_list = torch.sort(enc_predict_prob, descending=True)
                    # print('index_list:',index_list.shape)
                    predict_label = index_list[:, :self.args.n_best_size]
                    # print('predict_label:',predict_label.shape)
                    predict_word_list = []
                    for _bs in range(predict_label.shape[0]):  # each batch
                        predict_word = [self.args.tokenizer.decode([_label]) for _label in predict_label[_bs].tolist()]
                        predict_word = [normalize_answer(_p) for _p in predict_word]
                        predict_word_list.append(predict_word)  # predict_word: list of n best for this batch
                        # print('predict_word:',predict_word)
                # print('predict_word_list:',type(predict_word_list),len(predict_word_list))
                # print(predict_word_list)

                if self.args.metric_type == "best_pred":
                    suc_cnt = 0
                    for i in range(len(target_word_list)):
                        if target_word_list[i] == predict_word_list[i]:
                            suc_cnt += 1
                    batch_train_acc = suc_cnt / float(len(target_word_list))  # ACC
                elif self.args.metric_type == "n_best":
                    suc_cnt = 0
                    for i in range(len(target_word_list)):
                        if target_word_list[i] in predict_word_list[i]:
                            suc_cnt += 1
                    batch_train_acc = suc_cnt / float(len(target_word_list))  # ACC
                else:
                    assert 1 > 2, 'metric type not supported'


            else:  # MMLU
                choice_id_list = []
                for choice in self.args.label_dict.keys():
                    choice_id_list.append(self.args.tokenizer(choice).input_ids[-1])
                    _id = self.args.tokenizer(choice).input_ids[-1]
                enc = next_token_logits[:, choice_id_list]  # [bs, num_choice]
                enc_predict_prob = nn.functional.softmax(enc, dim=-1)  # [bs, num_choice]

                predict_label = torch.argmax(enc_predict_prob, dim=-1)  # [bs]
                actual_label = gt_one_hot_label  # torch.argmax(gt_one_hot_label, dim=-1)

                test_predict_labels.extend(predict_label.detach().cpu().tolist())
                test_actual_labels.extend(actual_label.detach().cpu().tolist())
                # test_full_predict_labels.extend( list(full_predict_label.detach().cpu()) )

                sample_cnt += predict_label.shape[0]
                suc_cnt += torch.sum(predict_label == actual_label).item()

            return loss.item(), batch_train_acc

    def start_train(self):
        for ik in range(self.k):
            self.parties[ik].prepare_data_loader()

        print_every = 1
        self.num_total_comms = 0
        total_time = 0.0
        flag = 0
        self.current_epoch = 0

        start_time = time.time()
        results = []
        for i_epoch in range(self.args.main_epochs):
            self.current_epoch = i_epoch
            postfix = {'train_loss': 0.0, 'train_acc': 0.0, 'test_acc': 0.0}
            i = -1

            for ik in range(self.k - 1):
                self.loss, self.train_acc = self.parties[ik].train(i_epoch)

            # validation
            if (i + 1) % print_every == 0:
                print("validate and test")

                _exp_result, self.test_acc = self.inference()

                postfix['train_loss'] = self.loss
                postfix['train_acc'] = '{:.2f}%'.format(self.train_acc * 100)
                postfix['test_acc'] = '{:.2f}%'.format(self.test_acc * 100)
                # postfix['test_auc'] = '{:.2f}%'.format(self.test_auc * 100)
                # postfix['test_mcc'] = '{:.2f}%'.format(self.test_mcc * 100)

                exp_result = 'Epoch {}% \t train_loss:{:.2f} train_acc:{:.2f} test_acc:{:.2f}'.format(
                    i_epoch, self.loss, self.train_acc, self.test_acc)
                print(exp_result)
                results.append(exp_result)
                self.final_epoch = i_epoch

        return results

    def train(self):

        print_every = 1

        for ik in range(self.k):
            self.parties[ik].prepare_data_loader()

        test_acc = 0.0
        # Early Stop
        last_loss = 1000000
        early_stop_count = 0
        LR_passive_list = []
        LR_active_list = []

        self.num_total_comms = 0
        total_time = 0.0
        flag = 0
        self.current_epoch = 0

        last_adversarial_model_loss = 10000
        start_time = time.time()

        data_record = pd.DataFrame(columns=['Epoch', 'train_loss', 'train_acc', 'test_acc'])
        for i_epoch in range(self.epochs):
            self.current_epoch = i_epoch
            postfix = {'train_loss': 0.0, 'train_acc': 0.0, 'test_acc': 0.0}
            i = -1
            # for ik in range(self.k - 1):
            #     self.loss, self.train_acc = self.parties[ik].train(i_epoch)
            print_every = 1
            total_time = 0

            data_loader_list = [self.parties[ik].train_loader for ik in range(self.k - 1)]
            for parties_data in zip(*data_loader_list):
                ############ Allocate Data #################
                # parties_data[0]:  bs *( data, label, mask, token_type_ids, feature(forQA))
                _parties_data = []
                for party_id in range(len(parties_data)):  # iter through each passive party
                    batch_input_ids = []
                    batch_label = []
                    batch_attention_mask = []
                    batch_token_type_ids = []
                    batch_feature = []
                    for bs_id in range(len(parties_data[party_id])):
                        # Input_ids
                        batch_input_ids.append(parties_data[party_id][bs_id][0].tolist())
                        # Attention Mask
                        batch_attention_mask.append(parties_data[party_id][bs_id][2].tolist())

                        # ptoken_type_ids
                        if parties_data[party_id][bs_id][3] == []:
                            batch_token_type_ids = None
                        else:
                            batch_token_type_ids.append(parties_data[party_id][bs_id][3].tolist())

                        # feature (for QuestionAnswering only)
                        if parties_data[party_id][bs_id][4] == []:
                            batch_feature = None
                        else:
                            batch_feature.append(parties_data[party_id][bs_id][4])

                        # Label
                        if type(parties_data[party_id][bs_id][1]) != str:
                            batch_label.append(parties_data[party_id][bs_id][1].tolist())
                        else:
                            batch_label.append(parties_data[party_id][bs_id][1])

                    batch_input_ids = torch.tensor(batch_input_ids).to(self.device)
                    batch_attention_mask = torch.tensor(batch_attention_mask).to(self.device)
                    if batch_token_type_ids != None:
                        batch_token_type_ids = torch.tensor(batch_token_type_ids).to(self.device)

                    if type(batch_label[0]) != str:
                        if self.args.task_type == 'QuestionAnswering':
                            # origin_batch_label_shape [bs, 2, num_of_answers]
                            # 2*bs, num_of_answers
                            # print('batch_label:',batch_label)
                            if type(batch_label[0][0]) == list:
                                batch_label = pad_sequence([torch.tensor(position_list) for sample_label in batch_label \
                                                            for position_list in sample_label], batch_first=True,
                                                           padding_value=-1).to(self.device)
                                origin_batch_label_shape = [int(batch_label.shape[0] / 2), 2, batch_label.shape[1]]
                                batch_label = batch_label.reshape(origin_batch_label_shape)
                            else:
                                batch_label = torch.tensor(batch_label).to(self.device)
                        else:
                            batch_label = torch.tensor(batch_label).to(self.device)

                    _parties_data.append(
                        [batch_input_ids, batch_label, batch_attention_mask, batch_token_type_ids, batch_feature])
                parties_data = _parties_data

                if self.args.task_type == "SequenceClassification" and self.num_classes > 1:  # classification
                    gt_one_hot_label = self.label_to_one_hot(parties_data[0][1], self.num_classes)
                else:
                    gt_one_hot_label = parties_data[0][1]
                self.gt_one_hot_label = gt_one_hot_label

                i += 1

                # passive party call active party global model to a training mode
                self.parties[0]._send_global_model_train_message()

                # ====== train batch (start) ======
                enter_time = time.time()
                self.loss, self.train_acc = self.train_batch(parties_data, gt_one_hot_label)
                exit_time = time.time()
                # ====== train batch (end) ======

                total_time += (exit_time - enter_time)
                self.num_total_comms = self.num_total_comms + 1
                # if self.num_total_comms % 10 == 0:
                #     print(f"total time for {self.num_total_comms} communication is {total_time}")
                self.current_step = self.current_step + 1
                del (parties_data)

            # LR decay
            self.LR_Decay(i_epoch)

            if self.args.apply_adversarial:
                print(
                    f'adversarial_model_loss:{self.parties[0].adversarial_model_loss.item()} adversary_attack_loss:{self.parties[0].adversary_attack_loss.item()}')
            if self.args.apply_mid:
                print(f'global_loss={self.parties[0].global_loss},mid_loss={self.parties[0].mid_loss}')

            # validation
            if (i + 1) % print_every == 0:
                print("validate and test")
                self.parties[self.k - 1].global_model.eval()

                with torch.no_grad():

                    _exp_result, self.test_acc = self.inference()

                    postfix['train_loss'] = self.loss
                    postfix['train_acc'] = '{:.2f}%'.format(self.train_acc * 100)
                    postfix['test_acc'] = '{:.2f}%'.format(self.test_acc * 100)
                    # postfix['test_auc'] = '{:.2f}%'.format(self.test_auc * 100)
                    # postfix['test_mcc'] = '{:.2f}%'.format(self.test_mcc * 100)

                    exp_result = 'Epoch {}% \t train_loss:{:.2f} train_acc:{:.2f} test_acc:{:.2f}'.format(
                        i_epoch, self.loss, self.train_acc, self.test_acc)
                    print(exp_result)

                    if (i + 1) % 10 == 0:
                        if self.args.apply_adversarial:
                            if last_adversarial_model_loss < self.parties[0].adversarial_model_loss.item():
                                self.final_epoch = i_epoch
                                break
                            last_adversarial_model_loss = self.parties[0].adversarial_model_loss.item()

                    self.final_epoch = i_epoch

            data_record.loc[len(data_record)] = [i_epoch, self.loss, self.train_acc, self.test_acc]

        exp_result = f'train_loss:{self.loss} train_acc:{self.train_acc} test_acc:{self.test_acc} final_epoch:{self.final_epoch}'

        # self.final_state = self.save_state() 
        # self.final_state.update(self.save_state(False)) 
        # self.final_state = self.save_state(False) 
        # self.final_state.update(self.save_party_data()) 

        result_path = f'exp_result/{self.args.dataset}/Q{str(self.args.Q)}/'
        # if not os.path.exists(result_path):
        #     os.makedirs(result_path)
        model_name = self.args.model_list[str(0)]["type"]  # .replace('/','-')
        if self.args.pipeline == 'pretrained':
            filename = f'{self.args.defense_name}_{self.args.defense_param},pretrained_model={self.args.model_list[str(0)]["type"]}'
        else:
            filename = f'{self.args.defense_name}_{self.args.defense_param},finetuned_model={self.args.model_list[str(0)]["type"]}'
        result_file_name = result_path + filename + f'.csv'
        print('Save csv to:', result_file_name)
        data_record.to_csv(result_file_name)

        # print('Final mid model:')
        # mark = 0
        # for name, param in self.parties[0].mid_model.named_parameters():
        #     if mark == 0:
        #         print(name, param)
        #         mark = mark + 1

        return exp_result, self.test_acc  # , self.stopping_iter, self.stopping_time, self.stopping_commu_cost

    def save_state(self, BEFORE_MODEL_UPDATE=True):
        if BEFORE_MODEL_UPDATE:
            return {
                "model": [copy.deepcopy(self.parties[ik].local_model) for ik in range(self.args.k)],
                "global_model": copy.deepcopy(self.parties[self.args.k - 1].global_model),
                "model_names": [str(type(self.parties[ik].local_model)).split('.')[-1].split('\'')[-2] for ik in
                                range(self.args.k)] + [
                                   str(type(self.parties[self.args.k - 1].global_model)).split('.')[-1].split('\'')[-2]]

            }
        else:
            return {
                # "model": [copy.deepcopy(self.parties[ik].local_model) for ik in range(self.args.k)]+[self.parties[self.args.k-1].global_model],
                "data": copy.deepcopy(self.parties_data),
                "label": copy.deepcopy(self.gt_one_hot_label),
                "predict": [copy.deepcopy(self.parties[ik].local_pred_clone) for ik in range(self.k)],
                "gradient": [copy.deepcopy(self.parties[ik].local_gradient) for ik in range(self.k)],
                "local_model_gradient": [copy.deepcopy(self.parties[ik].weights_grad_a) for ik in range(self.k)],
                "train_acc": copy.deepcopy(self.train_acc),
                "loss": copy.deepcopy(self.loss),
                "global_pred": self.parties[self.k - 1].global_pred,
                "final_model": [copy.deepcopy(self.parties[ik].local_model) for ik in range(self.args.k)],
                "final_global_model": copy.deepcopy(self.parties[self.args.k - 1].global_model),

            }

    def save_party_data(self):
        return {
            # "aux_data": [copy.deepcopy(self.parties[ik].aux_data) for ik in range(self.k)],
            # "train_data": [copy.deepcopy(self.parties[ik].train_data) for ik in range(self.k)],
            # "test_data": [copy.deepcopy(self.parties[ik].test_data) for ik in range(self.k)],

            # "aux_label": [copy.deepcopy(self.parties[ik].aux_label) for ik in range(self.k)],
            # "train_label": [copy.deepcopy(self.parties[ik].train_label) for ik in range(self.k)],
            # "test_label": [copy.deepcopy(self.parties[ik].test_label) for ik in range(self.k)],

            # "aux_attribute": [copy.deepcopy(self.parties[ik].aux_attribute) for ik in range(self.k)],
            # "train_attribute": [copy.deepcopy(self.parties[ik].train_attribute) for ik in range(self.k)],
            # "test_attribute": [copy.deepcopy(self.parties[ik].test_attribute) for ik in range(self.k)],

            "aux_loader": [self.parties[ik].aux_loader for ik in range(self.k)],
            "train_loader": [self.parties[ik].train_loader for ik in range(self.k)],
            "test_loader": [self.parties[ik].test_loader for ik in range(self.k)],

            "batchsize": self.args.batch_size,
            "num_classes": self.args.num_classes
        }

    def save_trained_models(self):
        dir_path = self.exp_res_dir + f'trained_models/parties{self.k}_topmodel{self.args.apply_trainable_layer}_epoch{self.epochs}/'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if self.args.apply_defense:
            file_path = dir_path + f'{self.args.defense_name}_{self.args.defense_configs}.pkl'
        else:
            file_path = dir_path + 'NoDefense.pkl'
        torch.save(([self.trained_models["model"][i].state_dict() for i in range(len(self.trained_models["model"]))],
                    self.trained_models["model_names"]),
                   file_path)

    def evaluate_attack(self):
        self.attacker = AttackerLoader(self, self.args)
        if self.attacker != None:
            attack_acc = self.attacker.attack()
        return attack_acc

    def launch_defense(self, gradients_list, _type):

        if _type == 'gradients':
            return apply_defense(self.args, _type, gradients_list)
        elif _type == 'pred':
            return apply_defense(self.args, _type, gradients_list)
        else:
            # further extention
            return gradients_list

    def calc_label_recovery_rate(self, dummy_label, gt_label):
        success = torch.sum(torch.argmax(dummy_label, dim=-1) == torch.argmax(gt_label, dim=-1)).item()
        total = dummy_label.shape[0]
        return success / total

    def reset_barrier(self):
        self._barrier = threading.Barrier(2)

    def _init_e2e_model(self):
        def communication_callback(_intermediate):
            if _intermediate is None:
                self._barrier.wait()
                print('============================================')
                _intermediate = self._last_result
            return _intermediate

        model_config = None
        if not self.args.task_type == 'DevLLMInference':
            return
        for party in self.parties:
            if party.local_model:
                model_config = party.local_model.config
                break
            elif party.global_model:
                model_config = party.global_model.config
                break
        if not model_config:
            logger.error(f"No model config for E2E_model")
        self.e2e_model = E2EModel(model_config, self.parties[0], self.parties[1], communication_callback,
                                  self.reset_barrier)
        # self.e2e_model.to(self.e2e_model.local_model.device)
