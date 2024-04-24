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

from sklearn.metrics import roc_auc_score, matthews_corrcoef
import scipy.stats as stats
import torch.nn as nn
import torch
import warnings

import inspect
from typing import List, Optional, Tuple, Union, Dict, Any

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from transformers.generation import GenerationMixin
from transformers.models.auto import (
    MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    MODEL_FOR_VISION_2_SEQ_MAPPING,
)
# from models.vision import resnet18, MLP2
from utils.basic_functions import cross_entropy_for_onehot, append_exp_res, multiclass_auc
from utils.communication_protocol_funcs import get_size_of

# from evaluates.attacks.attack_api import apply_attack

from utils.constants import *
import utils.constants as shared_var
from utils.marvell_functions import KL_gradient_perturb
from utils.noisy_label_functions import add_noise
from utils.noisy_sample_functions import noisy_sample
from utils.communication_protocol_funcs import compress_pred, Cache, ins_weight
from utils.squad_utils import normalize_answer, _get_best_indexes, get_tokens, compute_exact, compute_f1

from evaluates.defenses.defense_api import apply_defense
from evaluates.defenses.defense_functions import *
from evaluates.attacks.attack_api import AttackerLoader


from load.LoadModels import MODEL_PATH
from party.LocalCommunication import LocalCommunication

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True

STOPPING_ACC = {'mnist': 0.977, 'cifar10': 0.80, 'cifar100': 0.40, 'diabetes': 0.69, \
                'nuswide': 0.88, 'breast_cancer_diagnose': 0.88, 'adult_income': 0.84, 'cora': 0.72, \
                'avazu': 0.83, 'criteo': 0.74, 'nursery': 0.99, 'credit': 0.82, 'news20': 0.8, \
                'cola_public': 0.8, 'SST-2': 0.9}  # add more about stopping accuracy for different datasets when calculating the #communication-rounds needed

def create_main_task(global_model_type):
    print('inherited:',global_model_type)
    class MainTaskVFL_LLM( global_model_type): #GenerationMixin object, 
        def __init__(self, args):
            self.args = args
            ## generation related
            self.config  = args.config # model config
            self.generation_config  = args.generation_config
            self.k = args.k

            self.current_device = args.device
            self._device = args.parties[-1].global_model.device

            self.dataset_name = args.dataset

            self.epochs = args.main_epochs
            self.lr = args.main_lr
            self.batch_size = args.batch_size
            self.models_dict = args.model_list

            self.num_classes = args.num_classes
            self.exp_res_dir = args.exp_res_dir

            self.exp_res_path = args.exp_res_path
            self.parties = args.parties

            self.Q = args.Q  # FedBCD

            self.train_party_time = [0 for i in range(self.k)]
            self.inference_party_time = [0 for i in range(self.k)]


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
            self.training_time = 0

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

        @property
        def device(self):
            return self._device

        @device.setter
        def device(self, device):
            if device == self._device:
                print("\nYou are already watching device ", device)
            else:
                self._device = device
                print("\nYou are now watching device ", device)

        def label_to_one_hot(self, target, num_classes=10):
            target = torch.tensor(target)
            target = target.long()
            # print('label_to_one_hot:', target, type(target),type(target[0]))
            try:
                _ = target.size()[1]
                # print("use target itself", target.size())
                onehot_target = target.type(torch.float32).to(self.current_device)
            except:
                target = torch.unsqueeze(target, 1).to(self.current_device)
                # print("use unsqueezed target", target.size(),type(target))
                onehot_target = torch.zeros(target.size(0), num_classes, device=self.current_device)
                onehot_target.scatter_(1, target, 1)
            return onehot_target

        def init_communication(self, communication=None):
            if communication is None:
                communication = LocalCommunication(self.args.parties[self.args.k - 1])
            self._communication = communication

        def LR_Decay(self, i_epoch):
            # for ik in range(self.k):
            #     self.parties[ik].LR_decay(i_epoch)
            self._communication.send_global_lr_decay(i_epoch)

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

        def pred_transmit(self, use_cache=None, count_time=False):
            '''
            Active party gets pred from passive parties
            '''
            all_pred_list = []
            for ik in range(self.k - 1):
                start_time = time.time()
                result_dict = self.parties[ik].give_pred(use_cache=use_cache)  # use_cache=use_cache

                pred_detach = result_dict['inputs_embeds']

                # Defense
                if self.args.apply_defense:
                    if (ik in self.args.defense_configs['party']):
                        # print('Apply DP')
                        pred_detach = self.apply_defense_on_transmission(pred_detach)
                # Communication Process
                pred_detach = self.apply_communication_protocol_on_transmission(pred_detach)
                pred_clone = torch.autograd.Variable(pred_detach, requires_grad=True).to(self.args.device)
                # attention_mask = torch.autograd.Variable(attention_mask).to(self.args.device)

                result_dict['inputs_embeds'] = pred_clone
                # result_dict['attention_mask'] = attention_mask

                pred_list = result_dict 
                self.parties[self.k - 1].receive_pred(pred_list, ik)
                self.parties[ik].update_local_pred(pred_clone)

                all_pred_list.append(pred_list)
                
                end_time = time.time()
                if count_time=='train':
                    self.train_party_time[ik] += end_time-start_time
                elif count_time=='inference':
                    self.inference_party_time[ik] += end_time-start_time

            self.all_pred_list = all_pred_list
            return all_pred_list

        def global_pred_transmit(self, pred_list, use_cache=None, count_time=False):
            start_time = time.time()
            final_pred = self._communication.send_pred_message(pred_list,use_cache=use_cache)#  
            end_time = time.time()
            if count_time=='train':
                self.train_party_time[-1] += end_time-start_time
            elif count_time=='inference':
                self.inference_party_time[-1] += end_time-start_time
            return final_pred

        def local_gradient_transmit(self, count_time='train'):
            for ik in range(self.k - 1):
                if self.parties[ik].local_model_optimizer != None:
                    start_time = time.time()
                    passive_local_gradient = self.parties[self.k - 1].cal_passive_local_gradient(ik)
                    end_time = time.time()

                    if count_time=='train':
                        self.train_party_time[self.k - 1] += end_time - start_time

                    self.parties[ik].local_gradient = passive_local_gradient

        def global_gradient_transmit(self, final_pred, count_time='train'):
            start_time = time.time()
            global_loss = self.parties[0].cal_loss(final_pred)
            global_gradients = self.parties[0].cal_global_gradient(global_loss, final_pred)
            end_time = time.time()
            if count_time=='train':
                self.train_party_time[0] += end_time - start_time

            self.communication_cost += get_size_of(global_gradients)

            self.parties[self.k - 1].receive_loss_and_gradients(self.parties[0].global_gradients)  # self.parties[0].global_loss,
            # self.parties[self.k-1].global_loss = self.parties[0].global_loss
            # self.parties[self.k-1].global_gradients = self.parties[0].global_gradients

        def generate_result(self, test_logit, gt_one_hot_label, parties_data):
            test_preds = []
            test_targets = []
            test_predict_labels = []
            test_actual_labels = []
            target_word_list = []
            predict_word_list = []
            suc_cnt = 0
            sample_cnt = 0

            if self.args.model_architect=='CLS': #task_type == "SequenceClassification":
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

            elif self.args.model_architect=='CLM': #.task_type == "CausalLM":
                if self.args.task_type == "CausalLM":#dataset == "Lambada":
                    print('== generate result ==')
                    # next_token_logits = test_logit[:, -1]  # [bs, 32000]   test_logit: bs, seq_len, vocab_dim

                    predict_label_list = test_logit[:,self.seq_length:] # bs, new_token_num
                
                    predict_label_list = [int(_id.item()) for _id in list(predict_label_list)]

                    target_label_list = [int(_id.item()) for _id in list(gt_one_hot_label)]

                    # enc_predict_prob = nn.functional.softmax(next_token_logits, dim=-1)
                    # if self.args.metric_type == "best_pred":
                    #     predict_label = torch.argmax(enc_predict_prob, dim=-1)  # [bs]
                    #     predict_label_list = predict_label  # predict_word: bs * best_pred
                    #     # predict_word = [self.args.tokenizer.decode([_best_id]) for _best_id in predict_label_list.tolist()]
                    #     # predict_word = [normalize_answer(_p) for _p in predict_word]
                    #     # predict_word_list = predict_word  # predict_word: bs * best_pred
                    # elif self.args.metric_type == "n_best":
                    #     logit_list, index_list = torch.sort(enc_predict_prob, descending=True)
                    #     predict_label_list = index_list[:, :self.args.n_best_size]
                    #     # for _bs in range(predict_label.shape[0]):  # each batch
                    #     #     predict_word = [self.args.tokenizer.decode([_label]) for _label in predict_label[_bs].tolist()]
                    #     #     predict_word = [normalize_answer(_p) for _p in predict_word]
                    #     #     predict_word_list.append(predict_word)  # predict_word: list of n best for this batch
                    
                    print('predict_label_list:',predict_label_list)
                    print('target_label_list:',target_label_list)

                    return target_label_list, predict_label_list, None #target_word_list, predict_word_list, None
                elif self.args.task_type == "SequenceClassification":
                    print('test_logit:',type(test_logit),test_logit.shape)
                    print('self.seq_length:',self.seq_length)
                    new_token_logits = test_logit #[:,self.seq_length:]
                    print('new_token_logits:',type(new_token_logits),new_token_logits.shape)

                    for _i in range(new_token_logits.shape[0]):
                        model_output_ids = new_token_logits[_i]
                        print('text:',self.args.tokenizer.decode(model_output_ids, skip_special_tokens=True))
                    
                    target_label_list = [int(_id.item()) for _id in list(gt_one_hot_label)]
                    print('target_label_list:',target_label_list,type(target_label_list[0]))
                    assert 1>2
                    predict_label_list = torch.argmax(enc_predict_prob, dim=-1)  # [bs]
                    predict_label_list = predict_label  # predict_word: bs * best_pred

                    # if ('positive' in model_outputs[i] or 'pos' in model_outputs[i]) and ('neg' not in model_outputs[i]):
                    #     model_prediction = 'positive'
                    # elif ('negative' in model_outputs[i] or 'neg' in model_outputs[i]) and ('pos' not in model_outputs[i]):
                    #     model_prediction = 'negative'

                # else:  # MMLU
                #     choice_id_list = []
                #     probs = []
                #     for choice in self.args.label_dict.keys():
                #         choice_id = self.args.tokenizer.convert_tokens_to_ids(choice)
                #         choice_id_list.append( choice_id )
                #         probs.append( next_token_logits[:, choice_id] )
                    
                #     predict_prob = torch.stack(probs,dim = -1)
                #     # enc = next_token_logits[:, choice_id_list]  # [bs, num_choice]
                #     predict_prob = nn.functional.softmax(predict_prob, dim=-1)  # [bs, num_choice]

                #     predict_label = torch.argmax(predict_prob, dim=-1)  # [bs]
                #     actual_label = gt_one_hot_label 

                #     target_label_list = actual_label.detach().cpu().tolist()
                #     predict_label_list = predict_label.detach().cpu().tolist()
                #     # print('predict_label:',predict_label)
                #     # print('actual_label:',actual_label)

                #     # test_predict_labels.extend(predict_label.detach().cpu().tolist())
                #     # test_actual_labels.extend(actual_label.detach().cpu().tolist())

                #     sample_cnt = predict_label.shape[0]
                #     suc_cnt = torch.sum(predict_label == actual_label).item()

                #     return target_label_list, predict_label_list, sample_cnt 


            elif self.args.model_architect=='TQA': #.task_type == "QuestionAnswering":
                start_logits = test_logit.start_logits # bs, 512
                end_logits = test_logit.end_logits # bs, 512
                sample_cnt = start_logits.shape[0] # bs

                n_best_size = self.args.n_best_size
                start_indexes = [_get_best_indexes(_logits, n_best_size) for _logits in start_logits]
                end_indexes = [_get_best_indexes(_logits, n_best_size) for _logits in end_logits]
                # start_indexes: list bs * n_nest_size [nbest start index]

                exact_score_list = []
                f1_list = []
                batch_nbest_list = []
                batch_gold_ans_list = []
                for i in range(start_logits.shape[0]):  # for each sample in this batch
                    
                    ############ Gold ################
                    feature = parties_data[0][i]['feature']  # print('parties_data[0][4]:',type(parties_data[0][4]),'feature:',type(feature))
                    # feature_tokens = [_token for _token in feature["tokens"]]  # [_token[0] for _token in feature["tokens"]]
                    
                    gold_start_indexs, gold_end_indexs = gt_one_hot_label[i]  # the i'th sample in a batch
                    if len(gold_start_indexs.shape) == 0:
                        gold_start_indexs = gold_start_indexs.unsqueeze(0)
                    if len(gold_end_indexs.shape) == 0:
                        gold_end_indexs = gold_end_indexs.unsqueeze(0)
                    
                    gold_ans = []  # gold answers for this sample
                    for _i in range(len(gold_start_indexs)):
                        gold_start_index = int(gold_start_indexs[_i])
                        gold_end_index = int(gold_end_indexs[_i])

                        # if gold_start_index == -1:
                        #     continue
                        # gold_ans_text = " ".join(feature_tokens[gold_start_index:(gold_end_index + 1)])
                        # gold_ans_text = normalize_answer(gold_ans_text)

                        gold_ans_text = list(range(gold_start_index,gold_end_index+1))
                        gold_ans.append(gold_ans_text)
                    
                    batch_gold_ans_list.append(gold_ans)

                    ############ Pred ################
                    _start_logits = start_logits[i]
                    _end_logits = end_logits[i]
                    _start_indexes = start_indexes[i] #[nbest start index] list  n_best_size 
                    _end_indexes = end_indexes[i] #[nbest end index] list  n_best_size

                    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                        "PrelimPrediction",
                        ["start_index", "end_index", "start_logit", "end_logit"])
                    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                        "NbestPrediction", ["text", "start_logit", "end_logit"])
                    # throw out all invalid predictions.
                    prelim_predictions = []
                    for start_index in _start_indexes:
                        for end_index in _end_indexes:
                            # We could hypothetically create invalid predictions, e.g., predict
                            # that the start of the span is in the question. We throw out all
                            # invalid predictions.
                            if start_index >= feature['len_tokens']: #len(feature["tokens"]):
                                continue
                            if end_index >= feature['len_tokens']: #len(feature["tokens"]):
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
                        reverse=True) # length=2
                    # print('prelim_predictions:',len(prelim_predictions)) 

                    exact_score = 0
                    f1 = 0
                    nbest = [] # Get n best prediction text
                    n_best_size = min(n_best_size, len(prelim_predictions))
                    for _id in range(n_best_size):
                        start_index = prelim_predictions[_id].start_index
                        end_index = prelim_predictions[_id].end_index
                        
                        # pred_ans_text = " ".join(feature_tokens[start_index:(end_index + 1)])
                        # pred_ans_text = normalize_answer(pred_ans_text)

                        pred_ans_text = list(range(start_index,end_index+1))
                        nbest.append(
                            _NbestPrediction(
                                text=pred_ans_text,
                                start_logit=prelim_predictions[_id].start_logit,
                                end_logit=prelim_predictions[_id].end_logit))
                    
                    batch_nbest_list.append(nbest)

                # print('batch_nbest_list:',batch_nbest_list)
                # print('batch_gold_ans_list:',batch_gold_ans_list)

                return batch_nbest_list, batch_gold_ans_list, sample_cnt

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
                    for party_id in range(len(parties_data)):  # parties_data[party_id]: list of bs
                        batch_input_dicts = []
                        batch_label = []
                        for bs_id in range(len(parties_data[party_id])):
                            # Input Dict
                            batch_input_dicts.append(parties_data[party_id][bs_id][0])
                            # Label
                            batch_label.append(parties_data[party_id][bs_id][1])
                        _parties_data.append([batch_input_dicts, batch_label])
                    parties_data = _parties_data

                    if self.args.task_type == "SequenceClassification" and self.args.num_classes > 1:  # classification
                        gt_one_hot_label = self.label_to_one_hot(parties_data[0][1], self.args.num_classes)
                    elif self.args.task_type == "QuestionAnswering":
                        gt_one_hot_label = list(parties_data[0][1])
                    else:
                        gt_one_hot_label = parties_data[0][1]

                    data_inputs = {}
                    for key_name in parties_data[0][0][0].keys():
                        if key_name in ['feature']:
                            continue
                        data_inputs[key_name] = torch.stack( [parties_data[0][0][i][key_name] for i in range(len(parties_data[0][0]))] )
                    
                    self.seq_length = data_inputs['input_ids'].shape[-1]
                    
                    print('self.args.model_architect:',self.args.model_architect)
                    # test_logit -> standard output for each task
                    if self.args.model_architect=='CLS': #task_type == "SequenceClassification":  # and self.args.num_classes > 1: # classification
                        global_output = self.forward(**data_inputs)
                        test_logit = global_output.logits
                        batch_predict_label, batch_actual_label, sample_cnt = self.generate_result(test_logit, gt_one_hot_label, parties_data)
                        predict_label_list.extend(batch_predict_label)
                        actual_label_list.extend(batch_actual_label)
                        if sample_cnt is not None:
                            total_sample_cnt += sample_cnt
                    elif self.args.model_architect=='TQA': #task_type == "QuestionAnswering":
                        global_output = self.forward(**data_inputs)
                        test_logit = global_output
                        batch_nbest, batch_gold_ans, sample_cnt = self.generate_result(test_logit, gt_one_hot_label, parties_data)
                        nbest_list.extend(batch_nbest)
                        gold_ans_list.extend(batch_gold_ans)
                        if sample_cnt is not None:
                            total_sample_cnt += sample_cnt
                    elif self.args.model_architect=='CLM': #task_type == "CausalLM":
                        # test_logit = global_output.logits
                        # batch_target_word, batch_predict_word, sample_cnt = self.generate_result(test_logit, gt_one_hot_label, parties_data)
                        
                        generation_output = self.generate(**data_inputs, \
                        generation_config = self.generation_config,max_new_tokens=1)
                        self._clear_past_key_values()
                        print('generation_output:',type(generation_output),generation_output.shape)
                        
                        batch_target_word, batch_predict_word, sample_cnt = self.generate_result(generation_output, gt_one_hot_label, parties_data)
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
                print('target_word_list:',len(target_word_list),target_word_list[:5])
                print('predict_word_list:',predict_word_list[:5])
                print('total_sample_cnt:',total_sample_cnt)
                return target_word_list, predict_word_list, total_sample_cnt
            else:
                assert 1 > 2, 'Task type not supported'
    
        def seq_inference(self):
            # SequenceClassification / Regression
            postfix = {'test_acc': 0.0}

            predict_labels, actual_labels, total_sample_cnt = self.predict()

            # prediction result assessment
            if self.num_classes == 1:
                self.test_mse = torch.mean(
                    (torch.tensor(predict_labels) - torch.tensor(actual_labels)) ** 2).item()
                self.test_pearson_corr = stats.pearsonr(torch.tensor(predict_labels), torch.tensor(actual_labels))[0]
                self.test_spearmanr_corr = stats.spearmanr(torch.tensor(predict_labels), torch.tensor(actual_labels))[0]
                postfix['test_mse'] = '{:.4f}%'.format(self.test_mse * 100)
                postfix['test_pearson_corr'] = '{:.4f}%'.format(self.test_pearson_corr * 100)
                exp_result = '|test_mse={:.4f}|test_pearson_corr={:.4f}|test_spearmanr_corr={:.4f}' \
                    .format(self.test_mse, self.test_pearson_corr, self.test_spearmanr_corr)
                print(exp_result)
                return exp_result, [self.test_mse, self.test_pearson_corr, self.test_spearmanr_corr]
            else:
                # print('predict_labels:',predict_labels[:10])
                # print('actual_labels:',actual_labels[:10])

                suc_cnt = torch.sum(torch.tensor(predict_labels) == \
                                    torch.tensor(actual_labels)).item()
                self.test_acc = suc_cnt / float(total_sample_cnt)  # ACC
                self.test_mcc = matthews_corrcoef(np.array(predict_labels), np.array(actual_labels))  # MCC
                postfix['test_acc'] = '{:.2f}%'.format(self.test_acc * 100)
                postfix['test_mcc'] = '{:.2f}%'.format(self.test_mcc * 100)
                exp_result = '|test_acc={:.2f}|test_mcc={:.2f}'.format(self.test_acc, self.test_mcc)
                print(exp_result)
                return exp_result, self.test_acc

        def causal_lm_inference(self):
            postfix = {'test_acc': 0.0}
            print('=== causal_lm_inference ===')

            target_word_list, predict_word_list, total_sample_cnt = self.predict()
            # target_label_list, predict_label_list, None 

            print('target_word_list:\n', target_word_list[:5])
            print('predict_word_list:\n', predict_word_list[:5])
            print('total_sample_cnt:',total_sample_cnt)

            # prediction result assessment
            if self.args.metric_type == "best_pred":
                suc_cnt = 0
                for i in range(len(target_word_list)):
                    if target_word_list[i] == predict_word_list[i]:
                        suc_cnt += 1
                self.test_acc = suc_cnt / float(len(target_word_list))
            elif self.args.metric_type == "n_best":
                suc_cnt = 0
                for i in range(len(target_word_list)):
                    if target_word_list[i] in predict_word_list[i]:
                        suc_cnt += 1
                self.test_acc = suc_cnt / float(len(target_word_list))  # ACC
            else:
                assert 1 > 2, 'metric type not supported'

            postfix['test_acc'] = '{:.2f}%'.format(self.test_acc * 100)

            exp_result = '|test_acc={:.2f}'.format(self.test_acc)
            print(exp_result)

            return exp_result, self.test_acc

        def qa_inference(self):
            # QA
            start_time = time.time()
            nbest_list, gold_ans_list, total_sample_cnt = self.predict()
            end_time = time.time()
            print('predict:',end_time-start_time)

            start_time = time.time()
            # prediction result assessment
            # total_scores = []
            best_non_null_entry = None
            exact_score_list = []
            f1_list = []
            for nbest, gold_ans in zip(nbest_list, gold_ans_list): # iterate through each sample
                exact_score = 0
                f1 = 0
                best_non_null_entry = None
                if self.args.metric_type == "best_pred":
                    for entry in nbest:
                        # total_scores.append(entry.start_logit + entry.end_logit)
                        if not best_non_null_entry:
                            if entry.text:
                                best_non_null_entry = entry
                    pred_ans_text = best_non_null_entry.text if (best_non_null_entry != None) else ""
                    exact_score = max(compute_exact(a, pred_ans_text) for a in gold_ans)
                    f1 = max(compute_f1(a, pred_ans_text) for a in gold_ans)
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

            end_time = time.time()
            print('assess:',end_time-start_time)

            exact_score = np.mean(exact_score_list)
            f1 = np.mean(f1_list)
            exp_result = '|exact_score={:.4f}|f1={:.4f}'.format(exact_score, f1)

            self.test_acc = exact_score
            print(exp_result)
            return exp_result, self.test_acc

        def forward(self, **kwargs):
            self.parties[0].obtain_local_data(kwargs)

            # passive party do local pred
            pred_list = self.pred_transmit(use_cache=True)

            # passive party inform active party to do global pred
            self.global_pred_transmit(pred_list, use_cache=True)
            final_output = self.parties[1].global_output
            return final_output

        def inference(self, inference_data='test'):
            # set inference time back to 0
            self.inference_party_time = [0 for i in range(self.k)]

            # print(' ========= Inference ==========')
            for ik in range(self.k - 1):
                self.parties[ik].prepare_data_loader()
                self.parties[ik].local_model.eval()
            self.parties[self.k - 1].global_model.eval()

            if self.args.model_architect == 'TQA': #task_type == "QuestionAnswering":
                exp_result, main_task_result = self.qa_inference()
                self.final_state = self.save_state()
                # self.final_state.update(self.save_state(False))
                self.final_state.update(self.save_party_data())
                exp_result = f'|inference_party_time={self.inference_party_time}'+exp_result
                return exp_result, main_task_result

            if self.args.model_architect=='CLS':#task_type == "SequenceClassification":
                # exp_result, self.test_acc =
                exp_result, main_task_result = self.seq_inference()
                self.final_state = self.save_state()
                # self.final_state.update(self.save_state(False))
                self.final_state.update(self.save_party_data())
                exp_result = f'|inference_party_time={self.inference_party_time}'+exp_result
                return exp_result, main_task_result

            if self.args.model_architect=='CLM':#task_type == "CausalLM":
                print('=== inference ===')
                exp_result, main_task_result = self.causal_lm_inference()
                self.final_state = self.save_state()
                # self.final_state.update(self.save_state(False))
                self.final_state.update(self.save_party_data())
                exp_result = f'|inference_party_time={self.inference_party_time}'+exp_result
                return exp_result, main_task_result

        def train_batch(self, parties_data, batch_label):
            '''
            batch_label: self.gt_one_hot_label   may be noisy
                QA: bs * [start_position, end_position]
            '''
            print('== train_batch ==')
            ############### allocate data ###############
            gt_one_hot_label = batch_label
            self.gt_one_hot_label = gt_one_hot_label
            for ik in range(self.k - 1):
                # # allocate data (data/label/attention_mask/token_type_ids)
                data_inputs = {}
                for key_name in parties_data[ik][0][0].keys():
                    data_inputs[key_name] = torch.stack( [parties_data[ik][0][i][key_name] for i in range(len(parties_data[ik][0]))] )
                self.parties[ik].obtain_local_data(data_inputs)
                self.parties[ik].gt_one_hot_label = gt_one_hot_label

            ################ normal vertical federated learning ################
            # torch.autograd.set_detect_anomaly(True)
            # =================== Commu ===================
            # Passive Party -> pred_list[local pred]
            pred_list = self.pred_transmit(count_time = 'train')
            # pred_list[local pred] -> Active Party -> test_logit[final pred]
            final_pred = self.global_pred_transmit(pred_list, count_time = 'train')

            # generation_output = self.generate(**data_inputs, \
            #     generation_config = self.generation_config,max_new_tokens=1)
            # self._clear_past_key_values()
            # print('generation_output:',type(generation_output),generation_output.shape)
                        

            # passive party -> global gradient -> active party
            self.global_gradient_transmit(final_pred, count_time = 'train')
            # active party -> local gradient -> passive party
            self.local_gradient_transmit(count_time = 'train')

            # ============= Model Update =============
            start_time = time.time()
            self.args.parties[self.args.k - 1].global_backward()  # self._send_global_backward_message()
            end_time = time.time()
            self.train_party_time[self.args.k - 1] += end_time - start_time
            
            for ik in range(self.k - 1):
                start_time = time.time()
                self.parties[ik].local_backward()
                end_time = time.time()
                self.train_party_time[ik] += end_time - start_time

            ################ normal vertical federated learning ################

            # print train_acc each batch
            if self.args.task_type == 'QuestionAnswering':
                pred = self.parties[self.k - 1].global_pred  # QuestionAnsweringModelOutput
                loss = self.parties[0].global_loss

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
                if self.args.num_classes == 1:
                    real_batch_label = batch_label.cpu().detach()
                    pred = final_pred.cpu().detach().squeeze()
                    # print('real_batch_label:',real_batch_label[:5])
                    # print('pred:',pred[:5])

                    batch_mse = torch.mean((torch.tensor(pred) - torch.tensor(real_batch_label)) ** 2).item()
                    batch_pearson_corr = stats.pearsonr(torch.tensor(pred), torch.tensor(real_batch_label))[0]
                    batch_test_spearmanr_corr = stats.spearmanr(torch.tensor(pred), torch.tensor(real_batch_label))[0]

                    # predict_prob = F.softmax(pred, dim=-1)

                    # suc_cnt = torch.sum(torch.argmax(predict_prob, dim=-1) == torch.argmax(real_batch_label, dim=-1)).item()
                    # batch_train_acc = suc_cnt / predict_prob.shape[0]

                    loss = self.parties[0].global_loss

                    return loss.item(), [batch_mse, batch_pearson_corr, batch_test_spearmanr_corr]
                else:
                    real_batch_label = batch_label

                    pred = final_pred
                    predict_prob = F.softmax(pred, dim=-1)

                    suc_cnt = torch.sum(torch.argmax(predict_prob, dim=-1) == torch.argmax(real_batch_label, dim=-1)).item()
                    batch_train_acc = suc_cnt / predict_prob.shape[0]

                    loss = self.parties[0].global_loss

                    return loss.item(), batch_train_acc

            elif self.args.task_type == 'CausalLM':
                pred = self.parties[self.k - 1].global_pred  # logits
                loss = self.parties[0].global_loss
                test_logit = pred
                next_token_logits = test_logit[:,-1]  # [bs, 32000] # print('next_token_logits:',next_token_logits.shape,next_token_logits)

                if self.args.dataset == "Lambada":
                    # print('gt_one_hot_label:',type(gt_one_hot_label),gt_one_hot_label)
                    target_label_list = [int(_p) for _p in gt_one_hot_label]

                    # predict_word_list : bs * predicted words
                    enc_predict_prob = nn.functional.softmax(next_token_logits, dim=-1)
                    if self.args.metric_type == "best_pred":
                        predict_label_list = torch.argmax(enc_predict_prob, dim=-1)  # [bs]
                    elif self.args.metric_type == "n_best":
                        logit_list, index_list = torch.sort(enc_predict_prob, descending=True)
                        # print('index_list:',index_list.shape)
                        predict_label_list = index_list[:, :self.args.n_best_size]
                    
                    if self.args.metric_type == "best_pred":
                        suc_cnt = 0
                        for i in range(len(target_label_list)):
                            if target_label_list[i] == predict_label_list[i]:
                                suc_cnt += 1
                        batch_train_acc = suc_cnt / float(len(target_label_list))  # ACC
                    elif self.args.metric_type == "n_best":
                        suc_cnt = 0
                        for i in range(len(target_label_list)):
                            if target_label_list[i] in predict_label_list[i]:
                                suc_cnt += 1
                        batch_train_acc = suc_cnt / float(len(target_label_list))  # ACC
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

                    # test_predict_labels = predict_label.detach().cpu().tolist()
                    # test_actual_labels = actual_label.detach().cpu().tolist()
                    # test_full_predict_labels.extend( list(full_predict_label.detach().cpu()) )

                    sample_cnt += predict_label.shape[0]
                    suc_cnt += torch.sum(predict_label == actual_label).item()

                return loss.item(), batch_train_acc

        def train(self):

            print_every = 1

            for ik in range(self.k):
                self.parties[ik].prepare_data_loader()

            test_acc = 0.0
            # Early Stop
            early_stop_threshold = self.args.early_stop_threshold
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
                    _parties_data = []
                    for party_id in range(len(parties_data)):  # parties_data[party_id]: list of bs
                        batch_input_dicts = []
                        batch_label = []
                        for bs_id in range(len(parties_data[party_id])):
                            # Input Dict
                            batch_input_dicts.append(parties_data[party_id][bs_id][0])
                             # Label
                            if type(parties_data[party_id][bs_id][1]) != str:
                                batch_label.append(parties_data[party_id][bs_id][1].tolist())
                            else:
                                batch_label.append(parties_data[party_id][bs_id][1])
                        _parties_data.append([batch_input_dicts, batch_label])
                    parties_data = _parties_data

                    if self.args.task_type == "SequenceClassification" and self.num_classes > 1:  # classification
                        gt_one_hot_label = self.label_to_one_hot(parties_data[0][1], self.num_classes)
                    else:
                        gt_one_hot_label = parties_data[0][1]
                    self.gt_one_hot_label = gt_one_hot_label

                    i += 1

                    # passive party call active party global model to a training mode
                    self._communication.send_global_model_train_message()

                    # ====== train batch (start) ======
                    enter_time = time.time()
                    self.loss, self.train_acc = self.train_batch(parties_data, gt_one_hot_label)
                    exit_time = time.time()
                    total_time += (exit_time - enter_time)
                    # ====== train batch (end) ======
                    self.num_total_comms = self.num_total_comms + 1
                    # if self.num_total_comms % 10 == 0:
                    #     print(f"total time for {self.num_total_comms} communication is {total_time}")
                    self.current_step = self.current_step + 1

                    del (parties_data)

                # LR decay
                self.LR_Decay(i_epoch)

                if self.args.apply_adversarial:
                    print(
                        f'global_loss={self.parties[0].global_loss} adversarial_model_loss:{self.parties[0].adversarial_model_loss.item()} adversary_attack_loss:{self.parties[0].adversary_attack_loss.item()}')
                if self.args.apply_mid:
                    print(f'global_loss={self.parties[0].global_loss},mid_loss={self.parties[0].mid_loss}')

                # validation
                if (i + 1) % print_every == 0:
                    print("validate and test")
                    self.parties[self.k - 1].global_model.eval()

                    with torch.no_grad():

                        _exp_result, self.test_acc = self.inference()

                        postfix['train_loss'] = self.loss
                        # postfix['train_acc'] = '{:.2f}%'.format(self.train_acc * 100)
                        # postfix['test_acc'] = '{:.2f}%'.format(self.test_acc * 100)
                        # postfix['test_auc'] = '{:.2f}%'.format(self.test_auc * 100)
                        # postfix['test_mcc'] = '{:.2f}%'.format(self.test_mcc * 100)

                        if self.args.task_type == 'SequenceClassification' and self.args.num_classes == 1:
                            exp_result = 'Epoch {}% \t train_loss:{:.2f} train_mse:{:.2f} test_mse:{:.2f}'.format(
                                i_epoch, self.loss, self.train_acc[0], self.test_acc[0])
                        else:
                            exp_result = 'Epoch {}% \t train_loss:{:.2f} train_acc:{:.2f} test_acc:{:.2f}'.format(
                                i_epoch, self.loss, self.train_acc, self.test_acc)
                        print(exp_result)

                        self.final_epoch = i_epoch + 1

                data_record.loc[len(data_record)] = [i_epoch, self.loss, self.train_acc, self.test_acc]

                # Early Stop
                if self.loss >= last_loss:
                    early_stop_count = early_stop_count + 1
                if early_stop_count >= early_stop_threshold:
                    self.final_epoch = i_epoch + 1
                    break
                last_loss = min(last_loss,self.loss)

            self.training_time = total_time
            if self.args.task_type == 'SequenceClassification' and self.args.num_classes == 1:
                exp_result = f'|train_party_time={self.train_party_time}|training_time={total_time}|train_loss:{self.loss}|\
                train_mse={self.train_acc[0]}|train_pearson_corr={self.train_acc[1]}|train_spearmanr_corr={self.train_acc[2]}|\
                test_mse={self.test_acc[0]}|train_pearson_corr={self.test_acc[1]}|test_spearmanr_corr={self.test_acc[2]}|\
                final_epoch={self.final_epoch}'
            else:
                exp_result = f'|train_party_time={self.train_party_time}|training_time={total_time}|train_loss={self.loss}|train_acc={self.train_acc}|\
                test_acc={self.test_acc}|final_epoch={self.final_epoch}'

            self.final_state = self.save_state()
            self.final_state.update(self.save_state(False))
            self.final_state.update(self.save_party_data())

            # self.final_state = self.save_state(False)
            # self.final_state.update(self.save_party_data())

            result_path = f'exp_result/{self.args.dataset}/Q{str(self.args.Q)}/'
            model_name = self.args.model_list[str(0)]["type"]  # .replace('/','-')
            if self.args.pipeline == 'pretrained':
                filename = f'{self.args.defense_name}_{self.args.defense_param},pretrained_model={self.args.model_list[str(0)]["type"]}'
            else:
                filename = f'{self.args.defense_name}_{self.args.defense_param},finetuned_model={self.args.model_list[str(0)]["type"]}'
            result_file_name = result_path + filename + f'.csv'
            print('Save csv to:', result_file_name)
            data_record.to_csv(result_file_name)

            return exp_result, self.test_acc, total_time  # , self.stopping_iter, self.stopping_time, self.stopping_commu_cost

        def save_state(self, BEFORE_MODEL_UPDATE=True):
            if BEFORE_MODEL_UPDATE:
                return {
                    "model": [copy.deepcopy(self.parties[ik].local_model) for ik in range(self.args.k)],
                    "global_model": copy.deepcopy(self.parties[self.args.k - 1].global_model),
                    "model_names": [str(type(self.parties[ik].local_model)).split('.')[-1].split('\'')[-2] for ik in range(self.args.k)] + [
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
                "test_data": [copy.deepcopy(self.parties[ik].test_data) for ik in range(self.k)],

                # "aux_dst": [self.parties[ik].aux_dst for ik in range(self.k)],
                # "train_dst": [self.parties[ik].train_dst for ik in range(self.k)],
                # "test_dst": [self.parties[ik].test_dst for ik in range(self.k)],

                # "aux_label": [copy.deepcopy(self.parties[ik].aux_label) for ik in range(self.k)],
                # "train_label": [copy.deepcopy(self.parties[ik].train_label) for ik in range(self.k)],
                "test_label": [copy.deepcopy(self.parties[ik].test_label) for ik in range(self.k)],
                
                # "aux_attribute": [copy.deepcopy(self.parties[ik].aux_attribute) for ik in range(self.k)],
                # "train_attribute": [copy.deepcopy(self.parties[ik].train_attribute) for ik in range(self.k)],
                # "test_attribute": [copy.deepcopy(self.parties[ik].test_attribute) for ik in range(self.k)],
                
                # "aux_loader": [self.parties[ik].aux_loader for ik in range(self.k)],
                # "train_loader": [self.parties[ik].train_loader for ik in range(self.k)],
                # "test_loader": [self.parties[ik].test_loader for ik in range(self.k)],
                
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

        def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):            
            """Validates model kwargs for generation. Generate argument typos will also be caught here."""
            # If a `Cache` instance is passed, checks whether the model is compatible with it
            if isinstance(model_kwargs.get("past_key_values", None), Cache) and not self._supports_cache_class:
                raise ValueError(
                    f"{self.__class__.__name__} does not support an instance of `Cache` as `past_key_values`. Please "
                    "check the model documentation for supported cache formats."
                )

            # Excludes arguments that are handled before calling any model function
            if self.config.is_encoder_decoder:
                for key in ["decoder_input_ids"]:
                    model_kwargs.pop(key, None)

            unused_model_args = []
            model_args = set(inspect.signature(self.prepare_inputs_for_generation).parameters)

            # `kwargs`/`model_kwargs` is often used to handle optional forward pass inputs like `attention_mask`. If
            # `prepare_inputs_for_generation` doesn't accept them, then a stricter check can be made ;)
            if "kwargs" in model_args or "model_kwargs" in model_args:
                model_args |= set(inspect.signature(self.parties[-1].global_model.forward).parameters)

            # Encoder-Decoder models may also need Encoder arguments from `model_kwargs`
            if self.config.is_encoder_decoder:
                base_model = getattr(self, self.base_model_prefix, None)

                # allow encoder kwargs
                encoder = getattr(self, "encoder", None)
                # `MusicgenForConditionalGeneration` has `text_encoder` and `audio_encoder`.
                # Also, it has `base_model_prefix = "encoder_decoder"` but there is no `self.encoder_decoder`
                # TODO: A better way to handle this.
                if encoder is None and base_model is not None:
                    encoder = getattr(base_model, "encoder", None)

                if encoder is not None:
                    encoder_model_args = set(inspect.signature(encoder.forward).parameters)
                    model_args |= encoder_model_args

                # allow decoder kwargs
                decoder = getattr(self, "decoder", None)
                if decoder is None and base_model is not None:
                    decoder = getattr(base_model, "decoder", None)

                if decoder is not None:
                    decoder_model_args = set(inspect.signature(decoder.forward).parameters)
                    model_args |= {f"decoder_{x}" for x in decoder_model_args}

                # allow assistant_encoder_outputs to be passed if we're doing assisted generating
                if "assistant_encoder_outputs" in model_kwargs:
                    model_args |= {"assistant_encoder_outputs"}

            for key, value in model_kwargs.items():
                if value is not None and key not in model_args:
                    unused_model_args.append(key)

            if unused_model_args:
                raise ValueError(
                    f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
                    " generate arguments will also show up in this list)"
                )

        def _validate_model_class(self):
            pass
            
            # if not self.parties[-1].global_model.can_generate():
            #     generate_compatible_mappings = [
            #         MODEL_FOR_CAUSAL_LM_MAPPING,
            #         MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
            #         MODEL_FOR_VISION_2_SEQ_MAPPING,
            #         MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
            #         MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
            #     ]
            #     generate_compatible_classes = set()
            #     for model_mapping in generate_compatible_mappings:
            #         supported_models = model_mapping.get(type(self.config), default=None)
            #         if supported_models is not None:
            #             generate_compatible_classes.add(supported_models.__name__)
            #     exception_message = (
            #         f"The current model class ({self.parties[-1].global_model.__class__.__name__}) is not compatible with `.generate()`, as "
            #         "it doesn't have a language model head."
            #     )
            #     if generate_compatible_classes:
            #         exception_message += f" Please use one of the following classes instead: {generate_compatible_classes}"
            #     raise TypeError(exception_message)

        def prepare_inputs_for_generation(self, input_ids, **model_kwargs):
            return self.parties[-1].global_model.prepare_inputs_for_generation(input_ids=input_ids, **model_kwargs)

        def __call__(self, **kwargs):
            return self.forward(**kwargs)

        def _clear_past_key_values(self):
            for i in range(self.k-1):
                self.parties[i].local_model._clear_past_key_values()
            self.parties[-1].global_model._clear_past_key_values()
    
    return MainTaskVFL_LLM