import gc
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
from accelerate import init_empty_weights
import pickle

from sklearn.metrics import roc_auc_score, matthews_corrcoef
import scipy.stats as stats
import torch.nn as nn
import torch


import inspect
from typing import List, Optional, Tuple, Union, Dict, Any
from torch.utils.tensorboard import SummaryWriter

import warnings
from typing import List, Optional, Tuple, Union

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

from loguru import logger
from evaluates.defenses.defense_api import apply_defense
from evaluates.defenses.defense_functions import *
from evaluates.attacks.attack_api import AttackerLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from config import vfl_basic_config, _new_pipeline

from load.LoadModels import QuestionAnsweringModelOutput
from models.llm_models.qwen2 import E2EModel
from party.LocalCommunication import LocalCommunication

import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True

STOPPING_ACC = {'mnist': 0.977, 'cifar10': 0.80, 'cifar100': 0.40, 'diabetes': 0.69, \
                'nuswide': 0.88, 'breast_cancer_diagnose': 0.88, 'adult_income': 0.84, 'cora': 0.72, \
                'avazu': 0.83, 'criteo': 0.74, 'nursery': 0.99, 'credit': 0.82, 'news20': 0.8, \
                'cola_public': 0.8,
                'SST-2': 0.9}  # add more about stopping accuracy for different datasets when calculating the #communication-rounds needed


def create_main_task(global_model_type):
    # todo: when 3slice inherit from passive party
    print('inherited:',global_model_type)
    class MainTaskVFL_LLM( global_model_type, nn.Module): #GenerationMixin object,
        def __init__(self, args, job_id=None):
            self.job_id = job_id
            
            super(global_model_type,self).__init__(args.config)
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
            self.e2e_model = None  # type:E2EModel
            self._init_e2e_model()

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
                if count_time == 'train':
                    self.train_party_time[ik] += end_time - start_time
                elif count_time == 'inference':
                    self.inference_party_time[ik] += end_time - start_time

            self.all_pred_list = all_pred_list
            return all_pred_list

        def parse_pred_message_result(self, test_logit):
            if self.args.model_type == 'Bert':
                if self.args.task_type == 'SequenceClassification':
                    logits = torch.Tensor(test_logit['logits'])
                    if test_logit['requires_grad']:
                        logits.requires_grad_()
                    return logits.to(self.args.device)
                elif self.args.task_type == 'QuestionAnswering':
                    start_logits = torch.Tensor(test_logit['start_logits'])
                    end_logits = torch.Tensor(test_logit['end_logits'])
                    if test_logit['requires_grad']:
                        start_logits.requires_grad_()
                        end_logits.requires_grad_()
                    return QuestionAnsweringModelOutput(
                        loss=None,
                        start_logits=start_logits.to(self.args.device),
                        end_logits=end_logits.to(self.args.device),
                        hidden_states=None,
                        attentions=None,
                    )
            elif self.args.model_type == 'GPT2':
                if self.args.task_type == 'CausalLM':
                    logits = torch.Tensor(test_logit['logits'])
                    if test_logit['requires_grad']:
                        logits.requires_grad_()
                    return logits.to(self.args.device)
                elif self.args.task_type == 'SequenceClassification':
                    logits = torch.Tensor(test_logit['logits'])
                    if test_logit['requires_grad']:
                        logits.requires_grad_()
                    return logits.to(self.args.device)
                elif self.args.task_type == 'QuestionAnswering':
                    start_logits = torch.Tensor(test_logit['start_logits'])
                    end_logits = torch.Tensor(test_logit['end_logits'])
                    if test_logit['requires_grad']:
                        start_logits.requires_grad_()
                        end_logits.requires_grad_()
                    return QuestionAnsweringModelOutput(
                        loss=None,
                        start_logits=start_logits.to(self.args.device),
                        end_logits=end_logits.to(self.args.device),
                        hidden_states=None,
                        attentions=None,
                    )
                else:
                    assert 1 > 2, 'Task type no supported'
            elif self.args.model_type == 'Llama':
                if self.args.task_type == 'SequenceClassification':
                    logits = torch.Tensor(test_logit['logits'])
                    if test_logit['requires_grad']:
                        logits.requires_grad_()
                    return logits.to(self.args.device)
                elif self.args.task_type == 'CausalLM':
                    logits = torch.Tensor(test_logit['logits'])
                    if test_logit['requires_grad']:
                        logits.requires_grad_()
                    return logits.to(self.args.device)
                elif self.args.task_type == 'QuestionAnswering':
                    start_logits = torch.Tensor(test_logit['start_logits'])
                    end_logits = torch.Tensor(test_logit['end_logits'])
                    if test_logit['requires_grad']:
                        start_logits.requires_grad_()
                        end_logits.requires_grad_()
                    return QuestionAnsweringModelOutput(
                        loss=None,
                        start_logits=start_logits.to(self.args.device),
                        end_logits=end_logits.to(self.args.device),
                        hidden_states=None,
                        attentions=None,
                    )
                else:
                    assert 1 > 2, 'Task type no supported'

        def global_pred_transmit(self, pred_list, use_cache=None, count_time=False):
            start_time = time.time()
            final_pred = self._communication.send_pred_message(pred_list,use_cache=use_cache)#
            end_time = time.time()
            if count_time == 'train':
                self.train_party_time[-1] += end_time - start_time
            elif count_time == 'inference':
                self.inference_party_time[-1] += end_time - start_time
            return final_pred

        def local_gradient_transmit(self, count_time='train'):
            for ik in range(self.k - 1):
                if self.parties[ik].local_model_optimizer != None:
                    start_time = time.time()
                    passive_local_gradient = self._communication.send_cal_passive_local_gradient_message(ik)
                    if not isinstance(passive_local_gradient, torch.Tensor):
                        passive_local_gradient = torch.Tensor(passive_local_gradient).to(self.args.device)
                    end_time = time.time()

                    if count_time == 'train':
                        self.train_party_time[self.k - 1] += end_time - start_time

                    self.parties[ik].local_gradient = passive_local_gradient

        def global_gradient_transmit(self, final_pred, count_time='train'):
            start_time = time.time()
            global_loss = self.parties[0].cal_loss(final_pred)
            global_gradients = self.parties[0].cal_global_gradient(global_loss, final_pred)
            end_time = time.time()
            if count_time == 'train':
                self.train_party_time[0] += end_time - start_time

            self.communication_cost += get_size_of(global_gradients)

            self._communication.send_global_loss_and_gradients(self.parties[0].global_gradients)  # self.parties[0].global_loss,
            return global_loss


        def generate_result(self, model_output, gt_one_hot_label):
            # raw_model_output --> standard prediction result
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
                    predict_label = model_output.logits.detach().cpu()
                    actual_label = gt_one_hot_label.detach().cpu()

                    predict_label = torch.tensor([_.item() for _ in predict_label])
                    actual_label = torch.tensor([_.item() for _ in actual_label])

                    sample_cnt = predict_label.shape[0]

                    return list(predict_label), list(actual_label), sample_cnt
                else:  # Classification
                    predict_label = torch.argmax(model_output.logits, dim=-1).detach().cpu()
                    actual_label = torch.argmax(gt_one_hot_label, dim=-1).detach().cpu()

                    sample_cnt = predict_label.shape[0]
                    suc_cnt += torch.sum(predict_label == actual_label).item()
                    return list(predict_label), list(actual_label), sample_cnt

            elif self.args.model_architect=='CLM': #.task_type == "CausalLM":
                if self.args.task_type == "CausalLM":#dataset == "Lambada": 
                    if isinstance(model_output,torch.Tensor): # generation -- generated token ids
                        # model_output: torch.tensor : bs, seq_len+generated_len
                        predict_label_list = model_output[:,self.seq_length:] # [bs, max_new_tokens]
                        target_label_list = list(gt_one_hot_label)
                        # print('generate_result predict_label_list:',predict_label_list.shape)
                    else:  # forward -- raw model output
                        generated_token_logits = model_output.logits[:,-1,:]
                        predict_label_list = torch.argmax(generated_token_logits, dim=-1) 
                        target_label_list = list(gt_one_hot_label)
                        # predict_label_list = [int(_id.item()) for _id in list(predict_label_list)]
                        # target_label_list = [int(_id.item()) for _id in list(gt_one_hot_label)]

                        # predict_label_list = list(model_output.logits) # bs, seq_len, vocab_size
                        # target_label_list = list(gt_one_hot_label) # bs, seq_len   list of [label tensor]
                    
                    return target_label_list, predict_label_list, len(predict_label_list) 

                elif self.args.task_type == "SequenceClassification":
                    target_label_list = list(gt_one_hot_label) # [bs* target_word_id]

                    # forward -- raw model output
                    generated_token_logits = model_output.logits[:,-1,:]
                    probs = []
                    for choice_class in range(self.args.num_classes):
                        choice_id = self.args.tokenizer.convert_tokens_to_ids(self.args.label_dict[choice_class])
                        probs.append( generated_token_logits[:, choice_id] ) # [bs, 1]
                    probs = torch.stack(probs,dim = -1) # [bs, num_choice]
                    predict_label_list = torch.argmax(probs, dim=-1)  # [bs]
                    predict_label_list = [self.args.label_dict[pred_class.item()] for pred_class in predict_label_list]
                    predict_label_list = [self.args.tokenizer.convert_tokens_to_ids(pred_token)\
                             for pred_token in predict_label_list]
                    return target_label_list, predict_label_list, len(predict_label_list) 

                #     print('model_output:',type(model_output),model_output.shape)
                #     print('self.seq_length:',self.seq_length)
                #     new_token_logits = model_output #[:,self.seq_length:]
                #     print('new_token_logits:',type(new_token_logits),new_token_logits.shape)
                #     for _i in range(new_token_logits.shape[0]):
                #         model_output_ids = new_token_logits[_i]
                #         print('text:',self.args.tokenizer.decode(model_output_ids, skip_special_tokens=True))
                    
                #     target_label_list = [int(_id.item()) for _id in list(gt_one_hot_label)]
                #     print('target_label_list:',target_label_list,type(target_label_list[0]))
                #     assert 1>2
                #     predict_label_list = torch.argmax(enc_predict_prob, dim=-1)  # [bs]
                #     predict_label_list = predict_label  # predict_word: bs * best_pred

                #     # if ('positive' in model_outputs[i] or 'pos' in model_outputs[i]) and ('neg' not in model_outputs[i]):
                #     #     model_prediction = 'positive'
                #     # elif ('negative' in model_outputs[i] or 'neg' in model_outputs[i]) and ('pos' not in model_outputs[i]):
                #     #     model_prediction = 'negative'

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
                start_logits = model_output.start_logits # bs, 512
                end_logits = model_output.end_logits # bs, 512
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
                    feature = parties_data[0][0][i]['feature']  # print('parties_data[0][4]:',type(parties_data[0][4]),'feature:',type(feature))
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
                        reverse=True)  # length=2
                    # print('prelim_predictions:',len(prelim_predictions))

                    exact_score = 0
                    f1 = 0
                    nbest = []  # Get n best prediction text
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
                for parties_data in tqdm(zip(*data_loader_list),desc="inference process"):
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

                    if self.args.model_architect=='CLS' and self.args.num_classes > 1:  # classification
                        gt_one_hot_label = self.label_to_one_hot(parties_data[0][1], self.args.num_classes)
                    elif self.args.model_architect=='TQA':
                        gt_one_hot_label = list(parties_data[0][1])
                    else:
                        gt_one_hot_label = parties_data[0][1]


                    data_inputs = {}
                    for key_name in parties_data[0][0][0].keys():
                        if key_name in ['feature']:
                            continue
                        data_inputs[key_name] = torch.stack( [parties_data[0][0][i][key_name] for i in range(len(parties_data[0][0]))] )
                    self.seq_length = data_inputs['input_ids'].shape[-1]

                    # test_logit -> standard output for each task
                    if self.args.model_architect=='CLS': #task_type == "SequenceClassification":  # and self.args.num_classes > 1: # classification
                        global_output = self.forward(**data_inputs)
                        batch_predict_label, batch_actual_label, sample_cnt = self.generate_result(global_output, gt_one_hot_label, parties_data)
                        predict_label_list.extend(batch_predict_label)
                        actual_label_list.extend(batch_actual_label)
                        if sample_cnt is not None:
                            total_sample_cnt += sample_cnt
                    elif self.args.model_architect=='TQA': #task_type == "QuestionAnswering":
                        global_output = self.forward(**data_inputs)
                        batch_nbest, batch_gold_ans, sample_cnt = self.generate_result(global_output, gt_one_hot_label, parties_data)
                        nbest_list.extend(batch_nbest)
                        gold_ans_list.extend(batch_gold_ans)
                        if sample_cnt is not None:
                            total_sample_cnt += sample_cnt
                    elif self.args.model_architect=='CLM': #task_type == "CausalLM":
                        if self.args.max_new_tokens>1:
                            generation_output = self.generate(**data_inputs, \
                                    generation_config = self.generation_config,\
                                    temperature=0.7, top_p=1.0,
                                    max_new_tokens=self.args.max_new_tokens,\
                                    
                                    eos_token_id=2, pad_token_id=2)
                        else: # next token prediction
                            generation_output = self.forward(**data_inputs)
                        self._clear_past_key_values()

                        batch_target_word, batch_predict_word, sample_cnt = self.generate_result(generation_output, gt_one_hot_label, parties_data)
                        target_word_list.extend(batch_target_word)
                        predict_word_list.extend(batch_predict_word)
                        if sample_cnt is not None:
                            total_sample_cnt += sample_cnt
                    else:
                        assert 1 > 2, 'Task type not supported'

                    del parties_data

            if self.args.model_architect=='CLS':  # and self.args.num_classes > 1: # classification
                return predict_label_list, actual_label_list, total_sample_cnt
            elif self.args.model_architect=='TQA':
                return nbest_list, gold_ans_list, total_sample_cnt
            elif self.args.model_architect=='CLM':
                return predict_word_list, target_word_list, total_sample_cnt
            else:
                assert 1 > 2, 'Task type not supported'

        def generate_assessment(self, predict_list, label_list):
            if self.args.model_architect == 'TQA':
                nbest_list = predict_list
                gold_ans_list = label_list

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

                exact_score = np.mean(exact_score_list)
                f1 = np.mean(f1_list)
                return {'exact_score':exact_score, 'f1':f1}

            elif self.args.model_architect == 'CLM':
                predict_word_list = predict_list # bs, seq_len, vocab_size
                target_word_list = label_list # bs, seq_len

                # print('==generate_assessment==')

                
                if len(target_word_list[0].shape)>0: # not next token prediction
                    # def calculate_token_precision_recall(reference_ids, candidate_ids):
                    #     reference_ids = reference_ids.tolist()
                    #     candidate_ids = candidate_ids.tolist()

                    #     while(self.args.tokenizer.pad_token_id in reference_ids):
                    #         reference_ids.remove(self.args.tokenizer.pad_token_id)
                    #     while(self.args.tokenizer.eos_token_id in reference_ids):
                    #         reference_ids.remove(self.args.tokenizer.eos_token_id)
                        
                        
                    #     intersection = set( reference_ids ).intersection( set(candidate_ids ) )
                        
                    #     if len(candidate_ids) == 0 or len(reference_ids)==0:
                    #         precision = 0
                    #         recall = 0
                    #     else:
                    #         precision = len(intersection) / len(candidate_ids)
                    #         recall = len(intersection) / len(reference_ids)

                    #     print('='*50)
                    #     print('reference_ids:',self.args.tokenizer.decode(reference_ids))
                    #     print('-'*25)
                    #     print('candidate_ids:',self.args.tokenizer.decode(candidate_ids))
                    #     print(len(intersection), precision, recall)
                    #     print('='*50)

                    #     return precision, recall
                    
                    def calculate_token_precision_recall(reference_ids, candidate_ids):
                        reference_ids = reference_ids.tolist()
                        candidate_ids = candidate_ids.tolist()

                        def wash(ids, target_token_id):
                            while(target_token_id in ids):
                                ids.remove(target_token_id)
                            return ids
                        reference_ids = wash(reference_ids, self.args.tokenizer.pad_token_id)
                        reference_ids = wash(reference_ids, self.args.tokenizer.eos_token_id)

                        reference_tokens = [self.args.tokenizer.convert_ids_to_tokens(reference_ids)]
                        candidate_tokens = self.args.tokenizer.convert_ids_to_tokens(candidate_ids)

                        score = sentence_bleu(reference_tokens, candidate_tokens)

                        print('='*50)
                        print('Reference_tokens:',reference_tokens)
                        print('-'*25)
                        print('Candidate_tokens',candidate_tokens)
                        print('Score:',score)
                        print('='*50)

                        return score
                    
                    score = 0
                    for i in range(len(target_word_list)):
                        _score = calculate_token_precision_recall(target_word_list[i], predict_word_list[i])
                        score += _score
                    score = score/len(target_word_list)
                    acc = score
                else:
                    if self.args.metric_type == "best_pred":
                        suc_cnt = 0
                        for i in range(len(target_word_list)):
                            if target_word_list[i] == predict_word_list[i]:
                                suc_cnt += 1
                        acc = suc_cnt / float(len(target_word_list))
                    elif self.args.metric_type == "n_best":
                        suc_cnt = 0
                        for i in range(len(target_word_list)):
                            if target_word_list[i] in predict_word_list[i]:
                                suc_cnt += 1
                        acc = suc_cnt / float(len(target_word_list))  # ACC
                    else:
                        assert 1 > 2, 'metric type not supported'
                    
                return {'acc':acc}

            elif self.args.model_architect == 'CLS':
                predict_labels = predict_list
                actual_labels = label_list
                if self.num_classes == 1:
                    mse = torch.mean(
                        (torch.tensor(predict_labels) - torch.tensor(actual_labels)) ** 2).item()
                    pearson_corr = stats.pearsonr(torch.tensor(predict_labels), torch.tensor(actual_labels))[0]
                    spearmanr_corr = stats.spearmanr(torch.tensor(predict_labels), torch.tensor(actual_labels))[0]
                    return {'mse':mse, 'pearson_corr':pearson_corr, 'spearmanr_corr':spearmanr_corr}
                else:
                    suc_cnt = torch.sum(torch.tensor(predict_labels) == \
                                        torch.tensor(actual_labels)).item()
                    acc = suc_cnt / torch.tensor(predict_labels).shape[0]  # ACC
                    mcc = matthews_corrcoef(np.array(predict_labels), np.array(actual_labels))  # MCC

                    return {'acc':acc, 'mcc':mcc}

        def _llm_inference(self, **kwargs):
            if self.k > 2:
                raise ValueError('llm_inference only supports k=2')

            format_kwargs = self._format_forward_kwargs(**kwargs)
            generate_ids = self.generate(format_kwargs.get('input_ids'), max_new_tokens=20)
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

        def seq_inference(self):
            # SequenceClassification / Regression
            predict_labels, actual_labels, total_sample_cnt = self.predict()

            # prediction result assessment
            result_dict = self.generate_assessment(predict_labels, actual_labels)
            if self.num_classes == 1:
                self.test_mse = result_dict['mse']
                self.test_pearson_corr = result_dict['pearson_corr']
                self.test_spearmanr_corr = result_dict['spearmanr_corr']

                exp_result = f'|test_mse={self.test_mse}|test_pearson_corr={self.test_pearson_corr}|test_spearmanr_corr={self.test_spearmanr_corr}'
                print(exp_result)
                return exp_result, [self.test_mse, self.test_pearson_corr, self.test_spearmanr_corr]
            else:
                self.test_acc = result_dict['acc']
                self.test_mcc = result_dict['mcc']
                exp_result = f'|test_acc={self.test_acc}|test_mcc={self.test_mcc}'
                print(exp_result)
                return exp_result, self.test_acc

        def causal_lm_inference(self):
            self.eval()
            predict_word_list, target_word_list, total_sample_cnt = self.predict()
            
            # print('causal_lm_inference target_word_list[0]:',len(target_word_list),target_word_list[0].shape )
            # print('causal_lm_inference predict_word_list[0]:',len(predict_word_list),predict_word_list[0].shape )

            result_dict = self.generate_assessment(predict_word_list, target_word_list)
            self.test_acc = result_dict['acc']
            exp_result = f'|test_acc={self.test_acc}'
            # print(exp_result)

            return exp_result, self.test_acc

        def qa_inference(self):
            # generate all model prediction
            start_time = time.time()
            nbest_list, gold_ans_list, total_sample_cnt = self.predict()
            end_time = time.time()

            start_time = time.time()

            result_dict = self.generate_assessment(nbest_list, gold_ans_list)
            exp_result = '|exact_score={:.4f}|f1={:.4f}'.format(result_dict['exact_score'], result_dict['f1'])

            self.test_acc = result_dict['exact_score']
            print(exp_result)
            return exp_result, self.test_acc

        def forward(self, count_time=False, **kwargs):
            self.parties[0].obtain_local_data(kwargs)
            # passive party do local pred
            pred_list = self.pred_transmit(use_cache=False,count_time=count_time)

            # passive party inform active party to do global pred
            resp = self.global_pred_transmit(pred_list, use_cache=False,count_time=count_time)
            if not isinstance(resp,(dict,torch.Tensor)):
                resp=resp.prepare_for_forward()
                t=resp['inputs_embeds']
                req_grad=t.requires_grad
                resp['inputs_embeds']=t.detach().clone()
                if req_grad:
                    resp['inputs_embeds'].requires_grad=True
            if vfl_basic_config.num_of_slice > 2:
                # todo: deal with 3-slice inference
                final_output=self.parties[0].forward(2,**resp)
            else:
                # todo: use global_pred_transmit return instead
                final_output = self.parties[1].global_output
            return final_output

        def backward(self,final_pred):
            # passive party -> global gradient -> active party
            loss=self.global_gradient_transmit(final_pred, count_time = 'train')
            # active party -> local gradient -> passive party
            self.local_gradient_transmit(count_time = 'train')
            if vfl_basic_config.num_of_slice==3:
                # todo: deal with 3-slice backward
                loss.backward()
                # self.parties[0].backward(2)
                self.parties[0].optimizer_step(2)

        def inference(self, **kwargs):

            if self.args.task_type == "DevLLMInference":
                # LLM推理入口，兼容forward格式入参
                result = self._llm_inference(**kwargs)
                return result
            # set inference time back to 0
            self.inference_party_time = [0 for i in range(self.k)]

            # print(' ========= Inference ==========')
            for ik in range(self.k - 1):
                self.parties[ik].prepare_data_loader()
                self.parties[ik].eval()
            self.parties[self.k - 1].eval()

            if self.args.model_architect == 'TQA': #task_type == "QuestionAnswering":
                exp_result, main_task_result = self.qa_inference()
                self.final_state = self.save_state()
                # self.final_state.update(self.save_state(False))
                self.final_state.update(self.save_party_data())
                exp_result = f'|inference_party_time={self.inference_party_time}' + exp_result
                return exp_result, main_task_result

            if self.args.model_architect=='CLS':#task_type == "SequenceClassification":
                # exp_result, self.test_acc =
                exp_result, main_task_result = self.seq_inference()
                self.final_state = self.save_state()
                # self.final_state.update(self.save_state(False))
                self.final_state.update(self.save_party_data())
                exp_result = f'|inference_party_time={self.inference_party_time}' + exp_result
                return exp_result, main_task_result

            if self.args.model_architect=='CLM':#task_type == "CausalLM":
                exp_result, main_task_result = self.causal_lm_inference()
                self.final_state = self.save_state()
                # self.final_state.update(self.save_state(False))
                self.final_state.update(self.save_party_data())
                exp_result = f'|inference_party_time={self.inference_party_time}' + exp_result
                return exp_result, main_task_result


        def train_batch(self, parties_data, batch_label):
            ############### allocate data ###############
            gt_one_hot_label = batch_label
            self.gt_one_hot_label = gt_one_hot_label
            for ik in range(self.k - 1):
                # # allocate data (data/label/attention_mask/token_type_ids)
                data_inputs = {}
                for key_name in parties_data[ik][0][0].keys():
                    if isinstance(parties_data[ik][0][0][key_name], torch.Tensor):
                        data_inputs[key_name] = torch.stack( [parties_data[ik][0][i][key_name] for i in range(len(parties_data[ik][0]))] )
                    else:
                        data_inputs[key_name] =  [parties_data[ik][0][i][key_name] for i in range(len(parties_data[ik][0]))]
                self.parties[ik].obtain_local_data(data_inputs)
                self.parties[ik].gt_one_hot_label = gt_one_hot_label
                self.seq_length = data_inputs['input_ids'].shape[-1]

            ################ normal vertical federated learning ################
            # torch.autograd.set_detect_anomaly(True)
            # =================== Commu ===================

            final_pred = self.forward(**data_inputs)
            self._clear_past_key_values()
            # generation_output = self.generate(**data_inputs, \
            #     generation_config = self.generation_config,max_new_tokens=1)
            # self._clear_past_key_values()
            # print('generation_output:',type(generation_output),generation_output.shape)

            self.backward(final_pred)
            # loss = self.global_gradient_transmit(final_pred, count_time = 'train')
            # # active party -> local gradient -> passive party
            # self.local_gradient_transmit(count_time = 'train')

            # ============= Model Update =============
            start_time = time.time()
            self._communication.send_global_backward_message()
            end_time = time.time()
            self.train_party_time[self.args.k - 1] += end_time - start_time

            for ik in range(self.k - 1):
                start_time = time.time()
                self.parties[ik].local_backward()
                end_time = time.time()
                self.train_party_time[ik] += end_time - start_time

            ################ normal vertical federated learning ################

            # print train_acc each batch
            if self.args.model_architect=='TQA': #self.args.task_type == 'QuestionAnswering':
                pred = self.parties[self.k - 1].global_output  # QuestionAnsweringModelOutput
                loss = self.parties[0].global_loss

                batch_nbest, batch_gold_ans, sample_cnt = self.generate_result(pred, gt_one_hot_label)

                result_dict = self.generate_assessment(batch_nbest, batch_gold_ans)

                exact_score = result_dict['exact_score']
                return loss.item(), exact_score

            elif self.args.model_architect=='CLS': #self.args.task_type == 'SequenceClassification':

                if self.args.num_classes == 1:
                    pred = self.parties[self.k - 1].global_output
                    loss = self.parties[0].global_loss

                    batch_predict_label, batch_actual_label, sample_cnt = self.generate_result(pred, gt_one_hot_label)          
                    result_dict = self.generate_assessment(batch_predict_label, batch_actual_label)

                    batch_mse = result_dict['mse']
                    batch_pearson_corr = result_dict['pearson_cor']
                    batch_spearmanr_corr = result_dict['spearmanr_corr']

                    return loss.item(), [batch_mse, batch_pearson_corr, batch_spearmanr_corr]
                else:
                    pred = self.parties[self.k - 1].global_output
                    loss = self.parties[0].global_loss
                    
                    batch_predict_label, batch_actual_label, sample_cnt = self.generate_result(pred, gt_one_hot_label)
                    result_dict = self.generate_assessment(batch_predict_label, batch_actual_label)

                    batch_train_acc = result_dict['acc']
                    return loss.item(), batch_train_acc

            elif self.args.model_architect=='CLM':  #self.args.task_type == 'CausalLM':
                pred = self.parties[self.k - 1].global_output  # logits
                loss = self.parties[0].global_loss
                
                batch_train_acc = 0
                
                # batch_target_word, batch_predict_word, sample_cnt = self.generate_result(pred, gt_one_hot_label, parties_data)
                # print('train_batch batch_target_word:',type(batch_target_word),type(batch_target_word[0]))
                # print('len:',batch_target_word[0].shape) # torch.size[512]  # torch.size[]
                
                # print('train_batch batch_predict_word:',type(batch_predict_word),type(batch_predict_word[0]))
                # print('len:',batch_predict_word[0].shape)# torch.size[]  # torch.size[]
                
                # result_dict = self.generate_assessment(batch_predict_word, batch_target_word)
                # batch_train_acc = result_dict['acc']
                
                # if self.args.dataset == "Lambada":
                #     # print('gt_one_hot_label:',type(gt_one_hot_label),gt_one_hot_label)
                #     target_label_list = [int(_p) for _p in gt_one_hot_label]

                #     # predict_word_list : bs * predicted words
                #     enc_predict_prob = nn.functional.softmax(next_token_logits, dim=-1)
                #     if self.args.metric_type == "best_pred":
                #         predict_label_list = torch.argmax(enc_predict_prob, dim=-1)  # [bs]
                #     elif self.args.metric_type == "n_best":
                #         logit_list, index_list = torch.sort(enc_predict_prob, descending=True)
                #         # print('index_list:',index_list.shape)
                #         predict_label_list = index_list[:, :self.args.n_best_size]

                #     if self.args.metric_type == "best_pred":
                #         suc_cnt = 0
                #         for i in range(len(target_label_list)):
                #             if target_label_list[i] == predict_label_list[i]:
                #                 suc_cnt += 1
                #         batch_train_acc = suc_cnt / float(len(target_label_list))  # ACC
                #     elif self.args.metric_type == "n_best":
                #         suc_cnt = 0
                #         for i in range(len(target_label_list)):
                #             if target_label_list[i] in predict_label_list[i]:
                #                 suc_cnt += 1
                #         batch_train_acc = suc_cnt / float(len(target_label_list))  # ACC
                #     else:
                #         assert 1 > 2, 'metric type not supported'


                # else:  # MMLU
                #     choice_id_list = []
                #     for choice in self.args.label_dict.keys():
                #         choice_id_list.append(self.args.tokenizer(choice).input_ids[-1])
                #         _id = self.args.tokenizer(choice).input_ids[-1]
                #     enc = next_token_logits[:, choice_id_list]  # [bs, num_choice]
                #     enc_predict_prob = nn.functional.softmax(enc, dim=-1)  # [bs, num_choice]

                #     predict_label = torch.argmax(enc_predict_prob, dim=-1)  # [bs]
                #     actual_label = gt_one_hot_label  # torch.argmax(gt_one_hot_label, dim=-1)

                #     # test_predict_labels = predict_label.detach().cpu().tolist()
                #     # test_actual_labels = actual_label.detach().cpu().tolist()
                #     # test_full_predict_labels.extend( list(full_predict_label.detach().cpu()) )

                #     sample_cnt += predict_label.shape[0]
                #     suc_cnt += torch.sum(predict_label == actual_label).item()

                return loss.item(), batch_train_acc

            
        def train_vfl(self,**kwargs): # def train(self):
            training_args = vfl_basic_config.vfl_training_config.training_args
            # 创建 TensorBoard 摘要写入器
            tensorboard_writer = SummaryWriter(training_args.logging_dir)

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
            optimize_step = 0

            data_record = pd.DataFrame(columns=['Epoch', 'train_loss', 'train_acc', 'test_acc'])
            if "validation_before_epoch":
                for p in self.parties:
                    p.eval()
                with torch.no_grad():
                    _exp_result, test_acc = self.inference()
                tensorboard_writer.add_scalar('train/test_acc', test_acc, 0)
            for i_epoch in range(self.epochs):
                self.train()
                self.current_epoch = i_epoch
                tensorboard_writer.add_scalar('train/epoch', i_epoch, optimize_step)
                postfix = {'train_loss': 0.0, 'train_acc': 0.0, 'test_acc': 0.0}
                i = -1
                print_every = 1
                total_time = 0
                data_loader_list = [self.parties[ik].train_loader for ik in range(self.k - 1)]
                for parties_data in tqdm(zip(*data_loader_list), desc=f'Epoch {i_epoch}/{self.epochs - 1}'):
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

                    if self.args.model_architect=='CLS' and self.num_classes > 1:  # self.args.task_type == "SequenceClassification"
                        gt_one_hot_label = self.label_to_one_hot(parties_data[0][1], self.num_classes)
                    else:
                        gt_one_hot_label = torch.tensor(parties_data[0][1]).to(self.current_device)
                    self.gt_one_hot_label = gt_one_hot_label
                    i += 1

                    # passive party call active party global model to a training mode
                    self._communication.send_global_model_train_message()

                    # ====== train batch (start) ======
                    enter_time = time.time()
                    self.loss, self.train_acc = self.train_batch(parties_data, gt_one_hot_label)
                    exit_time = time.time()
                    total_time += (exit_time - enter_time)
                    optimize_step += 1
                    tensorboard_writer.add_scalar('train/loss', self.loss, optimize_step)
                    # todo： 添加逻辑，通过判断训练的层来获取lr
                    try:
                        tensorboard_writer.add_scalar('train/lr_local',
                                                      self.parties[0].local_model_optimizer.param_groups[0]['lr'],
                                                      optimize_step)
                    except Exception as e:
                        logger.debug(repr(e))
                        pass
                    try:
                        tensorboard_writer.add_scalar('train/lr_global',
                                                      self.parties[1].global_model_optimizer.param_groups[0]['lr'],
                                                      optimize_step)
                    except Exception as e:
                        logger.debug(repr(e))
                        pass
                    try:
                        tensorboard_writer.add_scalar('train/lr_model_2',
                                                      self.parties[0].optimizers[2].param_groups[0]['lr'],
                                                      optimize_step)
                    except Exception as e:
                        logger.debug(repr(e))
                        pass

                    gc.collect()
                    # ====== train batch (end) ======
                    self.num_total_comms = self.num_total_comms + 1
                    # if self.num_total_comms % 10 == 0:
                    #     print(f"total time for {self.num_total_comms} communication is {total_time}")
                    self.current_step = self.current_step + 1

                    del (parties_data)

                # LR decay
                self.LR_Decay(i_epoch)
                if vfl_basic_config.num_of_slice==3:
                    self.parties[0].lr_schedulers[2].step()
                # _lr = self.parties[0].global_LR_decay(i_epoch, is_return=True)

                if self.args.apply_adversarial:
                    print(
                        f'global_loss={self.parties[0].global_loss} adversarial_model_loss:{self.parties[0].adversarial_model_loss.item()} adversary_attack_loss:{self.parties[0].adversary_attack_loss.item()}')
                if self.args.apply_mid:
                    print(f'global_loss={self.parties[0].global_loss},mid_loss={self.parties[0].mid_loss}')

                # validation
                if (i + 1) % print_every == 0:
                    print("validate and test")
                    self.parties[self.k - 1].eval()
                    self.eval()

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
                    tensorboard_writer.add_scalar('train/test_acc', self.test_acc, optimize_step)

                data_record.loc[len(data_record)] = [i_epoch, self.loss, self.train_acc, self.test_acc]

                # Early Stop
                if self.loss >= last_loss:
                    early_stop_count = early_stop_count + 1
                if early_stop_count >= early_stop_threshold:
                    self.final_epoch = i_epoch + 1
                    break
                last_loss = min(last_loss, self.loss)

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
            result_file_name=result_file_name.replace('/','')
            print('Save csv to:', result_file_name)
            data_record.to_csv(result_file_name)

            if self.args.apply_defense:
                if self.args.apply_mid:
                    self.save_defense_models()
            return exp_result, self.test_acc, total_time  # , self.stopping_iter, self.stopping_time, self.stopping_commu_cost

        def save_state(self, BEFORE_MODEL_UPDATE=True):
            if BEFORE_MODEL_UPDATE:
                return {
                    "model": [copy.deepcopy(self.parties[ik].local_model) for ik in range(self.args.k)],
                    "global_model": copy.deepcopy(self.parties[self.args.k - 1].global_model),
                    "model_names": [str(type(self.parties[ik].local_model)).split('.')[-1].split('\'')[-2] for ik in
                                    range(self.args.k)] + [
                                       str(type(self.parties[self.args.k - 1].global_model)).split('.')[-1].split('\'')[
                                           -2]]

                }
            else:
                return {
                    # "model": [copy.deepcopy(self.parties[ik].local_model) for ik in range(self.args.k)]+[self.parties[self.args.k-1].global_model],
                    "data": copy.deepcopy(self.parties_data),
                    "label": copy.deepcopy(self.gt_one_hot_label),
                    # "predict": [copy.deepcopy(self.parties[ik].local_pred_clone) for ik in range(self.k)],
                    # "gradient": [copy.deepcopy(self.parties[ik].local_gradient) for ik in range(self.k)],
                    # "local_model_gradient": [copy.deepcopy(self.parties[ik].weights_grad_a) for ik in range(self.k)],
                    "train_acc": copy.deepcopy(self.train_acc),
                    "loss": copy.deepcopy(self.loss),
                    # "global_pred": self.parties[self.k - 1].global_output,
                    "final_model": [copy.deepcopy(self.parties[ik].local_model) for ik in range(self.args.k)],
                    # "final_global_model": copy.deepcopy(self.parties[self.args.k - 1].global_model),
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

        def save_defense_models(self):
            dir_path = self.exp_res_dir + f'/defense_models/'
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            file_path = dir_path + f'{self.args.defense_name}_{self.args.defense_configs}.pkl'

            with open(file_path, 'wb') as f:
                if self.args.apply_mid:
                    pickle.dump(self.parties[0].mid_model, f)
            
            # with open('my_model.pkl', 'rb') as f:
            #     model = pickle.load(f)
            # torch.save(self.parties[0].mid_model.state_dict(),self.trained_models["model_names"],
            #         file_path)

        def save_trained_models(self):
            dir_path = self.exp_res_dir + f'trained_models/parties{self.k}_topmodel{self.args.apply_trainable_layer}_epoch{self.epochs}/'
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            if self.args.apply_defense:
                file_path = dir_path + f'{self.args.defense_name}_{self.args.defense_configs}.pkl'
            else:
                file_path = dir_path + 'NoDefense.pkl'
            torch.save(
                ([self.trained_models["model"][i].state_dict() for i in range(len(self.trained_models["model"]))],
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
            # todo: fix 3-slice Compatibility
            if vfl_basic_config.num_of_slice==3:
                return super(global_model_type).prepare_inputs_for_generation(input_ids,**model_kwargs)
            else:
                # return self.parties[-1].global_model.prepare_inputs_for_generation(input_ids=input_ids, **model_kwargs)
                return self.parties[0].local_model.prepare_inputs_for_generation(input_ids=input_ids, **model_kwargs)

        def _prepare_encoder_decoder_kwargs_for_generation(self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None):
            return self.parties[-1].global_model._prepare_encoder_decoder_kwargs_for_generation(\
                inputs_tensor=inputs_tensor, model_kwargs = model_kwargs, model_input_name= model_input_name)

        def __call__(self, **kwargs):
            return self.forward(**kwargs)

        def _clear_past_key_values(self):
            for i in range(self.k-1):
                self.parties[i].local_model._clear_past_key_values()
            self.parties[-1].global_model._clear_past_key_values()

        def _init_e2e_model(self):
            return
            model_config = None
            if not self.args.task_type == 'DevLLMInference':
                return
            for party in self.parties:
                try:
                    for model in party.models.values():
                        model_config = model.config
                        break
                    break
                except Exception as e:
                    continue
                # if party.models:
                #     model_config = party.local_model.config
                #     break
                # elif party.models:
                #     model_config = party.global_model.config
                #     break
            if not model_config:
                logger.error(f"No model config for E2E_model")
            with init_empty_weights():
                self.e2e_model = E2EModel(model_config,
                                            {**self.parties[0].proxy_models, **self.parties[1].proxy_models})
            # self.e2e_model.to(self.e2e_model.local_model.device)

        @property
        def passive_party(self):
            return self.parties[0]

        def _train_e2e_model(self):
            """
            当前按照 MainTask, MainTask.e2emodel, e2emodel.models[0] 在同一个端可直接访问的结构来设计
            :return:
            """
            training_args = vfl_basic_config.vfl_training_config.training_args
            logger.info(f"Training saving to {training_args.output_dir}")
            # 创建 TensorBoard 摘要写入器
            tensorboard_writer = SummaryWriter(training_args.logging_dir)

            # # 创建 DataLoader 加载训练和验证数据集
            # # train_dataloader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=False)
            # # eval_dataloader = DataLoader(eval_dataset, batch_size=training_args.per_device_eval_batch_size)
            # if isinstance(model, E2EModel):
            #     trainable_params = filter(lambda x: x.requires_grad, model.global_model.parameters())
            # elif isinstance(model, E2EModelV2):
            #     trainable_params = filter(lambda x: x.requires_grad, model.parameters())
            # else:
            #     # trainable_params = get_trainable_parameters(model)
            #     trainable_params = filter(lambda x: x.requires_grad, model.parameters())

            # 开始训练
            forward_step = 0
            optimize_step = 0
            # best_eval_loss = float('inf')
            # todo: fix self.passive_party.dataset_train
            dataset_train = DataLoader(self.passive_party.dataset_train,
                                       batch_size=training_args.per_device_train_batch_size,
                                       collate_fn=dataset_batch_processor)
            for epoch in range(training_args.num_train_epochs):
                total_loss = 0
                for batch in tqdm(dataset_train, desc=f"Epoch {epoch + 1}/{training_args.num_train_epochs}",
                                  leave=False):
                    # 将数据传递给设备
                    # inputs = batch['input_ids'].to(training_args.device)
                    # labels = batch['labels'].to(training_args.device)

                    # 前向传播
                    self.e2e_model.train()
                    outputs = self.e2e_model(**self.__class__.data_collator(batch, self.e2e_model.device,
                                                                            batch_size=training_args.per_device_train_batch_size))
                    forward_step += 1

                    loss = outputs.loss

                    # 反向传播
                    # loss.backward()
                    self.e2e_model.backward()
                    if not forward_step % training_args.gradient_accumulation_steps == 0:
                        continue
                    # optimizer.step()
                    # optimizer.zero_grad()
                    self.e2e_model.optimizer_step()
                    self.e2e_model.optimizer_zero_grad()

                    # 累积总损失
                    total_loss += loss.item()

                    optimize_step += 1

                    # 记录训练损失
                    tensorboard_writer.add_scalar('train/loss', loss, optimize_step)
                    tensorboard_writer.add_scalar('train/learning_rate', self.e2e_model.optimizer.param_groups[0]['lr'],
                                                  optimize_step)
                    if optimize_step % training_args.eval_steps == 0:
                        # 在验证集上评估模型
                        eval_loss = self.__class__.validate_model(self.e2e_model, dataset_validate)
                        tensorboard_writer.add_scalar('train/eval_loss', eval_loss, optimize_step)

                    if optimize_step % training_args.save_steps == 0:
                        self.e2e_model.save_pretrained(
                            os.path.join(training_args.output_dir, f'checkpoint-{optimize_step}'))

                self.e2e_model.lr_scheduler_step()

                tensorboard_writer.add_scalar('train/epoch', epoch + 1, optimize_step)

                # 如果当前模型性能优于之前的最佳性能，则保存当前模型
                # if eval_loss < best_eval_loss:
                #     best_eval_loss = eval_loss
                # torch.save(model.state_dict(), f"{training_args.output_dir}/best_model.pt")

                # 更新学习率

            tensorboard_writer.close()

        def _validate_e2e_model(self):
            self.e2e_model.eval()
            count, loss = 0, 0
            _dataset = DataLoader(self.passive_party.dataset_train,
                                  batch_size=vfl_basic_config.vfl_training_config.training_args.per_device_train_batch_size,
                                  collate_fn=dataset_batch_processor)
            with torch.no_grad():
                for batch in tqdm(_dataset, leave=False):
                    output = self.e2e_model(**data_collator(batch, model.device))
                    if output.loss.item() != output.loss.item():
                        continue
                        # return example
                    loss += output.loss.item()
                    count += 1
                return loss / count

    return MainTaskVFL_LLM
