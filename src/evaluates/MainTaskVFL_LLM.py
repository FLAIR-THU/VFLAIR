import sys, os
sys.path.append(os.pardir)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
import numpy as np
import random
import time
import copy
import collections

from sklearn.metrics import roc_auc_score,matthews_corrcoef
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
from utils.communication_protocol_funcs import compress_pred,Cache,ins_weight
from utils.squad_utils import normalize_answer

from evaluates.attacks.attack_api import AttackerLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoModelForCausalLM

from load.LoadModels import MODEL_PATH

tf.compat.v1.enable_eager_execution() 
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True


STOPPING_ACC = {'mnist': 0.977, 'cifar10': 0.80, 'cifar100': 0.40,'diabetes':0.69,\
'nuswide': 0.88, 'breast_cancer_diagnose':0.88,'adult_income':0.84,'cora':0.72,\
'avazu':0.83,'criteo':0.74,'nursery':0.99,'credit':0.82, 'news20':0.8,\
 'cola_public':0.8,'SST-2':0.9}  # add more about stopping accuracy for different datasets when calculating the #communication-rounds needed


class MainTaskVFL_LLM(object):

    def __init__(self, args):
        self.args = args
        self.k = args.k
        # self.k_server = args.k_server
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
        # self.servers = args.servers

        
        self.Q = args.Q # FedBCD

        self.parties_data = None
        self.gt_one_hot_label = None
        self.clean_one_hot_label  = None
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
        self.num_batch_per_workset = args.Q #args.num_batch_per_workset
        self.max_staleness = self.num_update_per_batch*self.num_batch_per_workset 

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

    def LR_Decay(self,i_epoch):
        # for ik in range(self.k):
        #     self.parties[ik].LR_decay(i_epoch)
        self.parties[self.k-1].global_LR_decay(i_epoch)
    
    # def gradient_transmit(self):  # Active party sends gradient to passive parties
    #     gradient = self.parties[self.k-1].give_gradient() # gradient_clone

    #     if len(gradient)>1:
    #         for _i in range(len(gradient)-1):
    #             self.communication_cost += get_size_of(gradient[_i+1])#MB

    #     # defense applied on gradients
    #     if self.args.apply_defense == True and self.args.apply_dcor == False and self.args.apply_mid == False and self.args.apply_cae == False:
    #         if (self.k-1) in self.args.defense_configs['party']:
    #             gradient = self.launch_defense(gradient, "gradients")   
    #     if self.args.apply_dcae == True:
    #         if (self.k-1) in self.args.defense_configs['party']:
    #             gradient = self.launch_defense(gradient, "gradients")  
            
    #     # active party update local gradient
    #     self.parties[self.k-1].update_local_gradient(gradient[self.k-1])
        
    #     # active party transfer gradient to passive parties
    #     for ik in range(self.k-1):
    #         self.parties[ik].receive_gradient(gradient[ik])
    #     return

    def pred_transmit(self): 
        # Active party gets pred from passive parties
        for ik in range(self.k-1):
            # Party sends attention_mask/input_shape for aggregation
            input_shape = self.parties[ik].local_batch_data.size()
            self.parties[ik].input_shape = input_shape

            self.parties[self.k-1].input_shape = input_shape
            self.parties[self.k-1].receive_attention_mask(self.parties[ik].local_batch_attention_mask)
            self.parties[self.k-1].receive_token_type_ids(self.parties[ik].local_batch_token_type_ids)

            # give pred
            if self.args.model_type == 'Bert':
                pred, pred_detach, attention_mask  = self.parties[ik].give_pred() # , _input_shape

                pred_clone = torch.autograd.Variable(pred_detach, requires_grad=True).to(self.args.device)

                self.parties[self.k-1].receive_pred([pred_clone,attention_mask], ik) 

                self.parties[ik].update_local_pred(pred_clone)

                # print(' === transmit === ')
                # print('pred_clone:',pred_clone.shape)
                # print('local_batch_attention_mask:',local_batch_attention_mask.shape)
                # print('local_batch_token_type_ids:',local_batch_token_type_ids.shape)
                # print('torch.tensor(input_shape) :',torch.tensor(input_shape) .shape)

                self.communication_cost += get_size_of(pred_clone)+\
                    get_size_of(self.parties[ik].local_batch_attention_mask)+\
                    get_size_of(self.parties[ik].local_batch_token_type_ids)+\
                    get_size_of( torch.tensor(input_shape) ) #MB

            elif self.args.model_type == 'Llama':
                if self.args.task_type == 'SequenceClassification':
                    pred, pred_detach , sequence_lengths, attention_mask = self.parties[ik].give_pred() # , _input_shape
                elif self.args.task_type == 'CausalLM':
                    pred, pred_detach , attention_mask = self.parties[ik].give_pred() # , _input_shape
                elif self.args.task_type == 'QuestionAnswering':
                    pred, pred_detach , attention_mask = self.parties[ik].give_pred() # , _input_shape

                # defense applied on pred
                if self.args.apply_defense == True and self.args.apply_dp == True :
                    # Only add noise to pred when launching FR attack(attaker_id=self.k-1)
                    if (ik in self.args.defense_configs['party']) and (ik != self.k-1): # attaker won't defend its own attack
                        pred_detach = torch.tensor(self.launch_defense(pred_detach, "pred")) 
                ########### communication_protocols ###########
                if self.args.communication_protocol in ['Quantization','Topk']:
                    pred_detach = compress_pred( self.args ,pred_detach , self.parties[ik].local_gradient,\
                                    self.current_epoch, self.current_step).to(self.args.device)
                ########### communication_protocols ###########
                
                pred_clone = torch.autograd.Variable(pred_detach, requires_grad=True).to(self.args.device)
                attention_mask = torch.autograd.Variable(decoder_attention_mask, requires_grad=True).to(self.args.device)

                if self.args.task_type == 'SequenceClassification':
                    self.parties[self.k-1].receive_pred([pred_clone,sequence_lengths,attention_mask], ik) 
                elif self.args.task_type == 'CausalLM':
                    self.parties[self.k-1].receive_pred([pred_clone,attention_mask], ik) 
                elif self.args.task_type == 'QuestionAnswering':
                    self.parties[self.k-1].receive_pred([pred_clone,attention_mask], ik) 


                self.parties[ik].update_local_pred(pred_clone)

                # TODO
                self.communication_cost += get_size_of(pred_clone)+\
                    get_size_of(self.parties[ik].local_batch_attention_mask)+\
                    get_size_of( torch.tensor(input_shape) ) #MB

            elif self.args.model_type == 'GPT2':
                if self.args.task_type == 'SequenceClassification':
                    pred, pred_detach , sequence_lengths, attention_mask = self.parties[ik].give_pred() # , _input_shape
                elif self.args.task_type == 'CausalLM':
                    pred, pred_detach , attention_mask = self.parties[ik].give_pred() 
                elif self.args.task_type == 'QuestionAnswering':
                    pred, pred_detach , attention_mask = self.parties[ik].give_pred() 

                pred_clone = torch.autograd.Variable(pred_detach, requires_grad=True).to(self.args.device)
                attention_mask = torch.autograd.Variable(attention_mask, requires_grad=True).to(self.args.device)

                if self.args.task_type == 'SequenceClassification':
                    self.parties[self.k-1].receive_pred([pred_clone,sequence_lengths,attention_mask], ik) 
                elif self.args.task_type == 'CausalLM':
                    self.parties[self.k-1].receive_pred([pred_clone,attention_mask], ik)
                elif self.args.task_type == 'QuestionAnswering':
                    self.parties[self.k-1].receive_pred([pred_clone,attention_mask], ik) 
            
                self.parties[ik].update_local_pred(pred_clone)

                # print(' === transmit === ')
                # print('pred_clone:',pred_clone.shape)
                # print('local_batch_attention_mask:',local_batch_attention_mask.shape)
                # print('local_batch_token_type_ids:',local_batch_token_type_ids.shape)
                # print('torch.tensor(input_shape) :',torch.tensor(input_shape) .shape)

                # TODO
                self.communication_cost += get_size_of(pred_clone)+\
                    get_size_of(self.parties[ik].local_batch_attention_mask)+\
                    get_size_of( torch.tensor(input_shape) ) #MB
            
    def global_pred_transmit(self):
        # active party give global pred to passive party
        final_pred = self.parties[self.k-1].aggregate(self.parties[self.k-1].pred_received, test="True")
        
        for ik in range(self.k-1):
            self.communication_cost += get_size_of(final_pred)
            self.parties[ik].global_pred = final_pred
    
    def global_loss_transmit(self):
        # passive party give loss to active party -- used to update global model
        global_loss = self.parties[0].cal_loss() 
        self.communication_cost += get_size_of(global_loss)
        self.parties[self.k-1].global_loss = global_loss
    
    def inference(self, inference_data = 'test'):
        # current_model_type = self.args.model_list['0']['type']
        # full_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH[current_model_type]).to(self.args.device)

        print(' ========= Inference ==========')
        postfix = {'test_acc': 0.0}
        for ik in range(self.k):
            self.parties[ik].prepare_data_loader(batch_size=self.batch_size)
        self.parties[self.k-1].global_model.eval()
        
                
        suc_cnt = 0
        sample_cnt = 0
        full_suc_cnt = 0

        # QA
        exact_score_list = []
        f1_list = []

        test_preds = []
        test_targets = []
        full_test_preds = []

        test_predict_labels = []
        test_actual_labels = []
        test_full_predict_labels = []

        scores = []
        targets = []
        with torch.no_grad():
            data_loader_list = [self.parties[ik].test_loader for ik in range(self.k-1)]
            for parties_data in zip(*data_loader_list):
                # parties_data[0]: (data, label, mask, token_type_ids, doc_tokens(forQA) )
                # parties_data[k-1]: (None,None,None)  no data for active party
                parties_data = [ [_data[0],_data[1],_data[2],_data[3],_data[4]] for _data in parties_data]

                if parties_data[0][3] == []:
                    parties_data[0][3] = None
                if parties_data[0][4] == []:
                    parties_data[0][4] = None
                # if len(parties_data[0])==4:
                #     # print('4 with token_type_ids')
                #     parties_data = [ [_data[0].to(self.device),_data[1].to(self.device),_data[2].to(self.device),_data[3].to(self.device)] for _data in parties_data]
                # elif len(parties_data[0])==3:
                #     # print('3 without token_type_ids')
                #     parties_data = [ [_data[0].to(self.device),_data[1].to(self.device),_data[2].to(self.device),None] for _data in parties_data]

                if self.args.task_type == "SequenceClassification" and self.num_classes > 1: # regression
                    gt_val_one_hot_label = self.label_to_one_hot(parties_data[0][1], self.num_classes)
                else:
                    gt_val_one_hot_label = parties_data[0][1]
                
                pred_list = []
                for ik in range(self.k-1): # Passive data local predict
                    # allocate data (data/label/attention_mask/token_type_ids)
                    input_shape = parties_data[ik][0].shape[:2]# parties_data[ik][0].size()

                    self.parties[ik].input_shape = input_shape
                    self.parties[ik].obtain_local_data(parties_data[ik][0])
                    self.parties[ik].gt_one_hot_label = gt_val_one_hot_label
                    self.parties[ik].local_batch_attention_mask = parties_data[ik][2]
                    self.parties[ik].local_batch_token_type_ids = parties_data[ik][3]
                        

                    if self.args.model_type == 'Bert':
                        _local_pred, _local_pred_detach , _local_attention_mask = self.parties[ik].give_pred() # , _input_shape
                        # print(' === transmit === ')
                        # print('pred_clone:',_local_pred_detach[0].shape, type(_local_pred_detach[1]) )
                        # print('local_batch_attention_mask:',parties_data[0][2].shape)
                        # print('local_batch_token_type_ids:',self.parties[self.k-1].local_batch_token_type_ids.shape)
                        # print('torch.tensor(input_shape) :',torch.tensor(input_shape) .shape)
                        pred_list.append( [_local_pred, _local_attention_mask])
                    
                    elif self.args.model_type == 'GPT2':
                        if self.args.task_type == 'SequenceClassification':
                            _local_pred, _local_pred_detach ,_local_sequence_lengths, _local_attention_mask= self.parties[ik].give_pred() # , _input_shape
                            pred_list.append( [_local_pred,_local_sequence_lengths,_local_attention_mask] )
                        elif self.args.task_type == 'CausalLM':
                            _local_pred, _local_pred_detach , _local_attention_mask= self.parties[ik].give_pred() # , _input_shape
                            pred_list.append( [_local_pred,_local_attention_mask] )
                        elif self.args.task_type == 'QuestionAnswering':
                            _local_pred, _local_pred_detach , _local_attention_mask= self.parties[ik].give_pred() # , _input_shape
                            pred_list.append( [_local_pred,_local_attention_mask] )

                    
                    elif self.args.model_type == 'Llama':
                        # print(' === transmit === ')
                        # print('pred_clone:',_local_pred_detach.shape)
                        # print('_local_decoder_attention_mask:',_local_decoder_attention_mask.shape)
                        # print('_local_sequence_lengths :',type(_local_sequence_lengths))
                        # print(_local_sequence_lengths.shape)
                        if self.args.task_type == 'SequenceClassification':
                            _local_pred, _local_pred_detach ,_local_sequence_lengths, _local_attention_mask= self.parties[ik].give_pred() # , _input_shape
                            pred_list.append( [_local_pred,_local_sequence_lengths,_local_attention_mask] )
                        elif self.args.task_type == 'CausalLM':
                            _local_pred, _local_pred_detach , _local_attention_mask= self.parties[ik].give_pred() # , _input_shape
                            pred_list.append( [_local_pred,_local_attention_mask] )
                        elif self.args.task_type == 'QuestionAnswering':
                            _local_pred, _local_pred_detach , _local_attention_mask= self.parties[ik].give_pred() # , _input_shape
                            pred_list.append( [_local_pred,_local_attention_mask] )
                     
                test_logit = self.parties[self.k-1].aggregate(pred_list, test="True")
                
                # full_logit = full_model(self.parties[0].local_batch_data, \
                # attention_mask = self.parties[0].local_batch_attention_mask, \
                # token_type_ids=self.parties[0].local_batch_token_type_ids)
                # full_logit = full_logit.logits

                if self.args.task_type == "SequenceClassification":
                    if self.num_classes == 1: # regression
                        predict_label = test_logit.detach().cpu()
                        actual_label = gt_val_one_hot_label.detach().cpu()
                        
                        # full_predict_label = full_pred.detach().cpu()

                        predict_label = torch.tensor( [ _.item() for _ in predict_label] )
                        actual_label = torch.tensor( [ _.item() for _ in actual_label] )

                        test_predict_labels.extend( list(predict_label) )
                        test_actual_labels.extend( list(actual_label) )

                        # test_full_predict_labels.extend( list(full_predict_label) )
                    else: # Classification
                        enc_predict_prob = test_logit
                        # full_enc_predict_prob = full_pred
                        # enc_predict_prob = F.softmax(test_logit, dim=-1)

                        predict_label = torch.argmax(enc_predict_prob, dim=-1)
                        actual_label = torch.argmax(gt_val_one_hot_label, dim=-1)
                        # full_predict_label = torch.argmax(full_enc_predict_prob, dim=-1)

                        test_preds.append(list(enc_predict_prob.detach().cpu().numpy()))
                        test_targets.append(list(gt_val_one_hot_label.detach().cpu().numpy()))
                        # full_test_preds.append(list(full_enc_predict_prob.detach().cpu().numpy()))

                        test_predict_labels.extend( list(predict_label.detach().cpu()) )
                        test_actual_labels.extend( list(actual_label.detach().cpu()) )
                        # test_full_predict_labels.extend( list(full_predict_label.detach().cpu()) )

                        sample_cnt += predict_label.shape[0]
                        suc_cnt += torch.sum(predict_label == actual_label).item()
                        # full_suc_cnt += torch.sum(full_predict_label == actual_label).item()

                elif self.args.task_type == "CausalLM":
                    # get logits of last hidden state
                    # print('test_logit:',test_logit.shape) # [batchsize, maxlength512, vocab_size32000]
                    # print('full_logit:',full_logit) # [batchsize, maxlength512, vocab_size32000]
                    next_token_logits = test_logit[:,-1] #[bs, 32000]
                    # print('next_token_logits:',next_token_logits.shape,next_token_logits)
                    
                    if self.args.dataset == "Lambada":
                        print('Lambada gt_val_one_hot_label:',gt_val_one_hot_label)
                        enc_predict_prob = nn.functional.softmax( next_token_logits,dim=-1)
                        predict_label = torch.argmax(enc_predict_prob, dim=-1) #[bs]
                        print('predict_label:',predict_label)
                        actual_label = gt_val_one_hot_label

                    else: # MMLU
                        choice_id_list = []
                        for choice in self.args.label_dict.keys():
                            choice_id_list.append(self.args.tokenizer(choice).input_ids[-1])
                            _id = self.args.tokenizer(choice).input_ids[-1]
                        # print('choice_id_list:',choice_id_list)
                        enc = next_token_logits[:,choice_id_list] # [bs, num_choice]
                        # print('enc:',enc.shape,enc)
                        enc_predict_prob = nn.functional.softmax( enc,dim=-1) # [bs, num_choice]
                        # print('enc_predict_prob:',enc_predict_prob.shape,enc_predict_prob) # [bs, num_choice]


                        predict_label = torch.argmax(enc_predict_prob, dim=-1) #[bs]
                        actual_label = torch.argmax(gt_val_one_hot_label, dim=-1)
                        # print('gt_val_one_hot_label:',type(gt_val_one_hot_label),gt_val_one_hot_label)


                    # test_preds.append(list(enc_predict_prob.detach().cpu().numpy()))
                    # test_targets.append(list(gt_val_one_hot_label.detach().cpu().numpy()))
                    # full_test_preds.append(list(full_enc_predict_prob.detach().cpu().numpy()))
                    # print('predict_label:',type(predict_label),predict_label)
                    # print('actual_label:',type(actual_label),actual_label)

                    test_predict_labels.extend( predict_label.detach().cpu().tolist() )
                    test_actual_labels.extend( actual_label.detach().cpu().tolist() )
                    # test_full_predict_labels.extend( list(full_predict_label.detach().cpu()) )

                    sample_cnt += predict_label.shape[0]
                    suc_cnt += torch.sum(predict_label == actual_label).item()
                
                elif self.args.task_type == "QuestionAnswering":
                    
                    def _get_best_indexes(logits, n_best_size=20):
                        """Get the n-best logits from a list."""
                        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
                        best_indexes = []
                        for i in range(len(index_and_score)):
                            if i >= n_best_size:
                                break
                            best_indexes.append(index_and_score[i][0])
                        return best_indexes
                    
                    def get_tokens(s):
                        if not s: return []
                        return normalize_answer(s).split()

                    def compute_exact(a_gold, a_pred):
                        return int(normalize_answer(a_gold) == normalize_answer(a_pred))

                    def compute_f1(a_gold, a_pred):
                        gold_toks = get_tokens(a_gold)
                        pred_toks = get_tokens(a_pred)
                        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
                        num_same = sum(common.values())
                        if len(gold_toks) == 0 or len(pred_toks) == 0:
                            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
                            return int(gold_toks == pred_toks)
                        if num_same == 0:
                            return 0
                        precision = 1.0 * num_same / len(pred_toks)
                        recall = 1.0 * num_same / len(gold_toks)
                        f1 = (2 * precision * recall) / (precision + recall)
                        return f1
                    
                    start_logits = test_logit.start_logits
                    end_logits = test_logit.end_logits

                    n_best_size = self.args.n_best_size
                    start_indexes =[ _get_best_indexes(_logits, n_best_size) for _logits in start_logits ]
                    end_indexes =[ _get_best_indexes(_logits, n_best_size) for _logits in end_logits ]

                    for i in range(start_logits.shape[0]):
                        # for each sample in this batch
                        _start_logits = start_logits[i]
                        _end_logits = end_logits[i]
                        _start_indexes = start_indexes[i]
                        _end_indexes = end_indexes[i]

                        ############ Gold ################
                        feature = parties_data[0][4]
                        feature_tokens = [_token[0] for _token in feature["tokens"]]

                        gold_start_indexs, gold_end_indexs = gt_val_one_hot_label[0] 
                        gold_ans = []
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
                        _exact_score_list = []
                        _f1_list = []
                        exact_score = 0
                        f1 = 0
                        # Get n best prediction text
                        nbest = []       
                        n_best_size = min(n_best_size,len(prelim_predictions))
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
                        for entry in nbest:
                            total_scores.append(entry.start_logit + entry.end_logit)
                            if not best_non_null_entry:
                                if entry.text:
                                    best_non_null_entry = entry
                            
                            # pred_ans_text = best_non_null_entry.text
                            pred_ans_text = entry.text
                            # Calculate exact_score/f1
                            # print('best pred:',pred_ans_text)
                            exact_score = max(exact_score, max(compute_exact(a, pred_ans_text) for a in gold_ans) )
                            f1 = max(f1, max(compute_f1(a, pred_ans_text) for a in gold_ans) )
                            # exact_score = compute_exact(gold_ans_text, pred_ans_text)
                            # f1 = compute_f1(gold_ans_text, pred_ans_text)
                        print('this batch:',exact_score,f1)
                        exact_score_list.append(exact_score)
                        f1_list.append(f1)
                else:
                    assert 1>2, "task_type not supported"
                
                del(parties_data) # remove from cuda
            
            if self.args.task_type == "QuestionAnswering":
                self.exact_score = np.mean(exact_score_list)
                self.f1 = np.mean(f1_list)
                exp_result = 'exact_score:{:.4f} f1:{:.4f}'.format(self.exact_score, self.f1)
                print(exp_result)
                return exp_result , self.exact_score

            if self.args.task_type == "SequenceClassification":
                if self.num_classes == 1:
                    self.test_mse = torch.mean((torch.tensor(test_predict_labels)-torch.tensor(test_actual_labels))**2).item()
                    #torch.nn.MSELoss()(  torch.tensor(test_predict_labels), torch.tensor(test_actual_labels) ).item()

                    self.test_pearson_corr = stats.pearsonr( torch.tensor(test_predict_labels), torch.tensor(test_actual_labels) )[0]
                    # full_test_pearson_corr = pearsonr( torch.tensor(test_full_predict_labels), torch.tensor(test_actual_labels) )[0]
                    self.test_spearmanr_corr = stats.spearmanr(torch.tensor(test_predict_labels), torch.tensor(test_actual_labels))[0]
                    postfix['test_mse'] = '{:.4f}%'.format(self.test_mse * 100)
                    postfix['test_pearson_corr'] = '{:.4f}%'.format(self.test_pearson_corr * 100)
                    
                    exp_result = 'test_mse:{:.4f} test_pearson_corr:{:.4f} test_spearmanr_corr:{:.4f}'.format(self.test_mse, self.test_pearson_corr, self.test_spearmanr_corr )
                    print(exp_result)
                    # print('Full pred pearson:',full_test_pearson_corr)
                    return exp_result , self.test_mse
                else:
                    print('test_predict_labels:',test_predict_labels[:20]) 
                    print('test_actual_labels:',test_actual_labels[:20]) 

                    self.test_acc = suc_cnt / float(sample_cnt) # ACC
                    # full_test_acc = full_suc_cnt / float(sample_cnt) # ACC

                    # test_preds = np.vstack(test_preds)
                    # test_targets = np.vstack(test_targets)
                    # self.test_auc = np.mean(multiclass_auc(test_targets, test_preds)) # AUC

                    self.test_mcc = matthews_corrcoef(np.array(test_predict_labels),np.array(test_actual_labels) ) # MCC

                    postfix['test_acc'] = '{:.2f}%'.format(self.test_acc * 100)
                    # postfix['test_auc'] = '{:.2f}%'.format(self.test_auc * 100)
                    postfix['test_mcc'] = '{:.2f}%'.format(self.test_mcc * 100)

                    exp_result = 'test_acc:{:.2f} test_mcc:{:.2f}'.format(self.test_acc, self.test_mcc )
                    print(exp_result)
                    # print('Full acc:',full_test_acc)
                    return exp_result , self.test_acc

            if self.args.task_type == "CausalLM":
                print('test_predict_labels:',test_predict_labels[:20]) 
                print('test_actual_labels:',test_actual_labels[:20]) 

                self.test_acc = suc_cnt / float(sample_cnt) # ACC
                # full_test_acc = full_suc_cnt / float(sample_cnt) # ACC

                # test_preds = np.vstack(test_preds)
                # test_targets = np.vstack(test_targets)
                # self.test_auc = np.mean(multiclass_auc(test_targets, test_preds)) # AUC

                self.test_mcc = matthews_corrcoef(np.array(test_predict_labels),np.array(test_actual_labels) ) # MCC

                postfix['test_acc'] = '{:.2f}%'.format(self.test_acc * 100)
                # postfix['test_auc'] = '{:.2f}%'.format(self.test_auc * 100)
                postfix['test_mcc'] = '{:.2f}%'.format(self.test_mcc * 100)

                exp_result = 'test_acc:{:.2f} test_mcc:{:.2f}'.format(self.test_acc, self.test_mcc )
                print(exp_result)
                # print('Full acc:',full_test_acc)
                return exp_result , self.test_acc


    def train_batch(self, parties_data, batch_label):
        '''
        batch_label: self.gt_one_hot_label   may be noisy
        '''
        encoder = self.args.encoder
        if self.args.apply_cae:
            assert encoder != None, "[error] encoder is None for CAE"
            _, gt_one_hot_label = encoder(batch_label)      
        else:
            gt_one_hot_label = batch_label

        # allocate data/attention mask to passive party
        for ik in range(self.k-1):
            self.parties[ik].obtain_local_data(parties_data[ik][0])
            self.parties[ik].gt_one_hot_label = self.gt_one_hot_label
            self.parties[ik].local_batch_attention_mask = parties_data[ik][2]
            self.parties[ik].local_batch_token_type_ids = parties_data[ik][3]


        # ====== normal vertical federated learning ======
        torch.autograd.set_detect_anomaly(True)
        # ======== Commu ===========
        if self.args.communication_protocol in ['Vanilla','Quantization','Topk'] or self.Q ==1 : # parallel FedBCD & noBCD situation
            for q in range(self.Q):
                if q == 0: 
                    # exchange info between party: local_prd/global_pred
                    self.pred_transmit() 
                    self.global_pred_transmit() 
                    self.global_loss_transmit()
                    
                    # update parameters for global trainable part
                    self.parties[self.k-1].global_backward() 
                    # for ik in range(self.k):
                    #     self.parties[ik].local_backward()
        else:
            assert 1>2 , 'Communication Protocol not provided'
        # ============= Commu ===================
        
        # ###### Noisy Label Attack #######
        # convert back to clean label to get true acc
        if self.args.apply_nl==True:
            real_batch_label = self.clean_one_hot_label
        else:
            real_batch_label = batch_label
        # ###### Noisy Label Attack #######

        pred = self.parties[self.k-1].global_pred
        loss = self.parties[self.k-1].global_loss
        predict_prob = F.softmax(pred, dim=-1)
        if self.args.apply_cae:
            predict_prob = encoder.decode(predict_prob)

        suc_cnt = torch.sum(torch.argmax(predict_prob, dim=-1) == torch.argmax(real_batch_label, dim=-1)).item()
        train_acc = suc_cnt / predict_prob.shape[0]
        
        return loss.item(), train_acc

    def train(self):

        print_every = 1

        for ik in range(self.k):
            self.parties[ik].prepare_data_loader(batch_size=self.batch_size)

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


        start_time = time.time()
        for i_epoch in range(self.epochs):
            self.current_epoch = i_epoch
            postfix = {'train_loss': 0.0, 'train_acc': 0.0, 'test_acc': 0.0}
            i = -1

            data_loader_list = [self.parties[ik].train_loader for ik in range(self.k-1)]

            self.current_step = 0
            for parties_data in zip(*data_loader_list):
                if len(parties_data[0])==4:
                    # print('4 with token_type_ids')
                    self.parties_data = [ [_data[0].to(self.device),_data[1].to(self.device),_data[2].to(self.device),_data[3].to(self.device)] for _data in parties_data]
                elif len(parties_data[0])==3:
                    # print('3 without token_type_ids')
                    self.parties_data = [ [_data[0].to(self.device),_data[1].to(self.device),_data[2].to(self.device),None] for _data in parties_data]
                # self.parties_data = [ [_data[0].to(self.device),_data[1].to(self.device),_data[2].to(self.device),_data[3].to(self.device)] for _data in parties_data]

                # print('self.parties_data[0][0].shape:',self.parties_data[0][0].shape)
                # print('self.parties_data[0][1].shape:',self.parties_data[0][1].shape)
                # print('self.parties_data[0][2].shape:',self.parties_data[0][2].shape)

                if self.num_classes == 1: # regression
                    self.gt_one_hot_label = parties_data[0][1]
                    # print('gt_val_one_hot_label:',type(gt_val_one_hot_label),gt_val_one_hot_label.shape)
                else:
                    self.gt_one_hot_label = self.label_to_one_hot(parties_data[0][1], self.num_classes)
                
                # ###### Noisy Label Attack ######
                if self.args.apply_nl==True:
                    # noisy label
                    self.clean_one_hot_label = self.gt_one_hot_label
                    self.gt_one_hot_label = add_noise(self.args, self.gt_one_hot_label)
                # ###### Noisy Label Attack ######

                i += 1
                self.parties[self.k-1].global_model.train()
                
                # ====== train batch (start) ======
                enter_time = time.time()
                self.loss, self.train_acc = self.train_batch(self.parties_data,self.gt_one_hot_label)
                exit_time = time.time()
                total_time += (exit_time-enter_time)
                self.num_total_comms = self.num_total_comms + 1
                # if self.num_total_comms % 10 == 0:
                #     print(f"total time for {self.num_total_comms} communication is {total_time}")
                if self.train_acc > STOPPING_ACC[str(self.args.dataset)] and flag == 0:
                    self.stopping_time = total_time
                    self.stopping_iter = self.num_total_comms
                    self.stopping_commu_cost = self.communication_cost
                    flag = 1
                # ====== train batch (end) ======

                self.current_step = self.current_step + 1

                del(self.parties_data) # remove from cuda
                del(parties_data)

            # if self.args.apply_attack == True:
            #     if (self.args.attack_name in LABEL_INFERENCE_LIST) and i_epoch==1:
            #         print('Launch Label Inference Attack, Only train 1 epoch')
            #         break    

            # self.trained_models = self.save_state(True)
            # if self.args.save_model == True:
            #     self.save_trained_models()

            # LR decay
            self.LR_Decay(i_epoch)
            # LR record
            # if self.args.k == 2:
            #     LR_passive_list.append(self.parties[0].give_current_lr())
                # LR_active_list.append(self.parties[1].give_current_lr())

            # validation
            if (i + 1) % print_every == 0:
                print("validate and test")
                self.parties[self.k-1].global_model.eval()
                
                suc_cnt = 0
                sample_cnt = 0

                test_preds = []
                test_targets = []
                test_predict_labels = []
                test_actual_labels = []

                scores = []
                targets = []
                with torch.no_grad():
                    data_loader_list = [self.parties[ik].test_loader for ik in range(self.k-1)]
                    
                    for parties_data in zip(*data_loader_list):
                        parties_data = [ [_data[0].to(self.device),_data[1].to(self.device),_data[2].to(self.device)] for _data in parties_data]
                        
                        if self.num_classes == 1: # regression
                            gt_val_one_hot_label = parties_data[0][1]
                            # print('gt_val_one_hot_label:',type(gt_val_one_hot_label),gt_val_one_hot_label.shape)
                        else:
                            gt_val_one_hot_label = self.label_to_one_hot(parties_data[0][1], self.num_classes)
                        
                        pred_list = []
                        for ik in range(self.k-1): # Passive data local predict
                            input_shape = parties_data[ik][0].shape[:2]
                            _local_pred = self.parties[ik].local_model(\
                                parties_data[ik][0],attention_mask=parties_data[ik][2],token_type_ids = parties_data[ik][3])

                            self.parties[ik].input_shape = input_shape
                            self.parties[ik].obtain_local_data(parties_data[ik][0])
                            self.parties[ik].gt_one_hot_label = gt_val_one_hot_label
                            self.parties[ik].local_batch_attention_mask = parties_data[ik][2]
                            self.parties[ik].local_batch_token_type_ids = parties_data[ik][3]
                                
                            self.parties[self.k-1].input_shape = input_shape
                            # self.parties[self.k-1].local_batch_attention_mask = parties_data[0][2]
                            self.parties[self.k-1].local_batch_token_type_ids = parties_data[0][3]

                            if self.args.model_type == 'Bert':
                                _local_pred, _local_pred_detach, _local_attention_mask  = self.parties[ik].give_pred() # , _input_shape
                                pred_list.append( [_local_pred, _local_attention_mask])
                            
                            elif self.args.model_type == 'GPT2':
                                if self.args.task_type == 'SequenceClassification':
                                    _local_pred, _local_pred_detach,_local_sequence_lengths, _local_attention_mask= self.parties[ik].give_pred() # , _input_shape
                                    pred_list.append( [_local_pred,_local_sequence_lengths,_local_attention_mask] )
                                elif self.args.task_type == 'CausalLM':
                                    _local_pred, _local_pred_detach , _local_attention_mask= self.parties[ik].give_pred() # , _input_shape
                                    pred_list.append( [_local_pred,_local_attention_mask] )

                            elif self.args.model_type == 'Llama':
                                if self.args.task_type == 'SequenceClassification':
                                    _local_pred, _local_pred_detach , _local_sequence_lengths, _local_attention_mask= self.parties[ik].give_pred() # , _input_shape
                                    pred_list.append( [_local_pred,_local_sequence_lengths,_local_attention_mask] )
                                elif self.args.task_type == 'CausalLM':
                                    _local_pred, _local_pred_detach , _local_attention_mask= self.parties[ik].give_pred() # , _input_shape
                                    pred_list.append( [_local_pred,_local_attention_mask] )

                            pred_list.append(_local_pred)
                        
                        test_logit = self.parties[self.k-1].aggregate(pred_list, test="True")
                        # print('test_logit:',type(test_logit),test_logit.shape)
                        
                        enc_predict_prob = F.softmax(test_logit, dim=-1)

                        predict_label = torch.argmax(enc_predict_prob, dim=-1)
                        actual_label = torch.argmax(gt_val_one_hot_label, dim=-1)

                        test_preds.append(list(enc_predict_prob.detach().cpu().numpy()))
                        test_targets.append(list(gt_val_one_hot_label.detach().cpu().numpy()))

                        test_predict_labels.extend( list(predict_label.detach().cpu()) )
                        test_actual_labels.extend( list(actual_label.detach().cpu()) )
            
                        sample_cnt += predict_label.shape[0]
                        suc_cnt += torch.sum(predict_label == actual_label).item()
                        del(parties_data) # remove from cuda

                    self.test_acc = suc_cnt / float(sample_cnt)
                    
                    test_preds = np.vstack(test_preds)
                    test_targets = np.vstack(test_targets)
                    self.test_auc = np.mean(multiclass_auc(test_targets, test_preds))

                    self.test_mcc = matthews_corrcoef(np.array(test_predict_labels),np.array(test_actual_labels) ) # MCC

                    postfix['train_loss'] = self.loss
                    postfix['train_acc'] = '{:.2f}%'.format(self.train_acc * 100)
                    postfix['test_acc'] = '{:.2f}%'.format(self.test_acc * 100)
                    postfix['test_auc'] = '{:.2f}%'.format(self.test_auc * 100)
                    postfix['test_mcc'] = '{:.2f}%'.format(self.test_mcc * 100)
                    
                    exp_result = 'Epoch {}% \t train_loss:{:.2f} train_acc:{:.2f} test_acc:{:.2f} test_auc:{:.2f} test_mcc:{:.2f}'.format(
                        i_epoch, self.loss, self.train_acc, self.test_acc, self.test_auc, self.test_mcc)
                    print(exp_result)
                    self.final_epoch = i_epoch
        
        exp_result = 'train_loss:{:.2f} train_acc:{:.2f} test_acc:{:.2f} test_auc:{:.2f} test_mcc:{:.2f}'.format(
                        self.loss, self.train_acc, self.test_acc, self.test_auc, self.test_mcc)
                    
        save_path = self.args.exp_res_dir + '/pretrained_trainable_layer/' 
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(self.parties[self.k-1].global_model.trainable_layer.state_dict(),\
            save_path + f'/model={self.args.model_list[str(0)]["type"]}_lr{self.args.main_lr}_bs{self.args.batch_size}_acc{str(self.test_acc)}.pth')
        print(save_path + f'/model={self.args.model_list[str(0)]["type"]}_lr{self.args.main_lr}_bs{self.args.batch_size}_acc{str(self.test_acc)}.pth')
        
        # self.final_state = self.save_state(True) 
        # self.final_state.update(self.save_state(False)) 
        # self.final_state.update(self.save_party_data()) 


        return exp_result, self.test_acc #, self.stopping_iter, self.stopping_time, self.stopping_commu_cost



    def save_state(self, BEFORE_MODEL_UPDATE=True):
        if BEFORE_MODEL_UPDATE:
            return {
                "model": [copy.deepcopy(self.parties[ik].local_model) for ik in range(self.args.k)],
                "global_model":copy.deepcopy(self.parties[self.args.k-1].global_model),
                # type(model) = <class 'xxxx.ModelName'>
                "model_names": [str(type(self.parties[ik].local_model)).split('.')[-1].split('\'')[-2] for ik in range(self.args.k)]+[str(type(self.parties[self.args.k-1].global_model)).split('.')[-1].split('\'')[-2]]
            
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
                "global_pred":self.parties[self.k-1].global_pred,
                "final_model": [copy.deepcopy(self.parties[ik].local_model) for ik in range(self.args.k)],
                "final_global_model":copy.deepcopy(self.parties[self.args.k-1].global_model),
                
            }

    def save_party_data(self):
        return {
            "aux_data": [copy.deepcopy(self.parties[ik].aux_data) for ik in range(self.k)],
            "train_data": [copy.deepcopy(self.parties[ik].train_data) for ik in range(self.k)],
            "test_data": [copy.deepcopy(self.parties[ik].test_data) for ik in range(self.k)],
            "aux_label": [copy.deepcopy(self.parties[ik].aux_label) for ik in range(self.k)],
            "train_label": [copy.deepcopy(self.parties[ik].train_label) for ik in range(self.k)],
            "test_label": [copy.deepcopy(self.parties[ik].test_label) for ik in range(self.k)],
            "aux_attribute": [copy.deepcopy(self.parties[ik].aux_attribute) for ik in range(self.k)],
            "train_attribute": [copy.deepcopy(self.parties[ik].train_attribute) for ik in range(self.k)],
            "test_attribute": [copy.deepcopy(self.parties[ik].test_attribute) for ik in range(self.k)],
            "aux_loader": [copy.deepcopy(self.parties[ik].aux_loader) for ik in range(self.k)],
            "train_loader": [copy.deepcopy(self.parties[ik].train_loader) for ik in range(self.k)],
            "test_loader": [copy.deepcopy(self.parties[ik].test_loader) for ik in range(self.k)],
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
