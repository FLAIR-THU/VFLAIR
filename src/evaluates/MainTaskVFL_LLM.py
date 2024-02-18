import sys, os
sys.path.append(os.pardir)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
from utils.squad_utils import  normalize_answer,_get_best_indexes, get_tokens, compute_exact, compute_f1


from evaluates.attacks.attack_api import AttackerLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoModelForCausalLM

from load.LoadModels import MODEL_PATH

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

    def apply_defense_on_transmission(self, pred_detach):
        ########### Defense applied on pred transmit ###########
        if self.args.apply_defense == True and self.args.apply_dp == True :
            # print('pre pred_detach:',type(pred_detach),pred_detach.shape) # torch.size bs,12,768 intermediate
            pred_detach = torch.stack( self.launch_defense(pred_detach, "pred")  )
            # print('after pred_detach:',type(pred_detach),pred_detach.shape) # torch.size bs,12,768 intermediate
        return pred_detach
    
    def apply_communication_protocol_on_transmission(self, pred_detach):
        ########### communication_protocols ###########
        if self.args.communication_protocol in ['Quantization','Topk']:
            pred_detach = compress_pred( self.args ,pred_detach , self.parties[ik].local_gradient,\
                            self.current_epoch, self.current_step).to(self.args.device)
        return pred_detach


    def pred_transmit(self): 
        '''
        Active party gets pred from passive parties
        '''
        all_pred_list = []
        for ik in range(self.k-1):
            # give pred
            if self.args.model_type == 'Bert':
                if self.args.task_type == 'SequenceClassification':
                    pred, pred_detach, attention_mask  = self.parties[ik].give_pred() # , _input_shape
                elif self.args.task_type == 'QuestionAnswering':
                    pred, pred_detach, attention_mask  = self.parties[ik].give_pred() # , _input_shape
                else:
                    assert 1>2, "task type not supported for finetune"
            elif self.args.model_type == 'Llama':
                if self.args.task_type == 'SequenceClassification':
                    pred, pred_detach , sequence_lengths, attention_mask = self.parties[ik].give_pred() # , _input_shape
                elif self.args.task_type == 'CausalLM':
                    pred, pred_detach , attention_mask = self.parties[ik].give_pred() # , _input_shape
                elif self.args.task_type == 'QuestionAnswering':
                    pred, pred_detach , attention_mask = self.parties[ik].give_pred() # , _input_shape
                else:
                    assert 1>2, "task type not supported for finetune"

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
                    pred, pred_detach , sequence_lengths, attention_mask = self.parties[ik].give_pred()
                elif self.args.task_type == 'CausalLM':
                    pred, pred_detach , attention_mask = self.parties[ik].give_pred() 
                elif self.args.task_type == 'QuestionAnswering':
                    pred, pred_detach , attention_mask = self.parties[ik].give_pred() 
                else:
                    assert 1>2, "task type not supported for finetune"

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
                    pred_list = [pred_clone,attention_mask]
                    self.parties[self.k-1].receive_pred(pred_list, ik)  
                    self.parties[ik].update_local_pred(pred_clone)
                    self.communication_cost += get_size_of(pred_clone)+\
                        get_size_of(attention_mask) #MB
                elif self.args.task_type == 'QuestionAnswering':
                    pred_list = [pred_clone,attention_mask]
                    self.parties[self.k-1].receive_pred(pred_list, ik)  
                    self.parties[ik].update_local_pred(pred_clone)
                    self.communication_cost += get_size_of(pred_clone)+get_size_of(attention_mask) #MB
            elif self.args.model_type == 'Llama':
                if self.args.task_type == 'SequenceClassification':
                    pred_list = [pred_clone,sequence_lengths,attention_mask]
                    self.parties[self.k-1].receive_pred(pred_list, ik) 
                    self.parties[ik].update_local_pred(pred_clone)
                    self.communication_cost += get_size_of(pred_clone)+\
                    get_size_of(sequence_lengths)+\
                    get_size_of( attention_mask ) #MB
                elif self.args.task_type == 'CausalLM':
                    pred_list = [pred_clone,attention_mask]
                    self.parties[self.k-1].receive_pred(pred_list, ik)  
                    self.parties[ik].update_local_pred(pred_clone)
                    self.communication_cost += get_size_of(pred_clone)+\
                    get_size_of( attention_mask ) #MB
                elif self.args.task_type == 'QuestionAnswering':
                    pred_list = [pred_clone,attention_mask]
                    self.parties[self.k-1].receive_pred(pred_list, ik)  
                    self.parties[ik].update_local_pred(pred_clone)
                    self.communication_cost += get_size_of(pred_clone)+get_size_of( attention_mask ) #MB
            elif self.args.model_type == 'GPT2':
                if self.args.task_type == 'SequenceClassification':
                    pred_list = [pred_clone,sequence_lengths,attention_mask]
                    self.parties[self.k-1].receive_pred(pred_list, ik) 
                    self.parties[ik].update_local_pred(pred_clone)
                    self.communication_cost += get_size_of(pred_clone)+get_size_of(attention_mask)+get_size_of(sequence_lengths)  #MB
                elif self.args.task_type == 'CausalLM':
                    pred_list = [pred_clone,attention_mask]
                    self.parties[self.k-1].receive_pred( pred_list, ik) 
                    self.parties[ik].update_local_pred(pred_clone)
                    self.communication_cost += get_size_of(pred_clone)+get_size_of(attention_mask) #MB
                elif self.args.task_type == 'QuestionAnswering':
                    pred_list = [pred_clone,attention_mask]
                    self.parties[self.k-1].receive_pred( pred_list , ik) 
                    self.parties[ik].update_local_pred(pred_clone)
                    self.communication_cost += get_size_of(pred_clone)+get_size_of(attention_mask) #MB
            
            all_pred_list.append( pred_list )

        self.all_pred_list = all_pred_list
        return all_pred_list

    def global_pred_transmit(self):
        # active party give global pred to passive party
        final_pred = self.parties[self.k-1].aggregate(self.parties[self.k-1].pred_received, test="True")
        
        for ik in range(self.k-1):
            # self.communication_cost += get_size_of(final_pred)
            self.parties[ik].global_pred = final_pred

        return final_pred
    
    def local_gradient_transmit(self):
        for ik in range(self.k-1):
            if self.parties[ik].local_model_optimizer != None:
                passive_local_gradient= self.parties[self.k-1].cal_passive_local_gradient(ik)
                self.parties[ik].local_gradient = passive_local_gradient

    def global_gradient_transmit(self, final_pred):
        global_loss = self.parties[0].cal_loss(final_pred)  # raw global loss

        # add adversary loss
        # ============= Adversarial Training =============
        # update adversary
        # if self.args.apply_adversarial:
        #     self.imagined_adversary_optimizer.zero_grad()

        #     adversary_loss = 0
        #     mapping_distance = 0
        #     for ik in range(self.k-1):
        #         intermediate = all_pred_list[ik][0] # bs, seq, embed_dim768
        #         adversary_recovered_embedding = self.imagined_adversary(intermediate)
        #         real_embedding =  self.parties[ik].local_model.embedding_output
        #         adversary_loss += self.adversary_crit(adversary_recovered_embedding, real_embedding)

        #         mapping_distance += torch.norm( self.parties[ik].local_model.adversarial_model.origin_output -\
        #          self.parties[ik].local_model.adversarial_model.adversarial_output , p=2)

        #     adversary_loss = adversary_loss / intermediate.shape[0]

        #     adversary_loss.backward(retain_graph = True)

        #     self.imagined_adversary_optimizer.step()

        #     self.parties[0].adversary_loss = adversary_loss
        #     self.parties[0].mapping_distance = mapping_distance

        #     # renew loss function
        #     self.parties[0].global_loss = self.parties[0].global_loss \
        #                                         + self.adversary_lambda * mapping_distance \
        #                                         - adversary_loss
        # ============= Adversarial Training =============

        # update passive party local model (if needed)
        # for ik in range(self.k-1):
        #     self.parties[ik].local_backward()

        global_gradients = self.parties[0].cal_global_gradient(global_loss, final_pred)
        self.communication_cost += get_size_of(global_gradients)

        self.parties[self.k-1].global_loss = self.parties[0].global_loss
        self.parties[self.k-1].global_gradients = self.parties[0].global_gradients

    def qa_inference(self):
        # QA
        exact_score_list = []
        f1_list = []

        for ik in range(self.k - 1):
            # Passive data local predict
            exact_scores, f1s, _ = self.parties[ik].predict()
            exact_score_list.extend(exact_scores)
            f1_list.extend(f1s)

        exp_result, exact_score = self.parties[self.k - 1].mean_local((exact_score_list, f1_list))

        self.test_acc = exact_score

        self.final_state = self.save_state(False)
        self.final_state.update(self.save_party_data())

        return exp_result, self.test_acc

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
            # print('test_predict_labels:',type(test_predict_labels),len(test_predict_labels)) # list

            suc_cnt = torch.sum(torch.tensor(test_predict_labels) == \
            torch.tensor(test_actual_labels)).item()
            # print('test_predict_labels:',test_predict_labels[:20])
            # print('test_actual_labels:',test_actual_labels[:20])

            self.test_acc = suc_cnt / float(total_sample_cnt)  # ACC
            # full_test_acc = full_suc_cnt / float(sample_cnt) # ACC

            # test_preds = np.vstack(test_preds)
            # test_targets = np.vstack(test_targets)
            # self.test_auc = np.mean(multiclass_auc(test_targets, test_preds)) # AUC

            self.test_mcc = matthews_corrcoef(np.array(test_predict_labels), np.array(test_actual_labels))  # MCC

            postfix['test_acc'] = '{:.2f}%'.format(self.test_acc * 100)
            # postfix['test_auc'] = '{:.2f}%'.format(self.test_auc * 100)
            postfix['test_mcc'] = '{:.2f}%'.format(self.test_mcc * 100)

            exp_result = 'test_acc:{:.2f} test_mcc:{:.2f}'.format(self.test_acc, self.test_mcc)
            print(exp_result)

            self.final_state = self.save_state(False)
            self.final_state.update(self.save_party_data())

            print('self.test_acc:',self.test_acc)
            print('======== seq_inference ==========')

            return exp_result, self.test_acc


    def inference(self, inference_data = 'test'):
        # current_model_type = self.args.model_list['0']['type']
        # full_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH[current_model_type]).to(self.args.device)

        # print(' ========= Inference ==========')
        postfix = {'test_acc': 0.0}
        for ik in range(self.k):
            self.parties[ik].prepare_data_loader()
        self.parties[self.k-1].global_model.eval()
        # self.final_state = self.save_state(False) 
        # self.final_state.update(self.save_party_data()) 

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
        target_word_list = []
        predict_word_list = []

        scores = []
        targets = []

        if self.args.task_type == "QuestionAnswering":
            exp_result, main_task_result = self.qa_inference()
            return exp_result, main_task_result

        if self.args.task_type == "SequenceClassification":
            # exp_result, self.test_acc =
            exp_result, main_task_result = self.seq_inference()
            return exp_result, main_task_result


        with torch.no_grad():
            data_loader_list = [self.parties[ik].test_loader for ik in range(self.k-1)] # passive party's loaders

            for parties_data in zip(*data_loader_list):
                # parties_data[0]:  bs *( data, label, mask, token_type_ids, feature(forQA))
                _parties_data = []
                for party_id in range(len(parties_data)): # iter through each passive party
                    batch_input_ids = []
                    batch_label = []
                    batch_attention_mask = []
                    batch_token_type_ids = []
                    batch_feature = []
                    for bs_id in range(len(parties_data[party_id])):
                        # Input_ids
                        batch_input_ids.append( parties_data[party_id][bs_id][0].tolist() )
                        # Attention Mask
                        batch_attention_mask.append( parties_data[party_id][bs_id][2].tolist()  )

                        # ptoken_type_ids
                        if parties_data[party_id][bs_id][3] == []:
                            batch_token_type_ids = None
                        else:
                            batch_token_type_ids.append( parties_data[party_id][bs_id][3].tolist()  )

                        # feature (for QuestionAnswering only)
                        if parties_data[party_id][bs_id][4] == []:
                            batch_feature = None
                        else:
                            batch_feature.append( parties_data[party_id][bs_id][4] )
                        
                        # Label
                        if type(parties_data[party_id][bs_id][1]) != str:
                            batch_label.append( parties_data[party_id][bs_id][1].tolist()  )
                        else:
                            batch_label.append( parties_data[party_id][bs_id][1]  )

                    batch_input_ids = torch.tensor(batch_input_ids).to(self.device)
                    batch_attention_mask = torch.tensor(batch_attention_mask).to(self.device)
                    if batch_token_type_ids != None:
                        batch_token_type_ids = torch.tensor(batch_token_type_ids).to(self.device) 
                    if type( batch_label[0] ) != str:
                        batch_label = torch.tensor(batch_label).to(self.device)
                    
                    _parties_data.append([batch_input_ids,batch_label,batch_attention_mask,batch_token_type_ids,batch_feature] )
                
                parties_data = _parties_data
                # parties_data[0][0] : bs * sample_input_ids
                # parties_data[0][1] : bs * sample_label
                # parties_data[0][2] : bs * sample_attention_mask
                # parties_data[0][3] : bs * sample_token_type_ids
                # parties_data[0][4] : bs * sample_feature (dict)


                # parties_data[0]: (data, label, mask, token_type_ids, doc_tokens(forQA) )
                # parties_data[1]: (None,None,None,None,None)  no data for active party
                # parties_data = [ [_data[0],_data[1],_data[2],_data[3],_data[4]] for _data in parties_data]
                # if parties_data[0][3] == []:
                #     parties_data[0][3] = None
                # if parties_data[0][4] == []:
                #     parties_data[0][4] = None

                if self.args.task_type == "SequenceClassification" and self.num_classes > 1: # classification
                    gt_one_hot_label = self.label_to_one_hot(parties_data[0][1], self.num_classes)
                elif self.args.task_type == "QuestionAnswering":
                    gt_one_hot_label = list(parties_data[0][1])
                else:
                    gt_one_hot_label = parties_data[0][1]

                # pred_list = []
                for ik in range(self.k-1): # Passive data local predict
                    # allocate data (data/label/attention_mask/token_type_ids)
                    input_shape = parties_data[ik][0].shape[:2]# batchsize, seq_length
                    self.parties[ik].input_shape = input_shape
                    self.parties[ik].obtain_local_data(parties_data[ik][0], parties_data[ik][2], parties_data[ik][3])
                    self.parties[ik].gt_one_hot_label = gt_one_hot_label

                pred_list = self.pred_transmit()
                     
                test_logit = self.parties[self.k-1].aggregate(pred_list, test="True")
                # print('test_logit:',test_logit.shape)
                
                if self.args.task_type == "SequenceClassification":
                    if self.num_classes == 1: # regression
                        predict_label = test_logit.detach().cpu()
                        actual_label = gt_one_hot_label.detach().cpu()
                        
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
                        actual_label = torch.argmax(gt_one_hot_label, dim=-1)
                        # full_predict_label = torch.argmax(full_enc_predict_prob, dim=-1)

                        test_preds.append(list(enc_predict_prob.detach().cpu().numpy()))
                        test_targets.append(list(gt_one_hot_label.detach().cpu().numpy()))
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
                    next_token_logits = test_logit[:,-1] #[bs, 32000]
                    
                    if self.args.dataset == "Lambada":
                        # generated, generate_text = self.parties[self.k-1].generate(pred_list, test="True")
                        # print('gt_one_hot_label:',type(gt_one_hot_label),gt_one_hot_label.shape)

                        # print('gt_one_hot_label:',type(gt_one_hot_label),gt_one_hot_label) # list of target tokens
                        target_word = [ normalize_answer(_p) for _p in gt_one_hot_label] # list of normalized tokens
                        # print('target_word:',type(target_word),len(target_word),target_word) # 
                        target_word_list.extend( target_word )
                        # print('target_word_list:',type(target_word_list),len(target_word_list),target_word_list)

                        # print('gt_one_hot_label:',type(gt_one_hot_label),gt_one_hot_label)
                        # target_word = gt_one_hot_label[0]
                        # target_word_list.append(normalize_answer(target_word))
                        # print('target_word:',target_word)


                        enc_predict_prob = nn.functional.softmax( next_token_logits,dim=-1)
                        
                        if self.args.metric_type == "best_pred":
                            predict_label = torch.argmax(enc_predict_prob, dim=-1) #[bs]
                            predict_word = [ self.args.tokenizer.decode([_best_id]) for _best_id in predict_label.tolist() ]
                            predict_word = [ normalize_answer(_p) for _p in predict_word]
                            predict_word_list.extend(predict_word) # predict_word: bs * best_pred
                        elif self.args.metric_type == "n_best":
                            logit_list, index_list = torch.sort(enc_predict_prob, descending=True)
                            # print('index_list:',index_list.shape)
                            predict_label = index_list[:,:self.args.n_best_size]
                            # print('predict_label:',predict_label.shape)
                            for _bs in range(predict_label.shape[0]): # each batch
                                predict_word = [self.args.tokenizer.decode( [_label]) for _label in predict_label[_bs].tolist()]
                                predict_word = [ normalize_answer(_p) for _p in predict_word]
                                predict_word_list.append(predict_word) # predict_word: list of n best for this batch
                                # print('predict_word:',predict_word)
                            
                    else: # MMLU
                        choice_id_list = []
                        for choice in self.args.label_dict.keys():
                            choice_id_list.append(self.args.tokenizer(choice).input_ids[-1])
                            _id = self.args.tokenizer(choice).input_ids[-1]
                        enc = next_token_logits[:,choice_id_list] # [bs, num_choice]
                        enc_predict_prob = nn.functional.softmax( enc,dim=-1) # [bs, num_choice]
                        
                        predict_label = torch.argmax(enc_predict_prob, dim=-1) #[bs]
                        actual_label = gt_one_hot_label #torch.argmax(gt_one_hot_label, dim=-1)

                        test_predict_labels.extend( predict_label.detach().cpu().tolist() )
                        test_actual_labels.extend( actual_label.detach().cpu().tolist() )
                        # test_full_predict_labels.extend( list(full_predict_label.detach().cpu()) )

                        sample_cnt += predict_label.shape[0]
                        suc_cnt += torch.sum(predict_label == actual_label).item()

                else:
                    assert 1>2, "task_type not supported"
                
                del(parties_data) # remove from cuda
                # break

            if self.args.task_type == "CausalLM":
                # print('target_word:',target_word[:2]) 
                # print('predict_word_list:',predict_word_list[:2]) 

                if self.args.metric_type == "best_pred":
                    suc_cnt = 0
                    for i in range(len(target_word_list)):
                        if target_word_list[i] == predict_word_list[i]:
                            suc_cnt+=1
                    self.test_acc = suc_cnt / float(len(target_word_list)) # ACC
                elif self.args.metric_type == "n_best":
                    suc_cnt = 0
                    for i in range(len(target_word_list)):
                        if target_word_list[i] in predict_word_list[i]:
                            suc_cnt+=1
                    self.test_acc = suc_cnt / float(len(target_word_list)) # ACC
                else:
                    assert 1>2, 'metric type not supported'

                postfix['test_acc'] = '{:.2f}%'.format(self.test_acc * 100)
                # postfix['test_auc'] = '{:.2f}%'.format(self.test_auc * 100)
                # postfix['test_mcc'] = '{:.2f}%'.format(self.test_mcc * 100)

                exp_result = 'test_acc:{:.2f}'.format(self.test_acc)
                print(exp_result)

                self.final_state = self.save_state(False) 
                self.final_state.update(self.save_party_data()) 

                return exp_result , self.test_acc

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
        for i_epoch in range(self.epochs):
            self.current_epoch = i_epoch
            postfix = {'train_loss': 0.0, 'train_acc': 0.0, 'test_acc': 0.0}
            i = -1

            for ik in range(self.k - 1):
                self.loss, self.train_acc = self.parties[ik].train(i_epoch)

            # validation
            if (i + 1) % print_every == 0:
                print("validate and test")
                self.parties[self.k-1].global_model.eval()

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
                    if self.args.apply_adversarial:
                        print(f'adversarial_model_loss:{self.parties[0].adversarial_model_loss.item()} adversary_attack_loss:{self.parties[0].adversary_attack_loss.item()}')

                    if (i+1) % 10 == 0:
                        if self.args.apply_adversarial:
                            if last_adversarial_model_loss < self.parties[0].adversarial_model_loss.item():
                                self.final_epoch = i_epoch
                                break
                            last_adversarial_model_loss = self.parties[0].adversarial_model_loss.item()

                    self.final_epoch = i_epoch
        
        # exp_result = f'train_loss:{:.2f} train_acc:{:.2f} test_acc:{:.2f} final_epoch:{}'.format(
        #                 self.loss, self.train_acc, self.test_acc, self.final_epoch)

        exp_result = f'train_loss:{self.loss} train_acc:{self.train_acc} test_acc:{self.test_acc} final_epoch:{self.final_epoch}'
                    
        # save_path = self.args.exp_res_dir + '/pretrained_trainable_layer/' 
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # torch.save(self.parties[self.k-1].global_model.trainable_layer.state_dict(),\
        #     save_path + f'/model={self.args.model_list[str(0)]["type"]}_lr{self.args.main_lr}_bs{self.args.batch_size}_acc{str(self.test_acc)}.pth')
        # print(save_path + f'/model={self.args.model_list[str(0)]["type"]}_lr{self.args.main_lr}_bs{self.args.batch_size}_acc{str(self.test_acc)}.pth')
        
        # self.final_state = self.save_state() 
        # self.final_state.update(self.save_state(False)) 
        self.final_state = self.save_state(False) 
        self.final_state.update(self.save_party_data()) 

        return exp_result, self.test_acc #, self.stopping_iter, self.stopping_time, self.stopping_commu_cost


    def save_state(self, BEFORE_MODEL_UPDATE=True):
        if BEFORE_MODEL_UPDATE:
            return {
                "model": [copy.deepcopy(self.parties[ik].local_model) for ik in range(self.args.k)],
                "global_model":copy.deepcopy(self.parties[self.args.k-1].global_model),
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
