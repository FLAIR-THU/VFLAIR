import sys, os
sys.path.append(os.pardir)
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
import numpy as np
import copy
import pickle 
import matplotlib.pyplot as plt
import itertools 
from scipy import optimize

from evaluates.attacks.attacker import Attacker
from models.global_models import *
from utils.basic_functions import cross_entropy_for_onehot, append_exp_res
from dataset.party_dataset import PassiveDataset, PassiveDataset_LLM
from dataset.party_dataset import ActiveDataset

from evaluates.defenses.defense_functions import LaplaceDP_for_pred,GaussianDP_for_pred

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

class custom_AE(nn.Module):
    def __init__(self, latent_dim, target_dim):
        super(custom_AE,self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_dim, 600), 
            nn.LayerNorm(600),
            nn.ReLU(),
            
            nn.Linear(600, 200), 
            nn.LayerNorm(200),
            nn.ReLU(),
            
            nn.Linear(200, 100),
            nn.LayerNorm(100),
            nn.ReLU(),
            
            nn.Linear(100, target_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = torch.tensor(x,dtype=torch.float32)
        return self.net(x)


class VanillaModelInversion_WhiteBox(Attacker):
    def __init__(self, top_vfl, args):
        super().__init__(args)
        # 
        self.attack_name = "VanillaModelInversion_WhiteBox"
        self.args = args
        self.top_vfl = top_vfl
        self.vfl_info = top_vfl.final_state
        '''
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

        "aux_loader": [copy.deepcopy(self.parties[ik].aux_loader) for ik in range(self.k)],
        "train_loader": [copy.deepcopy(self.parties[ik].train_loader) for ik in range(self.k)],
        "test_loader": [copy.deepcopy(self.parties[ik].test_loader) for ik in range(self.k)],        
        '''
        # self.vfl_first_epoch = top_vfl.first_epoch_state
        
        # prepare parameters
        self.task_type = args.task_type

        self.device = args.device
        self.num_classes = args.num_classes
        self.label_size = args.num_classes
        self.k = args.k
        self.batch_size = args.batch_size

        # attack configs
        self.party = args.attack_configs['party'] # parties that launch attacks , default 1(active party attack)
        self.lr = args.attack_configs['lr']
        self.epochs = args.attack_configs['epochs']
        self.attack_batch_size = args.attack_configs['batch_size']
        self.attack_sample_num = args.attack_configs['attack_sample_num']
        
        self.criterion = cross_entropy_for_onehot
   
    
    def set_seed(self,seed=0):
        # random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def attack(self):
        self.set_seed(123)
        print_every = 1

        for attacker_ik in self.party: # attacker party #attacker_ik
            assert attacker_ik == (self.k - 1), 'Only Active party launch input inference attack'

            attacked_party_list = [ik for ik in range(self.k)]
            attacked_party_list.remove(attacker_ik)
            index = attacker_ik

            # collect necessary information
            local_model = self.vfl_info['final_model'][0].to(self.device) # Passive
            # global_model = self.vfl_info['final_global_model'].to(self.device)
            # global_model.eval()
            local_model.eval()

            batch_size = self.attack_batch_size

            attack_result = pd.DataFrame(columns = ['Pad_Length','Length','Precision', 'Recall'])

            # attack_test_dataset = self.top_vfl.parties[0].test_dst
            test_data = self.vfl_info["test_data"][0] 
            test_label = self.vfl_info["test_label"][0] 
            
            if len(test_data) > self.attack_sample_num:
                test_data = test_data[:self.attack_sample_num]
                test_label = test_label[:self.attack_sample_num]
                # attack_test_dataset = attack_test_dataset[:self.attack_sample_num]
            
            attack_test_dataset = PassiveDataset_LLM(self.args, test_data, test_label)
            attack_info = f'Attack Sample Num:{len(attack_test_dataset)}'
            print(attack_info)
            append_exp_res(self.args.exp_res_path, attack_info)

            test_data_loader = DataLoader(attack_test_dataset, batch_size=batch_size ,collate_fn=lambda x:x ) # ,
            del(self.vfl_info)
            # test_data_loader = self.vfl_info["test_loader"][0] # Only Passive party has origin input
            
            flag = 0
            enter_time = time.time()
            for origin_input in test_data_loader:
                ### origin_input: tuple of 5
                batch_input_ids = []
                batch_label = []
                batch_attention_mask = []
                batch_token_type_ids = []
                batch_feature = []
                for bs_id in range(len(origin_input)):
                    # Input_ids
                    batch_input_ids.append( origin_input[bs_id][0].tolist() )
                    # Attention Mask
                    batch_attention_mask.append( origin_input[bs_id][2].tolist()  )
                    # token_type_ids
                    if origin_input[bs_id][3] == []:
                        batch_token_type_ids = None
                    else:
                        batch_token_type_ids.append( origin_input[bs_id][3].tolist() ) 
                    # feature (for QuestionAnswering only)
                    if origin_input[bs_id][4] == []:
                        batch_feature = None
                    else:
                        batch_feature.append( origin_input[bs_id][4] )   
                    # Label
                    if type(origin_input[bs_id][1]) != str:
                        batch_label.append( origin_input[bs_id][1].tolist()  )
                    else:
                        batch_label.append( origin_input[bs_id][1]  )

                # # Input_ids
                # batch_input_ids = origin_input[0].tolist() 
                # # Attention Mask
                # batch_attention_mask = origin_input[2].tolist()  
                # # token_type_ids
                # if origin_input[3] == []:
                #     batch_token_type_ids = None
                # else:
                #     batch_token_type_ids = origin_input[3].tolist() 
                # # feature (for QuestionAnswering only)
                # if origin_input[4] == []:
                #     batch_feature = None
                # else:
                #     batch_feature = origin_input[4] 
                # # Label
                # if type(origin_input[1]) != str:
                #     batch_label = [ origin_input[1] ]#.tolist()  
                # else:
                #     batch_label = [ origin_input[1] ]

                batch_input_ids = torch.tensor(batch_input_ids).to(self.device)#.unsqueeze(0)
                batch_attention_mask = torch.tensor(batch_attention_mask).to(self.device)#.unsqueeze(0)
                if batch_token_type_ids != None:
                    batch_token_type_ids = torch.tensor(batch_token_type_ids).to(self.device)#.unsqueeze(0)
                if type( batch_label[0] ) != str:
                    batch_label = torch.tensor(batch_label).to(self.device)
        
                origin_input = [batch_input_ids,batch_label,batch_attention_mask,batch_token_type_ids,batch_feature] 

                # origin_input = [ data, label, mask, token_type_ids, feature(forQA) ]
                origin_data, origin_label, origin_attention_mask, origin_token_type_ids, origin_feature =  origin_input
                if origin_token_type_ids == []:
                    origin_token_type_ids = None
                if origin_feature == []:
                    origin_feature = None

                # real received intermediate result
                input_shape = origin_data.shape[:2]
                self.top_vfl.parties[0].input_shape = input_shape
                self.top_vfl.parties[0].obtain_local_data(origin_data, origin_attention_mask, origin_token_type_ids)
                self.top_vfl.parties[0].gt_one_hot_label = origin_label

                # real_results = self.top_vfl.parties[0].give_pred()
                all_pred_list = self.top_vfl.pred_transmit()
                real_results = all_pred_list[0]

                batch_received_intermediate = real_results[0].type(torch.float32).to(self.device)
                batch_received_attention_mask = real_results[1].to(self.device)
                # print('batch_received_intermediate:',batch_received_intermediate.shape)
                # print('batch_received_attention_mask:',batch_received_attention_mask.shape)
                    
                # each sample in a batch
                for _id in range(origin_data.shape[0]):
                    sample_origin_data = origin_data[_id].unsqueeze(0) # [1,sequence length]
                    received_intermediate = batch_received_intermediate[_id].unsqueeze(0) # [1,256,768]
                    received_attention_mask = batch_received_attention_mask[_id].unsqueeze(0) # [1,256]

                    # initial guess
                    dummy_data = torch.zeros_like(sample_origin_data).long().to(self.device)
                    dummy_attention_mask = received_attention_mask.to(self.device)
                    if origin_token_type_ids != None:
                        dummy_local_batch_token_type_ids = origin_token_type_ids[_id].unsqueeze(0).to(self.device)
                    else:
                        dummy_local_batch_token_type_ids = None

                    bs, seq_length = sample_origin_data.shape
                    if self.args.model_type in ['Bert','Roberta']:
                        dummy_embedding = torch.zeros([bs,seq_length,768]).type(torch.float32).to(self.device)
                    elif self.args.model_type == "GPT2":
                        dummy_embedding = torch.zeros([bs,seq_length,1024]).type(torch.float32).to(self.device)
                    elif self.args.model_type == "Llama":
                        dummy_embedding = torch.zeros([bs,seq_length,4096]).type(torch.float32).to(self.device)
                    else:
                        assert 1>2, f"{self.args.model_type} not supported"
                    dummy_embedding.requires_grad_(True) 
                    
                    optimizer = torch.optim.Adam([dummy_embedding], lr=self.lr)
                    
                    def get_cost(dummy_embedding):
                        # compute dummy result
                        if self.args.model_type  in ['Bert','Roberta']:
                            dummy_intermediate, _a  = local_model(input_ids=dummy_data, \
                                                                    attention_mask = dummy_attention_mask, \
                                                                    token_type_ids=dummy_local_batch_token_type_ids,\
                                                                    embedding_output = dummy_embedding)                 
                        elif self.args.model_type == 'GPT2':
                            dummy_intermediate,  _a, _b, _c = local_model(input_ids=dummy_data, \
                                                        attention_mask = dummy_attention_mask, \
                                                        token_type_ids=dummy_local_batch_token_type_ids,\
                                                        # past_key_values = received_past_key_values,\
                                                        embedding_output = dummy_embedding)    
                        elif self.args.model_type == 'Llama':
                            dummy_intermediate,  _a, _b, _c = local_model(input_ids=None,\
                                                        attention_mask = dummy_attention_mask, \
                                                        # past_key_values = received_past_key_values,\
                                                        inputs_embeds = dummy_embedding)    
                        else:
                            assert 1>2, 'model type not supported'
                    
                        crit = nn.CrossEntropyLoss()
                        _cost = crit(dummy_intermediate, received_intermediate)
                        return _cost
        
                    cost_function = torch.tensor(10000000)
                    _iter = 0
                    eps = 0.1#np.sqrt(np.finfo(float).eps)
                    while _iter<self.epochs: # cost_function.item()>=0.1 and 
                        optimizer.zero_grad()
                        cost_function = get_cost(dummy_embedding)
                        cost_function.backward()

                        dummy_grad = dummy_embedding.grad
                        # print('dummy_grad:',dummy_grad.shape, dummy_grad[:10]) #(256,) np.array
                        
                        optimizer.step()
                        _iter+=1
                        # if _iter%20 == 0:
                        #     print('=== iter ',_iter,'  cost:',cost_function)
                    
                    # recover tokens from dummy embeddings
                    dummy_embedding = dummy_embedding.squeeze()
                    # print('local_model.embeddings.word_embeddings.weight:',local_model.embeddings.word_embeddings.weight.shape)
                    # print('dummy_embedding:',dummy_embedding.shape)

                    predicted_indexs = []
                    for i in range(dummy_embedding.shape[0]):
                        _dum = dummy_embedding[i]
                        # print(_dum.unsqueeze(0).shape)
                        if self.args.model_type  in ['Bert','Roberta']:
                            cos_similarities = nn.functional.cosine_similarity\
                                            (local_model.embeddings.word_embeddings.weight, _dum.unsqueeze(0), dim=1) # .unsqueeze(0)
                        elif self.args.model_type == 'Llama':
                            cos_similarities = nn.functional.cosine_similarity\
                                            (local_model.embed_tokens.weight, _dum.unsqueeze(0), dim=1) # .unsqueeze(0)
                            # print('local_model.embed_tokens.weight:',local_model.embed_tokens.weight.shape)
                            # [32000, 4096] [vocab_size, embed_dim]
                        # print('cos_similarities:',cos_similarities.shape)
                        _, predicted_index = cos_similarities.max(0)
                        predicted_index = predicted_index.item()
                        predicted_indexs.append(predicted_index)
                    
                    sample_origin_id = sample_origin_data.squeeze().tolist()
                    origin_text = self.args.tokenizer.decode(sample_origin_id)

                    clean_sample_origin_id = sample_origin_id.copy()
                    while self.args.tokenizer.pad_token_id in clean_sample_origin_id:
                        clean_sample_origin_id.remove(self.args.tokenizer.pad_token_id) # with no pad
                    
                    suc_cnt = 0
                    for _sample_id in clean_sample_origin_id:
                        if _sample_id in predicted_indexs:
                            suc_cnt+=1
                    recall = suc_cnt / len(clean_sample_origin_id)

                    suc_cnt = 0
                    for _pred_id in predicted_indexs:
                        if _pred_id in clean_sample_origin_id:
                            suc_cnt+=1
                    precision = suc_cnt / len(predicted_indexs)


                    attack_result.loc[len(attack_result)] = [len(sample_origin_id), len(clean_sample_origin_id), precision,recall ]

                    
                    pred_text = self.args.tokenizer.decode(predicted_indexs)

                    if flag == 0:
                        print('len:',len(clean_sample_origin_id),'  precision:',precision, ' recall:',recall)
                        print('origin_text:\n',origin_text)
                        print('-'*25)
                        print('pred_text:\n',pred_text)
                        print('-'*25)
                        # append_exp_res(self.args.exp_res_path, origin_text)
                        # append_exp_res(self.args.exp_res_path, pred_text)
                    flag += 1
                
                    del(dummy_embedding)
                    del(dummy_data)
                    del(dummy_attention_mask)

                del(batch_received_intermediate)
                del(batch_received_attention_mask)

            end_time = time.time()
        
        attack_total_time = end_time - enter_time
        Precision = attack_result['Precision'].mean()
        Recall = attack_result['Recall'].mean()

        # model_name = self.args.model_list['0']['type']
        # if self.args.pretrained == 1:
        #     result_path = f'./exp_result/{str(self.args.dataset)}/{self.attack_name}/{self.args.defense_name}_{self.args.defense_param}_pretrained_{str(model_name)}/'
        # else:
        #     result_path = f'./exp_result/{str(self.args.dataset)}/{self.attack_name}/{self.args.defense_name}_{self.args.defense_param}_finetuned_{str(model_name)}/'

        # if not os.path.exists(result_path):
        #     os.makedirs(result_path)
        # result_file_name = result_path + f'{self.args.pad_info}_{str(Precision)}_{str(Recall)}.csv'
        # print(result_file_name)
        # attack_result.to_csv(result_file_name)

        return Precision, Recall, attack_total_time