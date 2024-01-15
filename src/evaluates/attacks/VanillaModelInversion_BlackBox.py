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


class VanillaModelInversion_BlackBox(Attacker):
    def __init__(self, top_vfl, args):
        super().__init__(args)
        # 
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

        self.device = 'cpu' #args.device
        self.num_classes = args.num_classes
        self.label_size = args.num_classes
        self.k = args.k
        self.batch_size = args.batch_size

        # attack configs
        self.party = args.attack_configs['party'] # parties that launch attacks , default 1(active party attack)
        self.lr = args.attack_configs['lr']
        self.epochs = args.attack_configs['epochs']
        self.attack_batch_size = args.attack_configs['batch_size']
        
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
            global_model = self.vfl_info['final_global_model'].to(self.device)
            global_model.eval()
            local_model.eval()

            batch_size = self.attack_batch_size
            
            # train_data = self.vfl_info["train_data"][attacker_ik]
            # train_label = self.vfl_info["train_label"][attacker_ik]
            # train_dst = PassiveDataset_LLM(self.args, train_data, train_label)
            # train_data_loader = DataLoader(train_dst, batch_size=1)

            # train_data_loader = self.vfl_info["train_loader"][0] # Only Passive party has origin input
            test_data_loader = self.vfl_info["test_loader"][0] # Only Passive party has origin input
            
            attack_result = pd.DataFrame(columns = ['Length','RR'])

            for origin_input in test_data_loader:
                # origin_input = [ data, label, mask, token_type_ids, feature(forQA) ]
                origin_data, origin_label, origin_attention_mask, origin_token_type_ids, origin_feature =  origin_input
                if origin_token_type_ids == []:
                    origin_token_type_ids = None
                if origin_feature == []:
                    origin_feature = None
                
                # real result
                input_shape = origin_data.shape[:2]
                self.top_vfl.parties[0].input_shape = input_shape
                self.top_vfl.parties[0].obtain_local_data(origin_data)
                self.top_vfl.parties[0].gt_one_hot_label = origin_label
                self.top_vfl.parties[0].local_batch_attention_mask = origin_attention_mask
                self.top_vfl.parties[0].local_batch_token_type_ids = origin_token_type_ids
                real_results = self.top_vfl.parties[0].give_pred()

                batch_received_intermediate = real_results[0].type(torch.float32).to(self.device)
                batch_received_attention_mask = real_results[2].to(self.device)

                def numerical_gradient(func, x, h=1):
                    '''
                    x : torch.size(x_length)
                    '''
                    assert len(x.shape)==1

                    grad = torch.zeros_like(x) # to store grads
                    for i in range(x.shape[0]):
                        tmp_val = x[i]
                        
                        x[i] = tmp_val + h
                        # print('x1:',x)
                        f1 = func(x)

                        x[i] = tmp_val - h
                        # print('x2:',x)
                        f2 = func(x)

                        x[i] = tmp_val

                        # if i == 0:
                        #     print('pos:',i,'==f1:',f1,'  f2:',f2)
                        _grad = (f1-f2)/(2*h)
                        grad[i] = _grad
                    
                    return grad

                # each sample in a batch
                for _id in range(origin_data.shape[0]):
                    sample_origin_data = origin_data[_id].unsqueeze(0) # [1,sequence length]
                    # print('sample_origin_data:',sample_origin_data.shape)

                    received_intermediate = batch_received_intermediate[_id].unsqueeze(0) # [1,256,768]
                    received_attention_mask = batch_received_attention_mask[_id].unsqueeze(0) # [1,256]
                    # print('received_intermediate:',received_intermediate.shape)
                    # print('received_attention_mask:',received_attention_mask.shape)

                    # initial guess
                    dummy_data = torch.zeros_like(sample_origin_data).long().to(self.device)
                    dummy_attention_mask = received_attention_mask.to(self.device)
                    dummy_local_batch_token_type_ids = origin_token_type_ids[_id].unsqueeze(0).to(self.device)

                    bs, seq_length = sample_origin_data.shape
                    dummy_embedding = torch.zeros([bs,seq_length,768]).type(torch.float32).to(self.device)
                    dummy_embedding.requires_grad_(True) 
                    # optimizer = torch.optim.Adam([dummy_embedding], lr=self.lr)
                    
                    def get_cost(dummy_embedding):
                        # compute dummy result
                        # if self.args.model_type == 'Bert':
                        dummy_intermediate, _  = local_model(input_ids=dummy_data, attention_mask = dummy_attention_mask, token_type_ids=dummy_local_batch_token_type_ids,\
                        embedding_output = dummy_embedding)                 
                        
                        # def EuclideanDistances(a,b):
                        #     a = a.reshape(-1) 
                        #     b = b.reshape(-1) 
                        #     return F.pairwise_distance(a, b, p=2)#.item()
                        # _cost = EuclideanDistances(received_intermediate, dummy_intermediate)
                        crit = nn.CrossEntropyLoss()
                        _cost = crit(dummy_intermediate, received_intermediate)
                        return _cost
        
                    cost_function = torch.tensor(10000000)
                    _iter = 0
                    eps = np.sqrt(np.finfo(float).eps)
                    while _iter<self.epochs: # cost_function.item()>=0.1 and 
                        cost_function = get_cost(dummy_embedding)

                        dummy_grad = numerical_gradient(get_cost, dummy_embedding, eps)
                        # dummy_grad = torch.tensor( optimize.approx_fprime(dummy_data, get_cost, eps) )
                        print('dummy_grad:',dummy_grad.shape) #(256,) np.array
                        
                        with torch.no_grad():
                            dummy_data = (dummy_data - self.lr * dummy_grad)

                        # print('dummy_data:',dummy_data.shape,dummy_data[:5]) # torch.size[256]

                        _iter+=1
                        if _iter%1 == 0:
                            print('=== iter ',_iter,'  cost:',cost_function)
                    
                    ####### recover tokens from dummy embeddings
                    dummy_embedding = dummy_embedding.squeeze()
                    # print('local_model.embeddings.word_embeddings.weight:',local_model.embeddings.word_embeddings.weight.shape)
                    # print('dummy_embedding:',dummy_embedding.shape)

                    predicted_indexs = []
                    for i in range(dummy_embedding.shape[0]):
                        _dum = dummy_embedding[i]
                        # print(_dum.unsqueeze(0).shape)
                        cos_similarities = nn.functional.cosine_similarity\
                                        (local_model.embeddings.word_embeddings.weight, _dum.unsqueeze(0), dim=1) # .unsqueeze(0)
                        # print('cos_similarities:',cos_similarities.shape)
                        _, predicted_index = cos_similarities.max(0)
                        predicted_index = predicted_index.item()
                        predicted_indexs.append(predicted_index)
                    
                    suc_cnt = 0
                    sample_origin_id = sample_origin_data.squeeze().tolist()
                    # print('sample_origin_id:',type(sample_origin_id), sample_origin_id) # list
                    # print('predicted_indexs:',predicted_indexs)

                    for _sample_id in sample_origin_id:
                        if _sample_id in predicted_indexs:
                            suc_cnt+=1
                    
                    recover_rate = suc_cnt / len(predicted_indexs)
                    print('len:',len(sample_origin_id),'  recover_rate:',recover_rate)

                    attack_result.loc[len(attack_result)] = [len(sample_origin_id), recover_rate]

                    origin_text = self.args.tokenizer.decode(sample_origin_id)
                    pred_text = self.args.tokenizer.decode(predicted_indexs)
                    print('origin_text:',origin_text)
                    print('pred_text:',pred_text)

        result_file_name = './attack_result_'+str(self.args.dataset)+'.csv'
        attack_result.to_csv(result_file_name)
        Recover_Rate = attack_result['RR'].mean()
        return Recover_Rate
            # attack_result.to_csv('../../exp_result/attack_result_cola.csv')
            # assert 1>2


       
            
        #     criterion = nn.MSELoss()

        #     if self.args.dataset == 'cifar10' or self.args.dataset == 'cifar100':
        #         decoder_list = [custom_AE(self.args.model_list[str(ik)]['output_dim'], 3*self.args.model_list[str(ik)]['input_dim']).to(self.device) for ik in attacked_party_list]
        #     else: # mnist
        #         decoder_list = [custom_AE(self.args.model_list[str(ik)]['output_dim'], self.args.model_list[str(ik)]['input_dim']).to(self.device) for ik in attacked_party_list]

        #     #custom_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim, output_dim=32).to(self.device)
        #     optimizer_list = [torch.optim.Adam(decoder.parameters(), lr=self.lr) for decoder in decoder_list]

        #     feature_dimention_list = [self.args.model_list[str(ik)]['input_dim'] for ik in attacked_party_list]

        #     print('========= Feature Inference Training ========')
        #     for i_epoch in range(self.epochs):
        #         ####### Train Generator for each attacked party #######
        #         decoder_list = [decoder.train() for decoder in decoder_list]
        #         for parties_data in zip(*aux_loader_list):
        #             self.gt_one_hot_label = label_to_one_hot(parties_data[self.k-1][1], self.num_classes)
        #             self.gt_one_hot_label = self.gt_one_hot_label.to(self.device)
        #             self.parties_data = parties_data
        #             batch_data_b = [parties_data[ik][0] for ik in range(len(parties_data)-1)] # Passive Party data  
        #             batch_data_a = parties_data[-1][0] # Active Party data   

        #             # target img
        #             img = batch_data_b
                    
        #             # Known Information : intermediate representation
        #             with torch.no_grad():
        #                 # ir = net_b(batch_data_b)
        #                 # print('batch_data_b[ik]:',batch_data_b[0].size())

        #                 ir = [net_b[ik](batch_data_b[ik]) for ik in range(len(parties_data)-1)]
        #                 ####### DP Defense On FR ########
        #                 if self.args.apply_dp == True:
        #                     if 'laplace' in self.args.defense_name.casefold():
        #                         ir = [LaplaceDP_for_pred(self.args, [ir[ik]]) for ik in range(len(ir))]
        #                         # ir = LaplaceDP_for_pred(self.args, ir)
        #                     elif 'gaussian' in self.args.defense_name.casefold():
        #                         ir = [GaussianDP_for_pred(self.args, [ir[ik]]) for ik in range(len(ir))]
        #                         # ir = GaussianDP_for_pred(self.args, ir)
        #                 ####### DP Defense On FR ########

        #             output = []
        #             for ik in range(len(batch_data_b)): # should have k-1 parties, except the attacker
        #                 img[ik], ir[ik] = img[ik].type(torch.FloatTensor), ir[ik].type(torch.FloatTensor)
        #                 img[ik], ir[ik] = Variable(img[ik]).to(self.device), Variable(ir[ik]).to(self.device)
                        
        #                 # recovered image
        #                 output.append(decoder_list[ik](ir[ik])) # torch.Size([10])
        #                 # print('ir:',ir[ik].size()) # [32,  10]
        #                 # print('img:',img[ik].size()) # [32,  3,16,32]
        #                 # print('output:',output[ik].size()) # [32,512]

        #                 img[ik] = img[ik].reshape(output[ik].size())

        #                 train_loss = criterion(output[ik], img[ik])

        #                 optimizer_list[ik].zero_grad()
        #                 train_loss.backward()
        #                 optimizer_list[ik].step()

        #         ####### Test Performance of Generator #######
        #         if (i_epoch + 1) % print_every == 0:
        #             mse_list = []
        #             rand_mse_list = []
        #             decoder_list = [decoder.eval() for decoder in decoder_list]
        #             with torch.no_grad():
        #                 # test_data_a = self.vfl_first_epoch['data'][1][0] # active party 
        #                 # test_data_b = self.vfl_first_epoch['data'][0][0] # passive party 
        #                 # # pred with possible defense 
        #                 # test_pred_a = self.vfl_first_epoch['predict'][1]
        #                 # test_pred_b = self.vfl_first_epoch['predict'][0]
        #                 # test_global_pred = self.vfl_first_epoch['global_pred'].to(self.device)

        #                 img = test_data_b # target img
        #                 # test_pred_b = net_b(test_data_b)

        #                 if self.args.dataset == 'cifar10' or self.args.dataset == 'cifar100':
        #                     test_data_b[ik] = test_data_b[ik].reshape([len(test_data_b[ik]),3,16,32])
                        
        #                 test_pred_b = [net_b[ik](test_data_b[ik]) for ik in range(len(test_data_b))]
        #                 ir = test_pred_b 
        #                 ####### DP Defense On FR ########
        #                 if self.args.apply_dp == True:
        #                     if 'laplace' in self.args.defense_name.casefold():
        #                         ir = [LaplaceDP_for_pred(self.args, [ir[ik]]) for ik in range(len(ir))]
        #                         # ir = LaplaceDP_for_pred(self.args, ir)
        #                     elif 'gaussian' in self.args.defense_name.casefold():
        #                         ir = [GaussianDP_for_pred(self.args, [ir[ik]]) for ik in range(len(ir))]
        #                         # ir = GaussianDP_for_pred(self.args, ir)
        #                 ####### DP Defense On FR ########
                       
        #                 output = []
        #                 for ik in range(len(test_data_b)): # should have k-1 parties, except the attacker
        #                     img[ik], ir[ik] = img[ik].type(torch.FloatTensor), ir[ik].type(torch.FloatTensor)
        #                     img[ik], ir[ik] = Variable(img[ik]).to(self.device), Variable(ir[ik]).to(self.device)
                            
        #                     output.append(decoder_list[ik](ir[ik])) # reconstruction result

        #                     img[ik] = img[ik].reshape(output[ik].size())
        #                     rand_img = torch.randn(img[ik].size()).to(self.device)
        #                     _mse = criterion(output[ik], img[ik])
        #                     _rand_mse = criterion(rand_img, img[ik])
        #                     mse_list.append(_mse)
        #                     rand_mse_list.append(_rand_mse)
        #                 mse = torch.sum(torch.tensor(mse_list) * torch.tensor(feature_dimention_list))/torch.sum(torch.tensor(feature_dimention_list))
        #                 rand_mse = torch.sum(torch.tensor(rand_mse_list) * torch.tensor(feature_dimention_list))/torch.sum(torch.tensor(feature_dimention_list))
                    
        #             print('Epoch {}% \t train_loss:{:.2f} mse:{:.4}, mse_reduction:{:.2f}'.format(
        #                 i_epoch, train_loss.item(), mse, rand_mse-mse))
            
        #     ####### Clean ######
        #     for decoder_ in decoder_list:
        #         del(decoder_)
        #     del(aux_dst_a)
        #     del(aux_loader_a)
        #     del(aux_dst_b)
        #     del(aux_loader_b)
        #     del(aux_loader_list)
        #     del(test_data_b)
        #     del(test_data_a)

        #     print(f"ResSFL, if self.args.apply_defense={self.args.apply_defense}")
        #     print(f'batch_size=%d,class_num=%d,attacker_party_index=%d,mse=%lf' % (self.batch_size, self.label_size, index, mse))

        # print("returning from ResSFL")
        # return rand_mse,mse