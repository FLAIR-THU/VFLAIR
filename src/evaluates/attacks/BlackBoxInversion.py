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

from evaluates.defenses.defense_functions import LaplaceDP_for_pred, GaussianDP_for_pred


class Gamma_Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(Net, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)
        # self.linear = nn.Linear(hidden_size, output_size=1)

    def forward(self, x, hidden_prev):
        out, hidden_prev = self.rnn(x, hidden_prev)
        #    # [b, seq, h]
        #    out = out.view(-1, hidden_size)
        #    out = self.linear(out)#[seq,h] => [seq,3]
        #    out = out.unsqueeze(dim=0)  # => [1,seq,3]
        return out, hidden_prev


class BlackBoxInversion(Attacker):
    def __init__(self, top_vfl, args):
        super().__init__(args)
        # 
        self.attack_name = "BlackBoxInversion"
        self.args = args
        self.top_vfl = top_vfl
        self.vfl_info = top_vfl.final_state

        # prepare parameters
        self.task_type = args.task_type

        self.device = args.device
        self.num_classes = args.num_classes
        self.label_size = args.num_classes
        self.k = args.k
        self.batch_size = args.batch_size

        # attack configs
        self.party = args.attack_configs['party']  # parties that launch attacks , default 1(active party attack)
        self.lr = args.attack_configs['lr']
        self.epochs = args.attack_configs['epochs']
        self.attack_batch_size = args.attack_configs['batch_size']
        self.T = args.attack_configs['T']

        self.criterion = cross_entropy_for_onehot

    def set_seed(self, seed=0):
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

        for attacker_ik in self.party:  # attacker party #attacker_ik
            assert attacker_ik == (self.k - 1), 'Only Active party launch input inference attack'

            attacked_party_list = [ik for ik in range(self.k)]
            attacked_party_list.remove(attacker_ik)
            index = attacker_ik

            # collect necessary information
            local_model = self.vfl_info['final_model'][0].to(self.device)  # Passive
            global_model = self.vfl_info['final_global_model'].to(self.device)
            global_model.eval()
            local_model.eval()

            batch_size = self.attack_batch_size

            attack_result = pd.DataFrame(columns=['Length', 'RR'])

            embedding_matrix = local_model.embeddings.word_embeddings.weight  # 30522, 768
            vocab_size, embed_dim = embedding_matrix.shape
            # print('embedding_matrix:',embedding_matrix.shape)

            Gamma = Gamma_Net(input_size=embed_dim, hidden_size=vocab_size, num_layers=1).to(self.device)

            aux_data_loader = self.vfl_info["test_loader"][0]  # Only Passive party has origin input
            for origin_input in aux_data_loader:
                batch_input_ids = []
                batch_label = []
                batch_attention_mask = []
                batch_token_type_ids = []
                batch_feature = []
                for bs_id in range(len(origin_input)):
                    # Input_ids
                    batch_input_ids.append(origin_input[bs_id][0].tolist())
                    # Attention Mask
                    batch_attention_mask.append(origin_input[bs_id][2].tolist())
                    # token_type_ids
                    if origin_input[bs_id][3] == []:
                        batch_token_type_ids = None
                    else:
                        batch_token_type_ids.append(origin_input[bs_id][3].tolist())
                        # feature (for QuestionAnswering only)
                    if origin_input[bs_id][4] == []:
                        batch_feature = None
                    else:
                        batch_feature.append(origin_input[bs_id][4])
                        # Label
                    if type(origin_input[bs_id][1]) != str:
                        batch_label.append(origin_input[bs_id][1].tolist())
                    else:
                        batch_label.append(origin_input[bs_id][1])

                batch_input_ids = torch.tensor(batch_input_ids).to(self.device)
                batch_attention_mask = torch.tensor(batch_attention_mask).to(self.device)
                if batch_token_type_ids != None:
                    batch_token_type_ids = torch.tensor(batch_token_type_ids).to(self.device)
                if type(batch_label[0]) != str:
                    batch_label = torch.tensor(batch_label).to(self.device)

                origin_input = [batch_input_ids, batch_label, batch_attention_mask, batch_token_type_ids, batch_feature]

                # origin_input = [ data, label, mask, token_type_ids, feature(forQA) ]
                origin_data, origin_label, origin_attention_mask, origin_token_type_ids, origin_feature = origin_input
                if origin_token_type_ids == []:
                    origin_token_type_ids = None
                if origin_feature == []:
                    origin_feature = None

                # Origin Data
                input_shape = origin_data.shape[:2]
                bs, seq_length = input_shape  # 10,30
                self.top_vfl.parties[0].input_shape = input_shape
                self.top_vfl.parties[0].obtain_local_data(origin_data.to(self.args.device), \
                                                          origin_attention_mask.to(self.args.device),
                                                          origin_token_type_ids.to(self.args.device))

                self.top_vfl.parties[0].gt_one_hot_label = origin_label

                # Real Intermediate Results
                real_results = self.top_vfl.parties[0].give_pred()
                batch_received_intermediate = real_results[0].type(torch.float32).to(self.device)  # real intermediate
                batch_received_attention_mask = real_results[2].to(self.device)

                embedding_shape = batch_received_intermediate.shape
                batch_size, seq_length, embed_dim = embedding_shape  # bs , seq_length, embed_dim 768
                vocab_size = local_model.config.vocab_size  # 30522

                def MLP_loss(origin_data, real_intermediate):
                    print('origin_data:', origin_data.shape)
                    print('real_intermediate:', real_intermediate.shape)

                    W_pred = torch.zeros(num_layers, batch_size, vocab_size)  # 输入数据
                    L = 0

                    input_data = origin_data
                    for i in range(seq_length):
                        out, W_pred = Gamma(input_data, W_pred)

                        print("out.shape:", out.shape)  # bs, seq_len, vocab_size
                        print("W_pred.shape:", W_pred.shape)  # num_layers, bs, vocab_size

                        predicted_w = torch.argmax(out, dim=-1)  # bs , seq_len

                    # compute dummy result
                    if self.args.model_type == 'Bert':
                        dummy_intermediate, _ = local_model(input_ids=dummy_data, attention_mask=dummy_attention_mask, \
                                                            token_type_ids=dummy_local_batch_token_type_ids,
                                                            embedding_output=dummy_embedding)
                    elif self.args.model_type == 'GPT2':
                        if self.args.task_type == 'SequenceClassification':
                            dummy_intermediate, _, _ = local_model(input_ids=dummy_data,
                                                                   attention_mask=dummy_attention_mask, \
                                                                   token_type_ids=dummy_local_batch_token_type_ids,
                                                                   embedding_output=dummy_embedding)
                        elif self.args.task_type == 'CausalLM':
                            dummy_intermediate, _, __ = local_model(input_ids=dummy_data,
                                                                    attention_mask=dummy_attention_mask, \
                                                                    token_type_ids=dummy_local_batch_token_type_ids,
                                                                    embedding_output=dummy_embedding)
                        elif self.args.task_type == 'QuestionAnswering':
                            dummy_intermediate, _, __ = local_model(input_ids=dummy_data,
                                                                    attention_mask=dummy_attention_mask, \
                                                                    token_type_ids=dummy_local_batch_token_type_ids,
                                                                    embedding_output=dummy_embedding)
                    else:
                        assert 1 > 2, 'model type not supported'

                    # Defense
                    dummy_intermediate = self.top_vfl.apply_defense_on_transmission(dummy_intermediate)
                    # Communication Process
                    dummy_intermediate = self.top_vfl.apply_communication_protocol_on_transmission(dummy_intermediate)

                    crit = nn.CrossEntropyLoss()
                    _cost = crit(dummy_intermediate, received_intermediate)
                    return _cost

                # each sample in a batch
                for _id in range(origin_data.shape[0]):
                    ###### Real data #####
                    sample_origin_data = origin_data[_id].unsqueeze(0)  # [1,sequence length]
                    # print('sample_origin_data:',sample_origin_data.shape)
                    received_intermediate = batch_received_intermediate[_id].unsqueeze(0)  # [1,256,768]
                    # print('received_intermediate:',received_intermediate.shape)
                    received_attention_mask = batch_received_attention_mask[_id].unsqueeze(0)  # [1,256]
                    # print('received_attention_mask:',received_attention_mask.shape)

                    ##### Dummy Data #####
                    dummy_data = torch.zeros_like(sample_origin_data).long().to(self.device)
                    dummy_attention_mask = received_attention_mask.to(self.device)
                    dummy_local_batch_token_type_ids = origin_token_type_ids[_id].unsqueeze(0).to(self.device)
                    _, seq_length = sample_origin_data.shape

                    # initial guess

                    optimizer = torch.optim.Adam(Gamma.parameters(), lr=self.lr)

                    def MLP_loss(x, received_intermediate):
                        soft_z = nn.functional.softmax(Z / self.T, dim=-1)  # 30(seq_length), 30522(vocab_size)
                        relaxed_Z = torch.mm(soft_z, embedding_matrix).unsqueeze(0)  # 1, seq_length, 768(embed_dim)
                        dummy_embedding = relaxed_Z

                        # compute dummy result
                        if self.args.model_type == 'Bert':
                            dummy_intermediate, _ = local_model(input_ids=dummy_data,
                                                                attention_mask=dummy_attention_mask, \
                                                                token_type_ids=dummy_local_batch_token_type_ids,
                                                                embedding_output=dummy_embedding)
                        elif self.args.model_type == 'GPT2':
                            if self.args.task_type == 'SequenceClassification':
                                dummy_intermediate, _, _ = local_model(input_ids=dummy_data,
                                                                       attention_mask=dummy_attention_mask, \
                                                                       token_type_ids=dummy_local_batch_token_type_ids,
                                                                       embedding_output=dummy_embedding)
                            elif self.args.task_type == 'CausalLM':
                                dummy_intermediate, _, __ = local_model(input_ids=dummy_data,
                                                                        attention_mask=dummy_attention_mask, \
                                                                        token_type_ids=dummy_local_batch_token_type_ids,
                                                                        embedding_output=dummy_embedding)
                            elif self.args.task_type == 'QuestionAnswering':
                                dummy_intermediate, _, __ = local_model(input_ids=dummy_data,
                                                                        attention_mask=dummy_attention_mask, \
                                                                        token_type_ids=dummy_local_batch_token_type_ids,
                                                                        embedding_output=dummy_embedding)
                        else:
                            assert 1 > 2, 'model type not supported'

                        # Defense
                        dummy_intermediate = self.top_vfl.apply_defense_on_transmission(dummy_intermediate)
                        # Communication Process
                        dummy_intermediate = self.top_vfl.apply_communication_protocol_on_transmission(
                            dummy_intermediate)

                        crit = nn.CrossEntropyLoss()
                        _cost = crit(dummy_intermediate, received_intermediate)
                        return _cost

                    cost_function = torch.tensor(10000000)
                    _iter = 0
                    eps = np.sqrt(np.finfo(float).eps)
                    while _iter < self.epochs:  # cost_function.item()>=0.1 and

                        optimizer.zero_grad()
                        cost_function = get_cost(Z, received_intermediate)
                        cost_function.backward()

                        z_grad = Z.grad  # 30(seq_length), 768(embed_dim)

                        optimizer.step()

                        _iter += 1
                        # if _iter%1 == 0:
                        #     print('=== iter ',_iter,'  cost:',cost_function)

                    ####### recover tokens from Z: [seq_length, vocab_size]

                    # print('local_model.embeddings.word_embeddings.weight:',local_model.embeddings.word_embeddings.weight.shape)
                    # print('dummy_embedding:',dummy_embedding.shape)

                    predicted_indexs = torch.argmax(Z, dim=-1)  # torch.size[seq_length]
                    predicted_indexs = predicted_indexs.tolist()

                    suc_cnt = 0
                    sample_origin_id = sample_origin_data.squeeze().tolist()
                    # print('sample_origin_id:',type(sample_origin_id), sample_origin_id) # list
                    # print('predicted_indexs:',predicted_indexs)

                    for _sample_id in sample_origin_id:
                        if _sample_id in predicted_indexs:
                            suc_cnt += 1

                    recover_rate = suc_cnt / len(predicted_indexs)
                    # print('len:',len(sample_origin_id),'  recover_rate:',recover_rate)

                    attack_result.loc[len(attack_result)] = [len(sample_origin_id), recover_rate]

                    origin_text = self.args.tokenizer.decode(sample_origin_id)
                    pred_text = self.args.tokenizer.decode(predicted_indexs)
                    # print('origin_text:',origin_text)
                    # print('pred_text:',pred_text)
        Recover_Rate = attack_result['RR'].mean()

        model_name = self.args.model_list['0']['type']
        if self.args.pretrained == 1:
            result_path = f'./exp_result/{str(self.args.dataset)}/{self.attack_name}/{self.args.defense_name}_{self.args.defense_param}_pretrained_{str(model_name)}/'
        else:
            result_path = f'./exp_result/{str(self.args.dataset)}/{self.attack_name}/{self.args.defense_name}_{self.args.defense_param}_finetuned_{str(model_name)}/'

        if not os.path.exists(result_path):
            os.makedirs(result_path)
        result_file_name = result_path + f'{self.args.pad_info}_rr_{str(Recover_Rate)}.csv'
        print(result_file_name)

        attack_result.to_csv(result_file_name)

        return Recover_Rate
