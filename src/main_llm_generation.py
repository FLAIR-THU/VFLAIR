import os
import sys
import numpy as np
import time

import random
import logging
import argparse
import torch
torch.cuda.current_device()

from load.LoadConfigs import * #load_configs
from load.LoadParty import load_parties, load_parties_llm
from evaluates.MainTaskVFL_LLM import *
from utils.basic_functions import append_exp_res

from load.LoadConfigs import INVERSION

from models.llm_models.generation_model import *

import warnings
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel,AutoModelForSequenceClassification,AutoModelForPreTraining

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def evaluate_no_attack_pretrained(args):
    # No Attack
    set_seed(args.current_seed)

    vfl = MainTaskVFL_LLM(args)
    vfl.init_communication()
    exp_result, metric_val = vfl.inference()

    # attack_metric = main_acc_noattack - main_acc
    # attack_metric_name = 'acc_loss'

    # # Save record 
    append_exp_res(args.exp_res_path, exp_result)
    
    return vfl, metric_val



if __name__ == '__main__':
    parser = argparse.ArgumentParser("backdoor")
    parser.add_argument('--device', type=str, default='cuda', help='use gpu or cpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--seed', type=int, default=97, help='random seed')
    parser.add_argument('--configs', type=str, default='basic_configs_news', help='configure json file path')
    parser.add_argument('--save_model', type=bool, default=False, help='whether to save the trained model')
    args = parser.parse_args()

    # for seed in range(97,102): # test 5 times 
    # for seed in [97]:
    for seed in [60]: # test 5 times 
        args.current_seed = seed
        set_seed(seed)
        print('================= iter seed ',seed,' =================')
        
        args = load_basic_configs(args.configs, args)
        args.need_auxiliary = 0 # no auxiliary dataset for attackerB

        if args.device == 'cuda':
            cuda_id = args.gpu
            torch.cuda.set_device(cuda_id)
            print(f'running on cuda{torch.cuda.current_device()}')
        else:
            print('running on cpu')

        
        ####### load configs from *.json files #######
        ############ Basic Configs ############
        assert args.dataset_split != None, "dataset_split attribute not found config json file"
        assert 'dataset_name' in args.dataset_split, 'dataset not specified, please add the name of the dataset in config json file'
        args.dataset = args.dataset_split['dataset_name']
        print('======= Defense ========')
        print('Defense_Name:',args.defense_name)
        print('Defense_Config:',str(args.defense_configs))
        print('===== Total Attack Tested:',args.attack_num,' ======')
        print('inversion:',args.inversion_list,args.inversion_index)

        # Save record for different defense method
        args.exp_res_dir = f'exp_result/{args.dataset}/Q{str(args.Q)}/'
        if not os.path.exists(args.exp_res_dir):
            os.makedirs(args.exp_res_dir)
        model_name = args.model_list[str(0)]["type"] #.replace('/','-')
        if args.pipeline=='pretrained':
            filename = f'{args.defense_name}_{args.defense_param},pretrained_model={args.model_list[str(0)]["type"]}.txt'
        else:
            filename = f'{args.defense_name}_{args.defense_param},finetuned_model={args.model_list[str(0)]["type"]}.txt'
        args.exp_res_path = args.exp_res_dir + filename
        print(args.exp_res_path)
        print('=================================\n')

        iterinfo='===== iter '+str(seed)+' ===='
        append_exp_res(args.exp_res_path, iterinfo)

        args.basic_vfl_withaux = None
        args.main_acc_noattack_withaux = None
        args.basic_vfl = None
        args.main_acc_noattack = None

        args = load_attack_configs(args.configs, args, -1)

        args = load_parties_llm(args, need_data = False)

        set_seed(args.current_seed)

        vfl = MainTaskVFL_LLM(args)

        if args.model_type == 'GPT2':
            GenerationModel = GPT2_VFLGeneration(vfl) #GPT2_VFLGeneration(vfl)
        elif args.model_type == 'Llama':
            GenerationModel = Llama_VFLGeneration(vfl) #GPT2_VFLGeneration(vfl)
        else:
            assert 1>2, 'model type not supported for generation'
            
        ######### define your input text here #########
        input_text = ["Hello, how are you these days?"]
        ######### define your input text here #########

        inputs = args.tokenizer(input_text, return_tensors="pt").to(args.device)
        ids = args.tokenizer(input_text, return_tensors='pt') 
        input_ids = torch.tensor(ids['input_ids']).squeeze().to(args.device)
        attention_mask = torch.tensor(ids['attention_mask']).squeeze().to(args.device)
        if 'token_type_ids' in list(ids.keys()):
            token_type_ids = torch.tensor(ids['token_type_ids']).squeeze().to(args.device)
        else:
            token_type_ids = None
        

        
        ##### normal full model #####
        print('full model path:',args.model_path)
        current_model_type = args.model_type #"gpt2"
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        full_model = AutoModelForCausalLM.from_pretrained(args.model_path) # AutoModelForCausalLM
        full_model = full_model.to(args.device)
        greedy_output = full_model.generate(**inputs, max_length=30)
        print("greedy_output:\n" + 100 * '-')
        print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
        print('\n\n')


        ######### VFL model ########
        greedy_output = GenerationModel.generate(**inputs, max_length=30)
        print("greedy_output:\n" + 100 * '-')
        print(args.tokenizer.decode(greedy_output[0], skip_special_tokens=True))






