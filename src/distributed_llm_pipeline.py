import os
import sys
import numpy as np
import time

import random
import logging
import argparse
import torch

from load.LoadConfigs import * #load_configs
from framework.ml.LoadParty import load_parties_llm
from evaluates.MainTask_LLM import *
from utils.basic_functions import append_exp_res
import warnings
warnings.filterwarnings("ignore")

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
    #set_seed(args.current_seed)

    vfl = MainTask_LLM(args)
    exp_result, metric_val = vfl.inference()

    # attack_metric = main_acc_noattack - main_acc
    # attack_metric_name = 'acc_loss'

    # # Save record 
    append_exp_res(args.exp_res_path, exp_result)

    return vfl, metric_val

def evaluate_no_attack_finetune(args):
    # No Attack
    set_seed(args.current_seed)

    vfl = MainTask_LLM(args)
    exp_result, metric_val= vfl.train()

    # attack_metric = main_acc_noattack - main_acc
    # attack_metric_name = 'acc_loss'

    # # Save record 
    exp_result = f"K_{args.k}|bs_{args.batch_size}|LR_{args.main_lr}|num_class_{args.num_classes}|Q_{args.Q}|epoch_{args.main_epochs}| " \
        + exp_result
    print(exp_result)
    append_exp_res(args.exp_res_path, exp_result)
    # append_exp_res(args.exp_res_path, f"==stopping_iter:{stopping_iter}==stopping_time:{stopping_time}==stopping_commu_cost:{stopping_commu_cost}")

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
        print('================= iter seed ',seed,' =================')

        args = load_llm_configs(args.configs, args)

        mode = args.apply_trainable_layer
        args.dataset = args.dataset_split['dataset_name']

        args.exp_res_dir = f'exp_result/{args.dataset}_1best/Q{str(args.Q)}/{str(mode)}/'
        if not os.path.exists(args.exp_res_dir):
            os.makedirs(args.exp_res_dir)
        model_name = args.model_list[str(0)]["type"]  # .replace('/','-')
        if args.pretrained == 1:
            filename = f'pretrained_model={model_name}.txt'
        else:
            filename = f'finetuned_model={model_name}.txt'
        args.exp_res_path = args.exp_res_dir + filename

        if args.dataset == 'MMLU':
            subject_list = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', \
            'college_biology', 'college_chemistry', 'college_computer_science','college_mathematics','college_medicine',\
            'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', \
            'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry',\
            'high_school_computer_science', 'high_school_european_history','high_school_geography', 'high_school_government_and_politics', \
            'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology',\
            'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law',\
            'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', \
            'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', \
            'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']

            acc_list = []
            for _subject in subject_list:
                print(' ===== Subject ',_subject,' ===== ')
                args.subject = _subject

                args = load_parties_llm(args)
                args.basic_vfl, args.main_acc_noattack = evaluate_no_attack_pretrained(args)

                result = f'{_subject}_acc:{args.main_acc_noattack}'
                append_exp_res(args.exp_res_path, result)
                acc_list.append(args.main_acc_noattack)
                torch.cuda.empty_cache()

            avg_acc = np.mean(acc_list)
            final_result = f'Average_acc:{avg_acc}'
            append_exp_res(args.exp_res_path, final_result)
            print(final_result)

        else:
            args = load_parties_llm(args)
            # commuinfo='== commu:'+args.communication_protocol
            # append_exp_res(args.exp_res_path, commuinfo)
            if args.pretrained == 1:
                args.basic_vfl, args.main_acc_noattack = evaluate_no_attack_pretrained(args)
            else:
                args.basic_vfl, args.main_acc_noattack = evaluate_no_attack_finetune(args)
