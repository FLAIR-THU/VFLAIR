import os
import sys
import numpy as np
import time

import random
import logging
import argparse
import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from torchvision import datasets
# import torch.utils
# import torch.backends.cudnn as cudnn
# from tensorboardX import SummaryWriter
from peft.peft_model import PeftModel

from load.LoadConfigs import *  # load_configs
from load.LoadParty import load_parties, load_parties_llm

from evaluates.MainTaskVFL_LLM import *
from utils.basic_functions import append_exp_res
from utils import recorder

from load.LoadConfigs import INVERSION

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
    set_seed(args.current_seed)

    vfl = MainTaskVFL_LLM(args)
    vfl.init_communication()

    exp_result, metric_val = vfl.inference()

    # # Save record 
    exp_result = f"NoAttack|{args.pad_info}|seed={args.current_seed}|K={args.k}" + exp_result
    print(exp_result)
    append_exp_res(args.exp_res_path, exp_result)

    return vfl, metric_val


def evaluate_no_attack_finetune(args):
    # No Attack
    set_seed(args.current_seed)

    vfl = MainTaskVFL_LLM(args)
    vfl.init_communication()

    exp_result, metric_val, training_time = vfl.train_vfl()

    # attack_metric = main_acc_noattack - main_acc
    # attack_metric_name = 'acc_loss'

    # # Save record 
    exp_result = f"NoAttack|{args.pad_info}|finetune={args.finetune_name}|seed={args.current_seed}|K={args.k}|bs={args.batch_size}|LR={args.main_lr}|num_class={args.num_classes}|Q={args.Q}|epoch={args.main_epochs}|headlayer={args.head_layer_trainable}|encoder={args.encoder_trainable}|embedding={args.embedding_trainable}|local_encoders_num={args.local_encoders_num}|" \
                 + exp_result
    print(exp_result)

    append_exp_res(args.exp_res_path, exp_result)

    return vfl, metric_val


def evaluate_inversion_attack(args):
    for index in args.inversion_index:
        torch.cuda.empty_cache()
        set_seed(args.current_seed)

        args = load_attack_configs(args.configs, args, index)
        print('======= Test Attack', index, ': ', args.attack_name, ' =======')
        print('attack configs:', args.attack_configs)

        if args.basic_vfl != None:
            vfl = args.basic_vfl
            main_tack_acc = args.main_acc_noattack
        else:
            # args.need_auxiliary = 1
            args = load_parties_llm(args)
            vfl = MainTaskVFL_LLM(args)
            vfl.init_communication()

            if args.pipeline == 'finetune':
                _exp_result, metric_val, training_time = vfl.train_vfl()
            elif args.pipeline == 'pretrained':
                _exp_result, metric_val = vfl.inference()
            main_tack_acc = metric_val
            print(_exp_result)

        print('=== Begin Attack ===')
        training_time = vfl.training_time
        train_party_time = vfl.train_party_time
        inference_party_time = vfl.inference_party_time
        precision, recall , attack_total_time= vfl.evaluate_attack()

        exp_result = f"{args.attack_name}|{args.pad_info}|finetune={args.finetune_name}|seed={args.current_seed}|K={args.k}|bs={args.batch_size}|LR={args.main_lr}|num_class={args.num_classes}|Q={args.Q}|epoch={args.main_epochs}|final_epoch={vfl.final_epoch}|headlayer={args.head_layer_trainable}|encoder={args.encoder_trainable}|embedding={args.embedding_trainable}|local_encoders_num={args.local_encoders_num}|main_task_acc={main_tack_acc}|precision={precision}|recall={recall}|training_time={training_time}|attack_time={attack_total_time}|train_party_time={train_party_time}|inference_party_time={inference_party_time}"
        print(exp_result)
        append_exp_res(args.exp_res_path, exp_result)
        return precision, recall


def get_cls_ancestor(model_type: str = 'qwen2', architecture: str = 'CLM'):
    if model_type == 'chatglm':
        from models.llm_models import chatglm
        target_cls = getattr(chatglm, "ChatGLMForConditionalGeneration")
    elif model_type == 'baichuan':
        from models.llm_models import baichuan
        target_cls = getattr(baichuan, "BaiChuanForCausalLM")
    else:
        from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, \
            MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES, MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
        target_module = __import__('transformers')
        aa = {"CLM": MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
              "TQA": MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
              "CLS": MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES}[architecture][model_type]
        target_cls = getattr(target_module, aa)
    return target_cls

def create_exp_dir_and_file(dataset, Q, model_name, pipeline, defense_name='', defense_param=''):
    exp_res_dir = f'exp_result/{dataset}/Q{str(Q)}/'
    if not os.path.exists(exp_res_dir):
        os.makedirs(exp_res_dir)
    if pipeline == 'pretrained':
        filename = f'{defense_name}_{defense_param},pretrained_model={model_name}.txt'
    else:
        filename = f'{defense_name}_{defense_param},finetuned_model={model_name}.txt'
    exp_res_path = exp_res_dir + str(filename).replace('/', '')
    return exp_res_dir, exp_res_path

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
    if args.seed != 97:
        seed_list = [args.seed]
    else:
        seed_list = [60, 61, 62, 63, 64]
    for seed in seed_list:  # [60,61,62,63,64]: # test 5 times
        args.current_seed = seed
        set_seed(seed)
        print('================= iter seed ', seed, ' =================')

        args = load_basic_configs(args.configs, args)
        args.need_auxiliary = 0  # no auxiliary dataset for attackerB

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
        print('Defense_Name:', args.defense_name)
        print('Defense_Config:', str(args.defense_configs))
        print('===== Total Attack Tested:', args.attack_num, ' ======')
        print('inversion:', args.inversion_list, args.inversion_index)

        # Save record for different defense method
        model_name = args.model_list[str(0)]["type"]  # .replace('/','-')
        exp_res_dir, exp_res_path = create_exp_dir_and_file(args.dataset, args.Q, model_name, args.pipeline, args.defense_name,args.defense_param)
        args.exp_res_dir = exp_res_dir
        args.exp_res_path = exp_res_path
        print(args.exp_res_path)
        print('=================================\n')

        iterinfo = '===== iter ' + str(seed) + ' ===='
        # append_exp_res(args.exp_res_path, iterinfo)
        print(iterinfo)

        args.basic_vfl_withaux = None
        args.main_acc_noattack_withaux = None
        args.basic_vfl = None
        args.main_acc_noattack = None

        args = load_attack_configs(args.configs, args, -1)

        

        # if args.dataset == 'MMLU':
        #     # subject_list = ['business_ethics',\
        #     # 'abstract_algebra','anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', \
        #     # 'college_biology', 'college_chemistry', 'college_computer_science','college_mathematics','college_medicine',\
        #     # 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', \
        #     # 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry',\
        #     # 'high_school_computer_science', 'high_school_european_history','high_school_geography', 'high_school_government_and_politics', \
        #     # 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology',\
        #     # 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law',\
        #     # 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', \
        #     # 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', \
        #     # 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']

        #     # acc_list = []
        #     # for _subject in subject_list:
        #     #     print(' ===== Subject ',_subject,' ===== ')
        #     #     subject_info = f"subject={_subject}"
        #     #     append_exp_res(args.exp_res_path, subject_info)
        #     #     args.subject = _subject

        #     args = load_parties_llm(args)

        #     # inherit generation functions from global model
        #     # args.global_model_type = type(args.parties[-1].global_model)
        #     # ancestor_cls = args.global_model_type
        #     # todo: infer from model_type might be enough, would also work under 3-slice
        #     ancestor_cls = get_cls_ancestor(args.config.model_type, args.model_architect)
        #     MainTaskVFL_LLM = create_main_task(ancestor_cls)

        #     # vanilla
        #     if args.pipeline == 'pretrained':
        #         args.basic_vfl, args.main_acc_noattack = evaluate_no_attack_pretrained(args)
        #     elif args.pipeline == 'finetune':
        #         args.basic_vfl, args.main_acc_noattack = evaluate_no_attack_finetune(args)
        #     # acc_list.append(args.main_acc_noattack)

        #     # with attack
        #     precision_list = []
        #     recall_list = []
        #     if args.inversion_list != []:
        #         precision, recall = evaluate_inversion_attack(args)
        #         precision_list.append(precision)
        #         recall_list.append(recall)

        #     torch.cuda.empty_cache()

        #     # avg_acc = np.mean(acc_list)
        #     # avg_precision = np.mean(precision_list)
        #     # avg_recall = np.mean(recall_list)
        #     # final_info = f"MMLU_avg_acc={avg_acc}|precision={avg_precision}|recall={avg_recall}"
        #     # append_exp_res(args.exp_res_path, final_info)
        # else:

        args = load_parties_llm(args)

        ###### inherit generation functions from global model
        # args.global_model_type = type(args.parties[-1].global_model)
        # ancestor_cls = args.global_model_type
        # todo: infer from model_type might be enough, would also work under 3-slice
        ancestor_cls = get_cls_ancestor(args.config.model_type, args.model_architect)
        MainTaskVFL_LLM = create_main_task(ancestor_cls)

        # commuinfo='== metrics:'+args.metric_type
        # append_exp_res(args.exp_res_path, commuinfo)

        # vanilla
        if args.pipeline == 'pretrained':
            args.basic_vfl, args.main_acc_noattack = evaluate_no_attack_pretrained(args)
        elif args.pipeline == 'finetune':
            args.basic_vfl, args.main_acc_noattack = evaluate_no_attack_finetune(args)

        # with attack
        if args.inversion_list != []:
            evaluate_inversion_attack(args)

        logger.info(recorder)
