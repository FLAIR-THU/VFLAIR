import sys, os

sys.path.append(os.pardir)

import argparse
import numpy as np
import pickle
from transformers import BertTokenizer, GPT2Tokenizer, LlamaTokenizer
from transformers import BertModel, GPT2Model, LlamaModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, \
    AutoModelForCausalLM, AutoModelForQuestionAnswering
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
import torch.nn as nn
import torch
import copy
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, PeftModelForCausalLM

from models.llm_models.bert import *
from models.llm_models.gpt2 import *
from models.llm_models.llama import *
from models.llm_models.baichuan import *
from models.llm_models.chatglm import *
from models.llm_models.falcon import *
from models.llm_models.mamba import *
from models.llm_models.gemma import *
from models.llm_models.mistral import *
from models.llm_models.xlnet import *
from models.llm_models.chatglm import *


from models.bottom_models import *
from models.global_models import *
from models.autoencoder import *
from utils.optimizers import MaliciousSGD, MaliciousAdam
from config import vfl_basic_config

YOUR_MODEL_PATH = "/home/DAIR/guzx/Git_FedProject/Models/"
MODEL_PATH = {
    'bert-base-uncased': YOUR_MODEL_PATH + "bert-base-uncased",
    "textattackbert-base-uncased-CoLA": YOUR_MODEL_PATH + "textattackbert-base-uncased-CoLA",
    "textattackbert-base-uncased-yelp-polarity": YOUR_MODEL_PATH + "textattackbert-base-uncased-yelp-polarity",
    "textattackbert-base-uncased-SST-2": YOUR_MODEL_PATH + "textattackbert-base-uncased-SST-2",
    "textattackbert-base-cased-STS-B": YOUR_MODEL_PATH + "textattackbert-base-cased-STS-B",
    "textattackbert-base-uncased-MRPC": YOUR_MODEL_PATH + "textattackbert-base-uncased-MRPC",
    "textattackbert-base-uncased-MNLI": YOUR_MODEL_PATH + "textattackbert-base-uncased-MNLI",
    "textattackbert-base-uncased-QNLI": YOUR_MODEL_PATH + "textattackbert-base-uncased-QNLI",
    "textattackbert-base-uncased-QQP": YOUR_MODEL_PATH + "textattackbert-base-uncased-QQP",
    "textattackbert-base-uncased-WNLI": YOUR_MODEL_PATH + "textattackbert-base-uncased-WNLI",
    "textattackbert-base-uncased-RTE": YOUR_MODEL_PATH + "textattackbert-base-uncased-RTE",
    "textattackroberta-base-CoLA": YOUR_MODEL_PATH + "textattackbert-base-uncased-CoLA",
    "textattackroberta-base-SST-2": YOUR_MODEL_PATH + "textattackroberta-base-SST-2",
    "textattackalbert-base-v2-CoLA": YOUR_MODEL_PATH + "textattackalbert-base-v2-CoLA",
    "textattackroberta-base-MNLI": YOUR_MODEL_PATH + "textattackroberta-base-MNLI",
    "nihaldsouza1yelp-rating-classification": YOUR_MODEL_PATH + "nihaldsouza1yelp-rating-classification",
    "deepsetroberta-base-squad2": YOUR_MODEL_PATH + "deepsetroberta-base-squad2",
    "rsvp-aibertserini-bert-base-squad": YOUR_MODEL_PATH + "rsvp-aibertserini-bert-base-squad",
    "FabianWillnerdistilbert-base-uncased-finetuned-squad": YOUR_MODEL_PATH + "FabianWillnerdistilbert-base-uncased-finetuned-squad",
    "google-bertbert-large-uncased-whole-word-masking-finetuned-squad": YOUR_MODEL_PATH + "google-bertbert-large-uncased-whole-word-masking-finetuned-squad",
    "Shunianyelp_review_classification": YOUR_MODEL_PATH + "Shunianyelp_review_classification",

    "gpt2": YOUR_MODEL_PATH + "gpt2",
    "gpt2-medium": YOUR_MODEL_PATH + "gpt2-medium",
    "gpt2-large": YOUR_MODEL_PATH + "gpt2-large",
    "George-Ogdengpt2-medium-finetuned-mnli": YOUR_MODEL_PATH + "George-Ogdengpt2-medium-finetuned-mnli",
    "michelecafagna26gpt2-medium-finetuned-sst2-sentiment": YOUR_MODEL_PATH + "michelecafagna26gpt2-medium-finetuned-sst2-sentiment",
    "tanzeelabbasGPT-2_fine-tuned_squad_2.0_QA": YOUR_MODEL_PATH + "tanzeelabbasGPT-2_fine-tuned_squad_2.0_QA",

    'llama-2-7b': YOUR_MODEL_PATH + "llama-2-7b",
    'chavinloalpaca-native': YOUR_MODEL_PATH + "chavinloalpaca-native",
    "lqtrung1998Codellama-7b-hf-ReFT-GSM8k": YOUR_MODEL_PATH+"lqtrung1998Codellama-7b-hf-ReFT-GSM8k",
    "meta-mathMetaMath-Mistral-7B" : YOUR_MODEL_PATH + "meta-mathMetaMath-Mistral-7B",

    "HuggingFaceM4tiny-random-LlamaForCausalLM": YOUR_MODEL_PATH + "HuggingFaceM4tiny-random-LlamaForCausalLM",
    "HuggingFaceH4tiny-random-LlamaForSequenceClassification": YOUR_MODEL_PATH + "HuggingFaceH4tiny-random-LlamaForSequenceClassification",

    "AudreyTrungNguyenllama-qnli-p-tuning": YOUR_MODEL_PATH + "AudreyTrungNguyenllama-qnli-p-tuning",

    "googleflan-t5-base": YOUR_MODEL_PATH + "googleflan-t5-base",
    "echarlaixtiny-random-mistral": YOUR_MODEL_PATH + "echarlaixtiny-random-mistral",
    "baichuan-incBaichuan-7B": YOUR_MODEL_PATH + "baichuan-incBaichuan-7B",
    "fxmartytiny-random-GemmaForCausalLM": YOUR_MODEL_PATH + 'fxmartytiny-random-GemmaForCausalLM',
    "state-spacesmamba-130m-hf": YOUR_MODEL_PATH + "state-spacesmamba-130m-hf",
    "THUDMchatglm3-6b": YOUR_MODEL_PATH + "THUDMchatglm3-6b",
    "katuni4katiny-random-chatglm2": YOUR_MODEL_PATH + "katuni4katiny-random-chatglm2",
    "xlnetxlnet-base-cased": YOUR_MODEL_PATH + "xlnetxlnet-base-cased",
    "tiiuaefalcon-rw-1b": YOUR_MODEL_PATH + "tiiuaefalcon-rw-1b",
    "fxmartytiny-random-GemmaForCausalLM": YOUR_MODEL_PATH + "fxmartytiny-random-GemmaForCausalLM",
    "mistralaiMistral-7B-Instruct-v0.2": YOUR_MODEL_PATH + "mistralaiMistral-7B-Instruct-v0.2",

    "googlegemma-2b": YOUR_MODEL_PATH + "googlegemma-2b",

}
MODEL_PATH.update({'Qwen/Qwen1.5-0.5B-Chat': None})

LLM_supported = MODEL_PATH.keys()


# ['bert-base-uncased','Bert-sequence-classification',"toxic-bert",\
# "textattackbert-base-uncased-CoLA","textattackbert-base-uncased-SST-2","textattackbert-base-uncased-STS-B",\
# "textattackbert-base-uncased-MRPC","textattackbert-base-uncased-MNLI","textattackbert-base-uncased-QNLI",\
# "textattackbert-base-uncased-QQP","textattackbert-base-uncased-WNLI","textattackbert-base-uncased-RTE",\
# "textattackroberta-base-CoLA","textattackroberta-base-SST-2","textattackalbert-base-v2-CoLA","textattackroberta-base-MNLI",\
# "michelecafagna26gpt2-medium-finetuned-sst2-sentiment","gpt2","gpt2-medium","gpt2-large",\
# "George-Ogdengpt2-medium-finetuned-mnli","Shitaollama2-mnli","llama-2-7b","benayasllama-2-7b-sst2_v0",\
# "AudreyTrungNguyenllama-qnli-p-tuning","HuggingFaceM4tiny-random-LlamaForCausalLM","rsvp-aibertserini-bert-base-squad",\
# "tanzeelabbasGPT-2_fine-tuned_squad_2.0_QA"]


def create_model(bottom_model, ema=False, size_bottom_out=10, num_classes=10):
    model = BottomModelPlus(bottom_model, size_bottom_out, num_classes,
                            num_layer=2,
                            activation_func_type='ReLU',
                            use_bn=0)
    model = model

    if ema:
        for param in model.parameters():
            param.detach_()

    return model


def load_models(args):
    args.net_list = [None] * args.k
    for ik in range(args.k):
        current_model_type = args.model_list[str(ik)]['type']
        current_model_path = args.model_list[str(ik)]['path']
        args.net_list[ik] = pickle.load(
            open('.././src/models/model_parameters/' + current_model_type + '/' + current_model_path + '.pkl', "rb"))
        args.net_list[ik] = args.net_list[ik].to(args.device)
    # important
    return args


def load_basic_models(args, index):
    current_model_type = args.model_list[str(index)]['type']
    print(f"current_model_type={current_model_type}")
    current_input_dim = args.model_list[str(index)]['input_dim'] if 'input_dim' in args.model_list[str(index)] else -1
    current_hidden_dim = args.model_list[str(index)]['hidden_dim'] if 'hidden_dim' in args.model_list[
        str(index)] else -1
    current_output_dim = args.model_list[str(index)]['output_dim'] if 'output_dim' in args.model_list[
        str(index)] else -1
    current_vocab_size = args.model_list[str(index)]['vocab_size'] if 'vocab_size' in args.model_list[
        str(index)] else -1
    # print(f"index={index}, current_input_dim={current_input_dim}, current_output_dim={current_output_dim}")
    # current_model_path = args.model_list[str(index)]['path']
    # local_model = pickle.load(open('.././model_parameters/'+current_model_type+'/'+current_model_path+'.pkl',"rb"))
    if 'resnet' in current_model_type.lower() or 'lenet' in current_model_type.lower() or 'cnn' in current_model_type.lower() or 'alexnet' in current_model_type.lower():
        local_model = globals()[current_model_type](current_output_dim)
    elif 'gcn' in current_model_type.lower():
        local_model = globals()[current_model_type](nfeat=current_input_dim, nhid=current_hidden_dim,
                                                    nclass=current_output_dim, device=args.device,
                                                    dropout=0.0, lr=args.main_lr)
    elif 'lstm' in current_model_type.lower():
        local_model = globals()[current_model_type](current_vocab_size, current_output_dim)
    else:
        local_model = globals()[current_model_type](current_input_dim, current_output_dim)
    local_model = local_model.to(args.device)
    print(f"local_model parameters: {sum(p.numel() for p in local_model.parameters())}")
    local_model_optimizer = torch.optim.Adam(list(local_model.parameters()), lr=args.main_lr, weight_decay=0.0)
    # print(f"use SGD for local optimizer for PMC checking")
    # local_model_optimizer = torch.optim.SGD(list(local_model.parameters()), lr=args.main_lr, momentum=0.9, weight_decay=5e-4)

    # update optimizer
    if 'activemodelcompletion' in args.attack_name.lower() and index in args.attack_configs['party']:
        print('AMC: use Malicious optimizer for party', index)
        # local_model_optimizer = torch.optim.Adam(list(local_model.parameters()), lr=args.main_lr, weight_decay=0.0)     
        # local_model_optimizer = MaliciousSGD(list(local_model.parameters()), lr=args.main_lr, momentum=0.9, weight_decay=5e-4)
        local_model_optimizer = MaliciousAdam(list(local_model.parameters()), lr=args.main_lr)

    global_model = None
    global_model_optimizer = None
    if index == args.k - 1:
        if args.apply_trainable_layer == 0:
            global_model = globals()[args.global_model]()
            global_model = global_model.to(args.device)
            global_model_optimizer = None
        else:
            print("global_model", args.global_model)
            global_input_dim = 0
            for ik in range(args.k):
                global_input_dim += args.model_list[str(ik)]['output_dim']
            global_model = globals()[args.global_model](global_input_dim, args.num_classes)
            global_model = global_model.to(args.device)
            global_model_optimizer = torch.optim.Adam(list(global_model.parameters()), lr=args.main_lr)
            # print(f"use SGD for global optimizer for PMC checking")
            # global_model_optimizer = torch.optim.SGD(list(global_model.parameters()), lr=args.main_lr, momentum=0.9, weight_decay=5e-4)

    return args, local_model, local_model_optimizer, global_model, global_model_optimizer


# def load_defense_models(args, index, local_model, local_model_optimizer, global_model, global_model_optimizer):
#     print('Load Defense models')
#     # no defense at all, set some variables as None
#     args.encoder = None
#     # some defense need model, add here
#     if args.apply_defense == True:
#         current_bottleneck_scale = int(
#             args.defense_configs['bottleneck_scale']) if 'bottleneck_scale' in args.defense_configs else 1

#         if 'std_shift_hyperparameter' in args.defense_configs:
#             std_shift_hyperparameter = int(args.defense_configs['std_shift_hyperparameter'])
#         else:
#             std_shift_hyperparameter = 5 if (
#                         'mnist' in args.dataset.lower() or 'nuswide' == args.dataset.lower() or 'cifar' in args.dataset.lower()) else 0.5

#         if 'MID' in args.defense_name.upper():
#             if not 'party' in args.defense_configs:
#                 args.defense_configs['party'] = [args.k - 1]
#                 print('[warning] default active party selected for applying MID')
#             if not 'lambda' in args.defense_configs:
#                 args.defense_configs['lambda'] = 0.001
#                 print('[warning] default hyper-parameter lambda selected for applying MID')
#             if not ('lr' in args.defense_configs):
#                 mid_lr = args.main_lr
#                 print('[warning] default hyper-parameter mid_lr selected for applying MID')
#             else:
#                 mid_lr = args.defense_configs['lr']

#             print(f"mid defense parties: {args.defense_configs['party']}")
#             if index in args.defense_configs['party']:
#                 print(f"begin to load mid model for party {index}")
#                 if index == args.k - 1:
#                     print(
#                         f"load global mid model for party {index},std_shift_hyperparameter={std_shift_hyperparameter}")
#                     # add args.k-1 MID model at active party with global_model
#                     if 'nuswide' in args.dataset.lower() or 'nus-wide' in args.dataset.lower():
#                         print(f"small MID model for nuswide")
#                         mid_model_list = [
#                             MID_model_small(args.model_list[str(_ik)]['output_dim'],
#                                             args.model_list[str(_ik)]['output_dim'], args.defense_configs['lambda'],
#                                             bottleneck_scale=current_bottleneck_scale,
#                                             std_shift=std_shift_hyperparameter) for _ik in range(args.k - 1)]
#                     else:
#                         mid_model_list = [
#                             MID_model(args.model_list[str(_ik)]['output_dim'], args.model_list[str(_ik)]['output_dim'],
#                                       args.defense_configs['lambda'],
#                                       bottleneck_scale=current_bottleneck_scale, std_shift=std_shift_hyperparameter) for
#                             _ik in range(args.k - 1)]
#                     mid_model_list = [model.to(args.device) for model in mid_model_list]
#                     global_model = Active_global_MID_model(global_model, mid_model_list)
#                     global_model = global_model.to(args.device)
#                     # update optimizer
#                     if args.apply_trainable_layer == 0:
#                         parameters = []
#                         for mid_model in global_model.mid_model_list:
#                             parameters += list(mid_model.parameters())
#                         global_model_optimizer = torch.optim.Adam(parameters, lr=mid_lr)
#                         print(f"mid_lr = {mid_lr}")
#                     else:
#                         parameters = []
#                         for mid_model in global_model.mid_model_list:
#                             parameters += list(mid_model.parameters())
#                         global_model_optimizer = torch.optim.Adam(
#                             [{'params': global_model.global_model.parameters(), 'lr': args.main_lr},
#                              {'params': parameters, 'lr': mid_lr}])
#                         print(f"mid_lr = {mid_lr}")

#                 else:
#                     print(f"load local mid model for party {index}")
#                     # add MID model at passive party with local_model
#                     print('lambda for passive party local mid model:', args.defense_configs['lambda'])
#                     if 'nuswide' in args.dataset.lower() or 'nus-wide' in args.dataset.lower():
#                         print(f"small MID model for nuswide")
#                         mid_model = MID_model_small(args.model_list[str(index)]['output_dim'],
#                                                     args.model_list[str(index)]['output_dim'],
#                                                     args.defense_configs['lambda'],
#                                                     bottleneck_scale=current_bottleneck_scale,
#                                                     std_shift=std_shift_hyperparameter)
#                     else:
#                         mid_model = MID_model(args.model_list[str(index)]['output_dim'],
#                                               args.model_list[str(index)]['output_dim'],
#                                               args.defense_configs['lambda'], bottleneck_scale=current_bottleneck_scale,
#                                               std_shift=std_shift_hyperparameter)
#                     mid_model = mid_model.to(args.device)
#                     local_model = Passive_local_MID_model(local_model, mid_model)
#                     local_model = local_model.to(args.device)

#                     # update optimizer
#                     if 'activemodelcompletion' in args.attack_name.lower() and index in args.attack_configs['party']:
#                         print('AMC: use Malicious optimizer for party', index)
#                         # local_model_optimizer = torch.optim.Adam(list(local_model.parameters()), lr=args.main_lr, weight_decay=0.0)     
#                         # local_model_optimizer = MaliciousSGD(list(local_model.parameters()), lr=args.main_lr, momentum=0.0, weight_decay=5e-4)
#                         # local_model_optimizer = MaliciousAdam(list(local_model.parameters()),lr=args.main_lr)
#                         local_model_optimizer = MaliciousAdam(
#                             [{'params': local_model.local_model.parameters(), 'lr': args.main_lr},
#                              {'params': local_model.mid_model.parameters(), 'lr': mid_lr}])
#                         # assert 1>2
#                     else:
#                         local_model_optimizer = torch.optim.Adam(
#                             [{'params': local_model.local_model.parameters(), 'lr': args.main_lr},
#                              {'params': local_model.mid_model.parameters(), 'lr': mid_lr}])

#         # for adversarial training
#         adversarial_model = None
#         if 'adversarial' in args.defense_name.lower():
#             # add adversarial model for local model
#             if not 'party' in args.defense_configs:
#                 args.defense_configs['party'] = [0]
#                 print('[warning] default passive party selected for applying adversarial training')
#             if not ('lr' in args.defense_configs):
#                 adversarial_lr = args.main_lr
#                 print('[warning] default hyper-parameter mid_lr selected for applying MID')
#             else:
#                 adversarial_lr = args.defense_configs['lr']
#             if not ('model' in args.defense_configs):
#                 model_name = 'Adversarial_MLP2'
#             else:
#                 model_name = args.defense_configs['model']
#             print(model_name)

#             if index in args.defense_configs['party']:
#                 # assert args.parties[index].train_attribute != None, "[Error] no attribute for adversarial"
#                 # add adversarial model to the the defense party=index
#                 adversarial_input_dim = args.model_list[str(index)]['output_dim']
#                 adversarial_output_dim = args.num_attributes
#                 # print(f"[debug] in load defense model, adversarial_input_dim={adversarial_input_dim}, adversarial_output_dim={adversarial_output_dim}")
#                 adversarial_model = globals()[model_name](adversarial_input_dim, adversarial_output_dim)
#                 local_model = Local_Adversarial_combined_model(local_model, adversarial_model)
#                 local_model = local_model.to(args.device)
#                 # update optimizer
#                 local_model_optimizer = torch.optim.Adam(
#                     [{'params': local_model.local_model.parameters(), 'lr': args.main_lr},
#                      {'params': local_model.adversarial_model.parameters(), 'lr': adversarial_lr}])

#         if 'CAE' in args.defense_name.upper():  # for CAE and DCAE
#             # print("CAE in defense_name,", args.defense_name)
#             if index == args.k - 1:
#                 # only active party can have encoder and decoder for CAE
#                 assert 'model_path' in args.defense_configs, "[error] no CAE model path given"
#                 if not 'input_dim' in args.defense_configs:
#                     args.defense_configs['input_dim'] = args.num_classes
#                     print('[warning] default input_dim selected as num_classes for applying CAE')
#                 if not 'encode_dim' in args.defense_configs:
#                     args.defense_configs['encode_dim'] = 2 + 6 * args.defense_configs['input_dim']
#                     print('[warning] default encode_dim selected as 2+6*input_dim for applying CAE')
#                 if args.num_classes > 20:
#                     encoder = AutoEncoder_large(real_dim=args.defense_configs['input_dim'], input_dim=20,
#                                                 encode_dim=args.defense_configs['encode_dim']).to(
#                         args.device)
#                 else:
#                     encoder = AutoEncoder(input_dim=args.defense_configs['input_dim'],
#                                           encode_dim=args.defense_configs['encode_dim']).to(args.device)
#                 encoder.load_model(args.defense_configs['model_path'], target_device=args.device)
#                 args.encoder = encoder

#         return args, local_model, local_model_optimizer, global_model, global_model_optimizer


def load_defense_models(args, index, local_model, local_model_optimizer, global_model, global_model_optimizer):
    print('Load Defense models')
    # no defense at all, set some variables as None
    args.encoder = None
    # some defense need model, add here
    if args.apply_defense == True:
        current_bottleneck_scale = int(args.defense_configs['bottleneck_scale']) if 'bottleneck_scale' in args.defense_configs else 1
        
        if 'std_shift_hyperparameter' in args.defense_configs:
            std_shift_hyperparameter = int(args.defense_configs['std_shift_hyperparameter'])
        else:
            std_shift_hyperparameter = 5 if ('mnist' in args.dataset.lower() or 'nuswide' == args.dataset.lower() or 'cifar' in args.dataset.lower()) else 0.5 

        if 'MID' in args.defense_name.upper():
            if not 'party' in args.defense_configs:
                args.defense_configs['party'] = [args.k-1]
                print('[warning] default active party selected for applying MID')
            if not 'lambda' in args.defense_configs:
                args.defense_configs['lambda'] = 0.001
                print('[warning] default hyper-parameter lambda selected for applying MID')
            if not ('lr' in args.defense_configs):
                mid_lr = args.main_lr  
                print('[warning] default hyper-parameter mid_lr selected for applying MID')
            else :
                mid_lr = args.defense_configs['lr'] 
            
            print(f"mid defense parties: {args.defense_configs['party']}")
            if index in args.defense_configs['party']:
                print(f"begin to load mid model for party {index}")
                if index == args.k-1:
                    print(f"load global mid model for party {index},std_shift_hyperparameter={std_shift_hyperparameter}")
                    # add args.k-1 MID model at active party with global_model
                    if 'nuswide' in args.dataset.lower() or 'nus-wide' in args.dataset.lower():
                        print(f"small MID model for nuswide")
                        mid_model_list = [MID_model_small(args.model_list[str(_ik)]['output_dim'],args.model_list[str(_ik)]['output_dim'],args.defense_configs['lambda'],bottleneck_scale=current_bottleneck_scale, std_shift=std_shift_hyperparameter) for _ik in range(args.k-1)]
                    else:
                        mid_model_list = [MID_model(args.model_list[str(_ik)]['output_dim'],args.model_list[str(_ik)]['output_dim'],args.defense_configs['lambda'],bottleneck_scale=current_bottleneck_scale, std_shift=std_shift_hyperparameter) for _ik in range(args.k-1)]
                    mid_model_list = [model.to(args.device) for model in mid_model_list]
                    global_model = Active_global_MID_model(global_model,mid_model_list)
                    global_model = global_model.to(args.device)
                    # update optimizer
                    if args.apply_trainable_layer == 0:
                        parameters = []
                        for mid_model in global_model.mid_model_list:
                            parameters += list(mid_model.parameters())
                        global_model_optimizer = torch.optim.Adam(parameters, lr=mid_lr)
                        print(f"mid_lr = {mid_lr}")
                    else:
                        parameters = []
                        for mid_model in global_model.mid_model_list:
                            parameters += list(mid_model.parameters())
                        global_model_optimizer = torch.optim.Adam(
                            [{'params': global_model.global_model.parameters(), 'lr': args.main_lr},              
                            {'params': parameters, 'lr': mid_lr}])
                        print(f"mid_lr = {mid_lr}")
                        
                else:
                    print(f"load local mid model for party {index}")
                    # add MID model at passive party with local_model
                    print('lambda for passive party local mid model:',args.defense_configs['lambda'])
                    if 'nuswide' in args.dataset.lower() or 'nus-wide' in args.dataset.lower():
                        print(f"small MID model for nuswide")
                        mid_model = MID_model_small(args.model_list[str(index)]['output_dim'],args.model_list[str(index)]['output_dim'],args.defense_configs['lambda'],bottleneck_scale=current_bottleneck_scale, std_shift=std_shift_hyperparameter)
                    else:
                        mid_model = MID_model(args.model_list[str(index)]['output_dim'],args.model_list[str(index)]['output_dim'],args.defense_configs['lambda'],bottleneck_scale=current_bottleneck_scale, std_shift=std_shift_hyperparameter)
                    mid_model = mid_model.to(args.device)
                    local_model = Passive_local_MID_model(local_model,mid_model)
                    local_model = local_model.to(args.device)

                    # update optimizer
                    if 'activemodelcompletion' in args.attack_name.lower() and index in args.attack_configs['party']:
                        print('AMC: use Malicious optimizer for party', index)
                        # local_model_optimizer = torch.optim.Adam(list(local_model.parameters()), lr=args.main_lr, weight_decay=0.0)     
                        # local_model_optimizer = MaliciousSGD(list(local_model.parameters()), lr=args.main_lr, momentum=0.0, weight_decay=5e-4)
                        # local_model_optimizer = MaliciousAdam(list(local_model.parameters()),lr=args.main_lr)
                        local_model_optimizer = MaliciousAdam(
                            [{'params': local_model.local_model.parameters(), 'lr': args.main_lr},              
                            {'params': local_model.mid_model.parameters(), 'lr': mid_lr}])
                        # assert 1>2
                    else:
                        local_model_optimizer = torch.optim.Adam(
                            [{'params': local_model.local_model.parameters(), 'lr': args.main_lr},              
                            {'params': local_model.mid_model.parameters(), 'lr': mid_lr}])

        if 'adversarial' in args.defense_name.lower(): # for adversarial training
            # add adversarial model for local model
            if not 'party' in args.defense_configs:
                args.defense_configs['party'] = [0]
                print('[warning] default passive party selected for applying adversarial training')
            if not ('lr' in args.defense_configs):
                adversarial_lr = args.main_lr  
                print('[warning] default hyper-parameter mid_lr selected for applying MID')
            else :
                adversarial_lr = args.defense_configs['lr']
            if not ('model' in args.defense_configs):
                model_name = 'Adversarial_MLP2'
            else:
                model_name = args.defense_configs['model']
            print(model_name)
            if index in args.defense_configs['party']:
                # assert args.parties[index].train_attribute != None, "[Error] no attribute for adversarial"
                # add adversarial model to the the defense party=index
                adversarial_input_dim = args.model_list[str(index)]['output_dim']
                adversarial_output_dim = args.num_attributes
                # print(f"[debug] in load defense model, adversarial_input_dim={adversarial_input_dim}, adversarial_output_dim={adversarial_output_dim}")
                adversarial_model = globals()[model_name](adversarial_input_dim, adversarial_output_dim)
                local_model = Local_Adversarial_combined_model(local_model,adversarial_model)
                local_model = local_model.to(args.device)
                # update optimizer
                local_model_optimizer = torch.optim.Adam(
                            [{'params': local_model.local_model.parameters(), 'lr': args.main_lr},              
                            {'params': local_model.adversarial_model.parameters(), 'lr': adversarial_lr}])
            
        if 'CAE' in args.defense_name.upper(): # for CAE and DCAE
            # print("CAE in defense_name,", args.defense_name)
            if index == args.k-1:
                # only active party can have encoder and decoder for CAE
                assert 'model_path' in args.defense_configs, "[error] no CAE model path given"
                if not 'input_dim' in args.defense_configs:
                    args.defense_configs['input_dim'] = args.num_classes
                    print('[warning] default input_dim selected as num_classes for applying CAE')
                if not 'encode_dim' in args.defense_configs:
                    args.defense_configs['encode_dim'] = 2 + 6 * args.defense_configs['input_dim']
                    print('[warning] default encode_dim selected as 2+6*input_dim for applying CAE')
                if args.num_classes > 20:
                    encoder = AutoEncoder_large(real_dim=args.defense_configs['input_dim'], input_dim=20, encode_dim=args.defense_configs['encode_dim']).to(args.device)
                else:
                    encoder = AutoEncoder(input_dim=args.defense_configs['input_dim'], encode_dim=args.defense_configs['encode_dim']).to(args.device)
                encoder.load_model(args.defense_configs['model_path'], target_device=args.device)
                args.encoder = encoder
    return args, local_model, local_model_optimizer, global_model, global_model_optimizer



def load_basic_models_llm_bert(args, index):
    current_model_type = args.model_list[str(index)]['type']
    current_output_dim = args.model_list[str(index)]['output_dim']
    model_path = args.model_list[str(index)]['path']

    print('load_basic_models_llm from:', current_model_type)
    args.tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=True)
    args.tokenizer.padding_side = args.padding_side if (args.padding_side in ["left", "right"]) else "left"

    if args.model_architect == 'CLM':
        full_model = AutoModelForCausalLM.from_pretrained(model_path)
    elif args.model_architect == 'TQA':
        full_model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    elif args.model_architect == 'CLS':
        full_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    else:
        assert 1 > 2, "task type not supported"

    if args.model_type == 'Roberta':
        full_llm = full_model.roberta
    elif args.model_type == 'Albert':
        full_llm = full_model.albert
    else:  # Bert
        full_llm = full_model.bert

    # full_qwen = full_model.model
    if args.model_architect == 'CLM':
        head_layer = full_model.cls
    elif args.model_architect == 'TQA':
        head_layer = full_model.qa_outputs
    elif args.model_architect == 'CLS':
        head_layer = full_model.classifier
    else:
        head_layer = None

    args.config = full_model.config
    args.generation_config = full_model.generation_config
    args.model_architectures = args.config.architectures
    args.model_embedded_dim = args.config.hidden_size

    all_encoder_num = args.config.num_hidden_layers
    print('all_encoder_num:', all_encoder_num)

    if args.pad_token == "default":
        print('Default pad')
        if args.tokenizer.pad_token is None:
            args.tokenizer.pad_token = args.tokenizer.eos_token  # ({'pad_token': '[PAD]'}) # args.tokenizer.eos_token #
            pad_id = args.tokenizer.convert_tokens_to_ids(args.tokenizer.eos_token)  #
            full_model.config.pad_token_id = pad_id
        args.pad_token = "default_" + args.tokenizer.pad_token
    else:
        args.tokenizer.pad_token = args.pad_token  # ({'pad_token': '[PAD]'}) # args.tokenizer.eos_token #
        pad_id = args.tokenizer.convert_tokens_to_ids(args.pad_token)  #
        full_model.config.pad_token_id = pad_id

    ########### Local Model ###########
    local_model = None
    local_model_optimizer = None
    if index < args.k - 1:
        print('Local Model encoders_num:', args.local_encoders_num)
        local_model = LocalBertModel(full_llm, args.local_encoders_num, model_type=args.model_type)
        local_model = local_model.to(args.device)

        if args.finetune_name == "LoRA":
            # print('args.finetune_detail_configs:',args.finetune_detail_configs)
            if args.finetune_detail_configs != None:
                lora_config = LoraConfig(
                    **args.finetune_detail_configs
                )
            else:
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=4,
                    lora_alpha=32,
                    lora_dropout=0.1
                )


            def get_lora_model(model):
                model.enable_input_require_grads()
                peft_model = get_peft_model(model, lora_config)
                return peft_model

            local_model = get_lora_model(local_model)
            print('after lora')
            local_model.print_trainable_parameters()

            encoder_trainable_ids = args.encoder_trainable_ids_list[index]
            print('encoder_trainable_ids = ', encoder_trainable_ids)
            for encoder_id in range(len(local_model.encoder_layer)):
                if encoder_id not in encoder_trainable_ids:
                    for param in local_model.encoder_layer.parameters():
                        param.requires_grad = False

            print('embedding_trainable = ', args.embedding_trainable[0])
            if args.embedding_trainable[0] == False:
                for param in local_model.embeddings.parameters():
                    param.requires_grad = False

            print('local final trainable param:')
            local_model.print_trainable_parameters()

            local_trainable_params = list(filter(lambda x: x.requires_grad, local_model.parameters()))

            local_model_optimizer = torch.optim.Adam(local_trainable_params, lr=args.main_lr)

        else:
            # Freeze Backbone
            for param in local_model.parameters():
                param.requires_grad = False
            print(f"local_model parameters: {sum(p.numel() for p in local_model.parameters())}")

            local_model_optimizer = None
            local_trainable_params = []
            print('Local Model: embedding_trainable = ', args.embedding_trainable[0])
            for param in local_model.embeddings.parameters():
                param.requires_grad = args.embedding_trainable[0]
            if args.embedding_trainable[0]:
                local_trainable_params.extend(list(local_model.embeddings.parameters()))

            print('Local Model: args.encoder_trainable = ', args.encoder_trainable[0])
            for param in local_model.encoder_layer.parameters():
                param.requires_grad = args.encoder_trainable[0]
            if args.encoder_trainable[0]:
                local_trainable_params.extend(list(local_model.encoder_layer.parameters()))

            if len(local_trainable_params) > 0:
                local_model_optimizer = torch.optim.Adam(local_trainable_params, lr=args.main_lr)

    ########### Global Model ###########
    global_model = None
    global_model_optimizer = None
    if index == args.k - 1:
        global_encoders_num = all_encoder_num - args.local_encoders_num
        print('global_encoders_num:', global_encoders_num)

        # global part of gpt2(frozen)
        global_model = GlobalBertModel(full_llm, global_encoders_num, model_type=args.model_type)

        # add Classification Layer(untrainable)
        if args.model_architect == 'CLM':
            global_model = BertLMHeadModel_pretrained(global_model, head_layer)
        elif args.model_architect == 'TQA':
            global_model = BertForQuestionAnswering_pretrained(global_model, head_layer)
        elif args.model_architect == 'CLS':
            global_model = BertForSequenceClassification_pretrained(global_model, head_layer)
        elif args.task_type == "Generation":
            global_model = BertLMHeadModel_pretrained(global_model, head_layer)
        else:
            assert 1 > 2, "task type not supported"

        print(f"global_model parameters: {sum(p.numel() for p in global_model.parameters())}")

        # Freeze Backbone
        for param in global_model.bert.parameters():
            param.requires_grad = False

        # Head Layer Trainable/Freeze
        if head_layer:  # head layer exists
            print('Global Model : head_layer_trainable = ', args.head_layer_trainable[1])
            for param in global_model.head_layer.parameters():
                param.requires_grad = args.head_layer_trainable[1]
            if args.head_layer_trainable[1]:
                global_model_optimizer = torch.optim.Adam(list(global_model.head_layer.parameters()), lr=args.main_lr)

        global_model = global_model.to(args.device)

    del (full_model)

    return args, local_model, local_model_optimizer, global_model, global_model_optimizer


def load_basic_models_llm_t5(args, index):
    current_model_type = args.model_list[str(index)]['type']
    model_path = args.model_list[str(index)]['path']

    print('load_basic_models_llm from:', current_model_type)
    args.tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=True)
    args.tokenizer.padding_side = args.padding_side if (args.padding_side in ["left", "right"]) else "left"

    if args.model_architect == 'CLM':
        full_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    elif args.task_type == "Generation":
        full_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    # elif args.model_architect == 'TQA':
    #     full_model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    # elif args.model_architect == 'CLS':
    #     full_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    else:
        assert 1 > 2, "task type not supported"

    full_t5 = full_model
    if args.model_architect == 'CLM':
        head_layer = full_model.lm_head
    # elif args.model_architect == 'TQA':
    #     head_layer = full_model.qa_outputs
    # elif args.model_architect == 'CLS':
    #     head_layer = full_model.score
    else:
        head_layer = None

    args.config = full_model.config
    args.generation_config = full_model.generation_config
    args.model_architectures = args.config.architectures
    args.model_embedded_dim = args.config.d_model

    all_encoder_num = args.config.num_layers
    print('all_encoder_num:', all_encoder_num)

    if args.pad_token == "default":
        print('Default pad')
        if args.tokenizer.pad_token is None:
            args.tokenizer.pad_token = args.tokenizer.eos_token  # ({'pad_token': '[PAD]'}) # args.tokenizer.eos_token #
            pad_id = args.tokenizer.convert_tokens_to_ids(args.tokenizer.eos_token)  #
            full_model.config.pad_token_id = pad_id
        args.pad_token = "default_" + args.tokenizer.pad_token
    else:
        args.tokenizer.pad_token = args.pad_token  # ({'pad_token': '[PAD]'}) # args.tokenizer.eos_token #
        pad_id = args.tokenizer.convert_tokens_to_ids(args.pad_token)  #
        full_model.config.pad_token_id = pad_id

    ########### Local Model ###########
    local_model = None
    local_model_optimizer = None
    if index < args.k - 1:
        print('args.local_encoders_num:', args.local_encoders_num)
        local_model = LocalT5Model(full_t5, num_encoders=args.local_encoders_num)

        # Freeze Backbone
        for param in local_model.parameters():
            param.requires_grad = False
        local_model = local_model.to(args.device)
        print(f"local_model parameters: {sum(p.numel() for p in local_model.parameters())}")

        local_model_optimizer = None
        local_trainable_params = []
        print('Local Model: embedding_trainable = ', args.embedding_trainable[0])
        for param in local_model.shared.parameters():
            param.requires_grad = args.embedding_trainable[0]
        if args.embedding_trainable[0]:
            local_trainable_params.extend(list(local_model.shared.parameters()))
        print('Local Model: args.encoder_trainable = ', args.encoder_trainable[0])
        for param in local_model.encoder.parameters():
            param.requires_grad = args.encoder_trainable[0]
        if args.encoder_trainable[0]:
            local_trainable_params.extend(list(local_model.encoder.parameters()))

        if len(local_trainable_params) > 0:
            local_model_optimizer = torch.optim.Adam(local_trainable_params, lr=args.main_lr)

    ########### Global Model ###########
    global_model = None
    global_model_optimizer = None
    if index == args.k - 1:
        global_encoders_num = all_encoder_num - args.local_encoders_num
        print('global_encoders_num:', global_encoders_num)

        # global part of gpt2(frozen)
        global_t5 = GlobalT5Model(full_t5, num_encoders=global_encoders_num)

        # add Classification Layer(untrainable)
        if args.model_architect == 'CLM':
            global_model = T5ForConditionalGeneration_pretrained(global_t5, head_layer, generation_config=full_model.generation_config)
        # elif args.model_architect == 'TQA':
        #     global_model = T5ForConditionalGeneration_pretrained(global_t5, head_layer)
        # elif args.model_architect == 'CLS':
        #     global_model = GPT2ForSequenceClassification_pretrained(global_t5, head_layer)
        elif args.task_type == "Generation":
            global_model = T5ForConditionalGeneration_pretrained(global_t5, head_layer,
                                                                 generation_config=full_model.generation_config)
        else:
            assert 1 > 2, "task type not supported"

        print(f"global_model parameters: {sum(p.numel() for p in global_model.parameters())}")

        # Freeze Backbone
        for param in global_model.t5.parameters():
            param.requires_grad = False

        # Head Layer Trainable/Freeze
        if head_layer:  # head layer exists
            print('Global Model : head_layer_trainable = ', args.head_layer_trainable[1])
            for param in global_model.head_layer.parameters():
                param.requires_grad = args.head_layer_trainable[1]
            if args.head_layer_trainable[1]:
                global_model_optimizer = torch.optim.Adam(list(global_model.head_layer.parameters()), lr=args.main_lr)

        global_model = global_model.to(args.device)

    return args, local_model, local_model_optimizer, global_model, global_model_optimizer


def load_basic_models_llm_gpt2(args, index):
    current_model_type = args.model_list[str(index)]['type']
    model_path = args.model_list[str(index)]['path']

    print('load_basic_models_llm from:', current_model_type)
    args.tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=True)
    args.tokenizer.padding_side = args.padding_side if (args.padding_side in ["left", "right"]) else "left"

    print('load gpt:', args.model_architect)
    if args.model_architect == 'CLM':  # task_type == 'CausalLM':
        print('gpt AutoModelForCausalLM')
        full_model = AutoModelForCausalLM.from_pretrained(model_path)
    # elif args.task_type == "Generation":
    #     full_model = AutoModelForCausalLM.from_pretrained(model_path)
    elif args.model_architect == 'TQA':  # .task_type == 'QuestionAnswering':
        print('gpt AutoModelForQuestionAnswering')
        full_model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    elif args.model_architect == 'CLS':  # .task_type == 'SequenceClassification':
        full_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    else:
        assert 1 > 2, "task type not supported"

    full_llm = full_model.transformer  # Active
    # full_qwen = full_model.model
    if args.model_architect == 'CLM':  # .task_type == 'CausalLM':
        head_layer = full_model.lm_head
    elif args.model_architect == 'TQA':#.task_type == 'QuestionAnswering':
        head_layer = full_model.qa_outputs
    elif args.model_architect == 'CLS':  # .task_type == 'SequenceClassification':
        head_layer = full_model.score
    else:
        head_layer = None

    args.config = full_model.config
    args.generation_config = full_model.generation_config
    print('gpt2 args.generation_config:', args.generation_config)
    args.model_architectures = args.config.architectures
    args.model_embedded_dim = args.config.n_embd

    all_encoder_num = args.config.num_hidden_layers
    print('all_encoder_num:', all_encoder_num)

    if args.pad_token == "default":
        print('Default pad')
        if args.tokenizer.pad_token is None:
            args.tokenizer.pad_token = args.tokenizer.eos_token  # ({'pad_token': '[PAD]'}) # args.tokenizer.eos_token #
            pad_id = args.tokenizer.convert_tokens_to_ids(args.tokenizer.eos_token)  #
            full_model.config.pad_token_id = pad_id
        args.pad_token = "default_" + args.tokenizer.pad_token
    else:
        args.tokenizer.pad_token = args.pad_token  # ({'pad_token': '[PAD]'}) # args.tokenizer.eos_token #
        pad_id = args.tokenizer.convert_tokens_to_ids(args.pad_token)  #
        full_model.config.pad_token_id = pad_id

    ########### Local Model ###########
    local_model = None
    local_model_optimizer = None
    if index < args.k - 1:
        #############
        print('args.local_encoders_num:',args.local_encoders_num)
        local_model = LocalGPT2Model(full_llm, args.local_encoders_num)
        print(f"local_model parameters: {sum(p.numel() for p in local_model.parameters())}")

        if args.finetune_name == "LoRA":
            # print('args.finetune_detail_configs:',args.finetune_detail_configs)
            if args.finetune_detail_configs != None:
                lora_config = LoraConfig(
                    **args.finetune_detail_configs
                )
            else:
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=4,
                    lora_alpha=32,
                    lora_dropout=0.1
                )

            def get_lora_model(model):
                model.enable_input_require_grads()
                peft_model = get_peft_model(model, lora_config)
                return peft_model

            local_model = get_lora_model(local_model)
            print('after lora')
            local_model.print_trainable_parameters()


        encoder_trainable_ids = args.encoder_trainable_ids_list[index]
        print('encoder_trainable_ids = ', encoder_trainable_ids)
        for encoder_id in range(len(local_model.h)):
            if encoder_id not in encoder_trainable_ids:
                for param in local_model.h.parameters():
                    param.requires_grad = False

        print('embedding_trainable = ', args.embedding_trainable[0])
        if args.embedding_trainable[0] == False:
            for param in local_model.wte.parameters():
                param.requires_grad = False
            for param in local_model.wpe.parameters():
                param.requires_grad = False

        if args.finetune_name == "LoRA":
            print('local final trainable param:')
            local_model.print_trainable_parameters()

        local_model = local_model.to(args.device)

        local_trainable_params = list(filter(lambda x: x.requires_grad, local_model.parameters()))
        if len(local_trainable_params)>0:
            local_model_optimizer = torch.optim.Adam(local_trainable_params, lr=args.main_lr)



        # local_model_optimizer = None
        # local_trainable_params = []
        # print('Local Model: embedding_trainable = ', args.embedding_trainable[0])
        # for param in local_model.wte.parameters():
        #     param.requires_grad = args.embedding_trainable[0]
        # if args.embedding_trainable[0]:
        #     local_trainable_params.extend(list(local_model.wte.parameters()))
        # for param in local_model.wpe.parameters():
        #     param.requires_grad = args.embedding_trainable[0]
        # if args.embedding_trainable[0]:
        #     local_trainable_params.extend(list(local_model.wpe.parameters()))
        # print('Local Model: args.encoder_trainable = ', args.encoder_trainable[0])
        # for param in local_model.h.parameters():
        #     param.requires_grad = args.encoder_trainable[0]
        # if args.encoder_trainable[0]:
        #     local_trainable_params.extend(list(local_model.h.parameters()))
        # if len(local_trainable_params)>0:
        #     local_model_optimizer = torch.optim.Adam(local_trainable_params, lr=args.main_lr)

    ########### Global Model ###########
    global_model = None
    global_model_optimizer = None
    if index == args.k - 1:
        global_encoders_num = all_encoder_num - args.local_encoders_num
        print('global_encoders_num:', global_encoders_num)

        # global part of gpt2(frozen)
        global_model = GlobalGPT2Model(full_llm, global_encoders_num)

        # add Classification Layer(untrainable)
        if args.model_architect == 'CLM':  # .task_type == "CausalLM":
            global_model = GPT2LMHeadModel_pretrained(global_model, head_layer,
                                                      generation_config=full_model.generation_config)
        elif args.model_architect == 'TQA':  # .task_type == "QuestionAnswering":
            global_model = GPT2ForQuestionAnswering_pretrained(global_model, head_layer)
        elif args.model_architect == 'CLS':  # .task_type == "SequenceClassification":
            global_model = GPT2ForSequenceClassification_pretrained(global_model, head_layer)
        # elif args.task_type == "Generation":
        #     global_model = GPT2LMHeadModel_pretrained(global_model, head_layer, generation_config=full_model.generation_config)
        else:
            assert 1 > 2, "task type not supported"

        print(f"global_model parameters: {sum(p.numel() for p in global_model.parameters())}")

        # Freeze Backbone
        for param in global_model.transformer.parameters():
            param.requires_grad = False

        # Head Layer Trainable/Freeze
        if head_layer:  # head layer exists
            print('Global Model : head_layer_trainable = ', args.head_layer_trainable[1])
            for param in global_model.head_layer.parameters():
                param.requires_grad = args.head_layer_trainable[1]
            if args.head_layer_trainable[1]:
                global_model_optimizer = torch.optim.Adam(list(global_model.head_layer.parameters()), lr=args.main_lr)

        global_model = global_model.to(args.device)

    del (full_model)
    return args, local_model, local_model_optimizer, global_model, global_model_optimizer


def load_basic_models_llm_llama(args, index):
    current_model_type = args.model_list[str(index)]['type']
    model_path = args.model_list[str(index)]['path']

    print('load_basic_models_llm from:', current_model_type)
    args.tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=True)
    args.tokenizer.padding_side = args.padding_side if (args.padding_side in ["left", "right"]) else "left"

    if args.model_architect == 'CLM':
        full_model = AutoModelForCausalLM.from_pretrained(model_path)
        full_llm = full_model.model
    elif args.model_architect == 'TQA':
        full_model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        full_llm = full_model.transformer
    elif args.model_architect == 'CLS':
        full_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        full_llm = full_model.model
    else:
        assert 1 > 2, "task type not supported"


    if args.model_architect == 'CLM':
        head_layer = full_model.lm_head
    elif args.model_architect == 'TQA':
        head_layer = full_model.qa_outputs
    elif args.model_architect == 'CLS':
        head_layer = full_model.score
    else:
        head_layer = None
        assert 1 > 2, "task type not supported"

    if args.pad_token == "default":
        if args.tokenizer.pad_token is None:
            args.tokenizer.pad_token = args.tokenizer.eos_token  # ({'pad_token': '[PAD]'}) # args.tokenizer.eos_token #
            pad_id = args.tokenizer.convert_tokens_to_ids(args.tokenizer.eos_token)  #
            full_model.config.pad_token_id = pad_id
        args.pad_token = "default_" + args.tokenizer.pad_token
    else:
        args.tokenizer.pad_token = args.pad_token  # ({'pad_token': '[PAD]'}) # args.tokenizer.eos_token #
        pad_id = args.tokenizer.convert_tokens_to_ids(args.pad_token)  #
        full_model.config.pad_token_id = pad_id

    args.config = full_model.config
    args.generation_config = full_model.generation_config
    args.model_architectures = args.config.architectures
    args.model_embedded_dim = args.config.hidden_size

    all_encoder_num = args.config.num_hidden_layers
    print('all_encoder_num:', all_encoder_num)

    ########### Local Model ###########
    local_model = None
    local_model_optimizer = None
    if index < args.k - 1:
        print('args.local_encoders_num:', args.local_encoders_num)
        local_model = LocalLlamaModel(full_llm, num_encoders=args.local_encoders_num)

        if args.finetune_name == "LoRA":
            # print('args.finetune_detail_configs:',args.finetune_detail_configs)
            if args.finetune_detail_configs != None:
                lora_config = LoraConfig(
                    **args.finetune_detail_configs
                )
            else:
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=4,
                    lora_alpha=32,
                    lora_dropout=0.1
                )

            def get_lora_model(model):
                model.enable_input_require_grads()
                peft_model = get_peft_model(model, lora_config)
                return peft_model

            local_model = get_lora_model(local_model)
            print('after lora')
            local_model.print_trainable_parameters()


        encoder_trainable_ids = args.encoder_trainable_ids_list[index]
        print('encoder_trainable_ids = ', encoder_trainable_ids)
        for encoder_id in range(len(local_model.layers)):
            if encoder_id not in encoder_trainable_ids:
                for param in local_model.layers.parameters():
                    param.requires_grad = False

        print('embedding_trainable = ', args.embedding_trainable[0])
        if args.embedding_trainable[0] == False:
            for param in local_model.embed_tokens.parameters():
                param.requires_grad = False

        if args.finetune_name == "LoRA":
            print('local final trainable param:')
            local_model.print_trainable_parameters()

        local_model = local_model.to(args.device)

        local_trainable_params = list(filter(lambda x: x.requires_grad, local_model.parameters()))
        if len(local_trainable_params)>0:
            local_model_optimizer = torch.optim.Adam(local_trainable_params, lr=args.main_lr)

    ########### Global Model ###########
    global_model = None
    global_model_optimizer = None
    if index == args.k - 1:
        global_encoders_num = all_encoder_num - args.local_encoders_num
        print('global_encoders_num:', global_encoders_num)

        # global part of llama(frozen)
        global_model = GlobalLlamaModel(full_llm, num_encoders=global_encoders_num)

        # add Classification Layer(untrainable)
        if args.model_architect == 'CLM':
            global_model = LlamaForCausalLM_pretrained(global_model, head_layer,generation_config=full_model.generation_config)
        elif args.model_architect == 'TQA':
            global_model = LlamaForQuestionAnswering_pretrained(global_model, head_layer,generation_config=full_model.generation_config)
        elif args.model_architect == 'CLS':
            global_model = LlamaForSequenceClassification_pretrained(global_model, head_layer)
        else:
            assert 1 > 2, "task type not supported"

        print(f"global_model parameters: {sum(p.numel() for p in global_model.parameters())}")

        # Freeze Backbone
        for param in global_model.model.parameters():
            param.requires_grad = False

        # Head Layer Trainable/Freeze
        if head_layer:  # head layer exists
            print('Global Model : head_layer_trainable = ', args.head_layer_trainable[1])
            for param in global_model.head_layer.parameters():
                param.requires_grad = args.head_layer_trainable[1]
            if args.head_layer_trainable[1]:
                global_model_optimizer = torch.optim.Adam(list(global_model.head_layer.parameters()), lr=args.main_lr)

        global_model = global_model.to(args.device)

    del (full_model)

    return args, local_model, local_model_optimizer, global_model, global_model_optimizer


def load_basic_models_llm_baichuan(args, index):
    current_model_type = args.model_list[str(index)]['type']
    model_path = args.model_list[str(index)]['path']

    print('load_basic_models_llm from:', current_model_type)
    args.tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=True, trust_remote_code=True)
    args.tokenizer.padding_side = args.padding_side if (args.padding_side in ["left", "right"]) else "left"

    if args.model_architect == 'CLM':
        full_model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True)
    else:
        assert 1 > 2, "task type not supported"

    full_llm = full_model.model

    if args.model_architect == 'CLM':
        head_layer = full_model.lm_head
    # elif args.model_architect == 'TQA':
    #     head_layer = full_model.qa_outputs
    # elif args.model_architect == 'CLS':
    #     head_layer = full_model.score
    else:
        head_layer = None
        assert 1 > 2, "task type not supported"

    if args.pad_token == "default":
        if args.tokenizer.pad_token is None:
            args.tokenizer.pad_token = args.tokenizer.eos_token  # ({'pad_token': '[PAD]'}) # args.tokenizer.eos_token #
            pad_id = args.tokenizer.convert_tokens_to_ids(args.tokenizer.eos_token)  #
            full_model.config.pad_token_id = pad_id
        args.pad_token = "default_" + args.tokenizer.pad_token
    else:
        args.tokenizer.pad_token = args.pad_token  # ({'pad_token': '[PAD]'}) # args.tokenizer.eos_token #
        pad_id = args.tokenizer.convert_tokens_to_ids(args.pad_token)  #
        full_model.config.pad_token_id = pad_id

    args.config = full_model.config
    args.generation_config = full_model.generation_config
    args.model_architectures = args.config.architectures
    args.model_embedded_dim = args.config.hidden_size
    all_encoder_num = args.config.num_hidden_layers
    print('all_encoder_num:', all_encoder_num)

    ########### Local Model ###########
    local_model = None
    local_model_optimizer = None
    if index < args.k - 1:
        print('args.local_encoders_num:',args.local_encoders_num)
        local_model = LocalBaichuanModel(full_llm, num_encoders = args.local_encoders_num)
        
        if args.finetune_name == "LoRA":
            # print('args.finetune_detail_configs:',args.finetune_detail_configs)
            if args.finetune_detail_configs != None:
                lora_config = LoraConfig(
                    **args.finetune_detail_configs
                )
            else:
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=4,
                    lora_alpha=32,
                    lora_dropout=0.1
                )

            def get_lora_model(model):
                model.enable_input_require_grads()
                peft_model = get_peft_model(model, lora_config)
                return peft_model

            local_model = get_lora_model(local_model)
            print('after lora')
            local_model.print_trainable_parameters()

        encoder_trainable_ids = args.encoder_trainable_ids_list[index]
        print('encoder_trainable_ids = ', encoder_trainable_ids)
        for encoder_id in range(len(local_model.layers)):
            if encoder_id not in encoder_trainable_ids:
                for param in local_model.layers.parameters():
                    param.requires_grad = False

        print('embedding_trainable = ', args.embedding_trainable[0])
        if args.embedding_trainable[0] == False:
            for param in local_model.embed_tokens.parameters():
                param.requires_grad = False

        if args.finetune_name == "LoRA":
            print('local final trainable param:')
            local_model.print_trainable_parameters()

        local_model = local_model.to(args.device)

        local_trainable_params = list(filter(lambda x: x.requires_grad, local_model.parameters()))
        if len(local_trainable_params)>0:
            local_model_optimizer = torch.optim.Adam(local_trainable_params, lr=args.main_lr)


    ########### Global Model ###########
    global_model = None
    global_model_optimizer = None
    if index == args.k - 1:
        global_encoders_num = all_encoder_num - args.local_encoders_num
        print('global_encoders_num:', global_encoders_num)

        # global part of llama(frozen)
        global_model = GlobalBaichuanModel(full_llm, num_encoders=global_encoders_num)

        # add Classification Layer(untrainable)
        if args.model_architect == 'CLM':
            global_model = BaiChuanForCausalLM_pretrained(global_model, head_layer,generation_config=full_model.generation_config)
        else:
            assert 1 > 2, "task type not supported"

        print(f"global_model parameters: {sum(p.numel() for p in global_model.parameters())}")

        # Freeze Backbone
        for param in global_model.model.parameters():
            param.requires_grad = False

        # Head Layer Trainable/Freeze
        if head_layer:  # head layer exists
            print('Global Model : head_layer_trainable = ', args.head_layer_trainable[1])
            for param in global_model.head_layer.parameters():
                param.requires_grad = args.head_layer_trainable[1]
            if args.head_layer_trainable[1]:
                global_model_optimizer = torch.optim.Adam(list(global_model.head_layer.parameters()), lr=args.main_lr)

        global_model = global_model.to(args.device)

    del (full_model)

    return args, local_model, local_model_optimizer, global_model, global_model_optimizer


def load_basic_models_llm_xlnet(args, index):
    current_model_type = args.model_list[str(index)]['type']
    model_path = args.model_list[str(index)]['path']

    print('load_basic_models_llm from:', current_model_type)
    args.tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=True)
    args.tokenizer.padding_side = args.padding_side if (args.padding_side in ["left", "right"]) else "left"

    if args.model_architect == 'CLM':
        full_model = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        assert 1 > 2, "task type not supported"

    full_llm = full_model.transformer

    if args.model_architect == 'CLM':
        head_layer = full_model.lm_loss
    else:
        head_layer = None
        assert 1 > 2, "task type not supported"

    if args.pad_token == "default":
        if args.tokenizer.pad_token is None:
            args.tokenizer.pad_token = args.tokenizer.eos_token  # ({'pad_token': '[PAD]'}) # args.tokenizer.eos_token #
            pad_id = args.tokenizer.convert_tokens_to_ids(args.tokenizer.eos_token)  #
            full_model.config.pad_token_id = pad_id
        args.pad_token = "default_" + args.tokenizer.pad_token
    else:
        args.tokenizer.pad_token = args.pad_token  # ({'pad_token': '[PAD]'}) # args.tokenizer.eos_token #
        pad_id = args.tokenizer.convert_tokens_to_ids(args.pad_token)  #
        full_model.config.pad_token_id = pad_id

    args.config = full_model.config
    args.generation_config = full_model.generation_config
    args.model_architectures = args.config.architectures
    args.model_embedded_dim = args.config.d_model

    all_encoder_num = args.config.n_layer
    print('all_encoder_num:', all_encoder_num)

    ########### Local Model ###########
    local_model = None
    local_model_optimizer = None
    if index < args.k - 1:
        print('args.local_encoders_num:',args.local_encoders_num)
        local_model = LocalXLNetModel(full_llm, num_encoders = args.local_encoders_num)

        if args.finetune_name == "LoRA":
            # print('args.finetune_detail_configs:',args.finetune_detail_configs)
            if args.finetune_detail_configs != None:
                lora_config = LoraConfig(
                    **args.finetune_detail_configs
                )
            else:
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=4,
                    lora_alpha=32,
                    lora_dropout=0.1
                )

            def get_lora_model(model):
                model.enable_input_require_grads()
                peft_model = get_peft_model(model, lora_config)
                return peft_model

            local_model = get_lora_model(local_model)
            print('after lora')
            local_model.print_trainable_parameters()

        encoder_trainable_ids = args.encoder_trainable_ids_list[index]
        print('encoder_trainable_ids = ', encoder_trainable_ids)
        for encoder_id in range(len(local_model.layer)):
            if encoder_id not in encoder_trainable_ids:
                for param in local_model.layer.parameters():
                    param.requires_grad = False

        print('embedding_trainable = ', args.embedding_trainable[0])
        if args.embedding_trainable[0] == False:
            for param in local_model.word_embedding.parameters():
                param.requires_grad = False

        if args.finetune_name == "LoRA":
            print('local final trainable param:')
            local_model.print_trainable_parameters()

        local_model = local_model.to(args.device)

        local_trainable_params = list(filter(lambda x: x.requires_grad, local_model.parameters()))
        if len(local_trainable_params)>0:
            local_model_optimizer = torch.optim.Adam(local_trainable_params, lr=args.main_lr)


    ########### Global Model ###########
    global_model = None
    global_model_optimizer = None
    if index == args.k - 1:
        global_encoders_num = all_encoder_num - args.local_encoders_num
        print('global_encoders_num:', global_encoders_num)

        # global part of llama(frozen)
        global_model = GlobalXLNetModel(full_llm, num_encoders=global_encoders_num)

        # add Classification Layer(untrainable)
        if args.model_architect == 'CLM':
            global_model = XLNetLMHeadModel_pretrained(global_model, head_layer,generation_config=full_model.generation_config)
        # elif args.model_architect == 'CLS':
        #     global_model = LlamaForSequenceClassification_pretrained(global_llama, head_layer)
        else:
            assert 1 > 2, "task type not supported"

        print(f"global_model parameters: {sum(p.numel() for p in global_model.parameters())}")

        # Freeze Backbone
        for param in global_model.transformer.parameters():
            param.requires_grad = False

        # Head Layer Trainable/Freeze
        if head_layer:  # head layer exists
            print('Global Model : head_layer_trainable = ', args.head_layer_trainable[1])
            for param in global_model.head_layer.parameters():
                param.requires_grad = args.head_layer_trainable[1]
            if args.head_layer_trainable[1]:
                global_model_optimizer = torch.optim.Adam(list(global_model.head_layer.parameters()), lr=args.main_lr)

        global_model = global_model.to(args.device)

    del (full_model)

    return args, local_model, local_model_optimizer, global_model, global_model_optimizer


def load_basic_models_llm_falcon(args, index):
    current_model_type = args.model_list[str(index)]['type']
    model_path = args.model_list[str(index)]['path']

    print('load_basic_models_llm from:', current_model_type)
    args.tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=True)
    args.tokenizer.padding_side = args.padding_side if (args.padding_side in ["left", "right"]) else "left"

    if args.model_architect == 'CLM':
        full_model = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        assert 1 > 2, "task type not supported"

    full_llm = full_model.transformer

    if args.model_architect == 'CLM':
        head_layer = full_model.lm_head
    else:
        head_layer = None
        assert 1 > 2, "task type not supported"

    if args.tokenizer.pad_token is None:
        args.tokenizer.pad_token = args.tokenizer.eos_token  # ({'pad_token': '[PAD]'}) # args.tokenizer.eos_token #
        pad_id = args.tokenizer.convert_tokens_to_ids(args.tokenizer.eos_token)  #
        full_model.config.pad_token_id = int(pad_id)
        args.tokenizer.pad_token_id = int(pad_id)

    args.config = full_model.config
    args.generation_config = full_model.generation_config
    args.model_architectures = args.config.architectures
    args.model_embedded_dim = args.config.hidden_size

    all_encoder_num = args.config.num_hidden_layers
    print('all_encoder_num:', all_encoder_num)

    ########### Local Model ###########
    local_model = None
    local_model_optimizer = None
    if index < args.k - 1:
        print('args.local_encoders_num:', args.local_encoders_num)
        local_model = LocalFalconModel(full_llm, num_encoders=args.local_encoders_num)

        if args.finetune_name == "LoRA":
            # print('args.finetune_detail_configs:',args.finetune_detail_configs)
            if args.finetune_detail_configs != None:
                lora_config = LoraConfig(
                    **args.finetune_detail_configs
                )
            else:
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=4,
                    lora_alpha=32,
                    lora_dropout=0.1
                )

            def get_lora_model(model):
                model.enable_input_require_grads()
                peft_model = get_peft_model(model, lora_config)
                return peft_model

            local_model = get_lora_model(local_model)
            print('after lora')
            local_model.print_trainable_parameters()

        encoder_trainable_ids = args.encoder_trainable_ids_list[index]
        print('encoder_trainable_ids = ', encoder_trainable_ids)
        for encoder_id in range(len(local_model.h)):
            if encoder_id not in encoder_trainable_ids:
                for param in local_model.h.parameters():
                    param.requires_grad = False

        print('embedding_trainable = ', args.embedding_trainable[0])
        if args.embedding_trainable[0] == False:
            for param in local_model.word_embeddings.parameters():
                param.requires_grad = False

        if args.finetune_name == "LoRA":
            print('local final trainable param:')
            local_model.print_trainable_parameters()

        local_model = local_model.to(args.device)

        local_trainable_params = list(filter(lambda x: x.requires_grad, local_model.parameters()))
        if len(local_trainable_params)>0:
            local_model_optimizer = torch.optim.Adam(local_trainable_params, lr=args.main_lr)

    ########### Global Model ###########
    global_model = None
    global_model_optimizer = None
    if index == args.k - 1:
        global_encoders_num = all_encoder_num - args.local_encoders_num
        print('global_encoders_num:', global_encoders_num)

        # global part of llama(frozen)
        global_model = GlobalFalconModel(full_llm, num_encoders=global_encoders_num)

        # add Classification Layer(untrainable)
        if args.model_architect == 'CLM':
            global_model = FalconForCausalLM_pretrained(global_model, head_layer,generation_config=full_model.generation_config)
        # elif args.model_architect == 'CLS':
        #     global_model = FalconForCausalLM_pretrained(global_model, head_layer)
        else:
            assert 1 > 2, "task type not supported"

        print(f"global_model parameters: {sum(p.numel() for p in global_model.parameters())}")

        # Freeze Backbone
        for param in global_model.transformer.parameters():
            param.requires_grad = False

        # Head Layer Trainable/Freeze
        if head_layer:  # head layer exists
            print('Global Model : head_layer_trainable = ', args.head_layer_trainable[1])
            for param in global_model.head_layer.parameters():
                param.requires_grad = args.head_layer_trainable[1]
            if args.head_layer_trainable[1]:
                global_model_optimizer = torch.optim.Adam(list(global_model.head_layer.parameters()), lr=args.main_lr)

        global_model = global_model.to(args.device)

    del (full_model)

    return args, local_model, local_model_optimizer, global_model, global_model_optimizer


# def load_basic_models_llm_qwen2(args, index):
#     model_path = args.model_list[str(index)]['path']
#     args.tokenizer = AutoTokenizer.from_pretrained(model_path)
#     local_model, global_model = None, None
#     if index == 0:
#
#         local_model = PipelineVFL2Slice(is_server_end=False).from_vfl(model_path,
#                                                                       device_map='auto',
#                                                                       torch_dtype=torch.bfloat16)  # type:Qwen2ModelHead
#         logger.debug(f"model: {type(local_model)}\ndevice_map: {local_model.hf_device_map}")
#
#     if index == 1:
#         global_model = PipelineVFL2Slice(is_server_end=True).from_vfl(model_path,
#                                                                       device_map='auto',
#                                                                       torch_dtype=torch.bfloat16)  # type:Qwen2TailForCausalLM
#         logger.debug(f"model: {type(global_model)}\ndevice_map: {global_model.hf_device_map}")
#     local_model_optimizer, global_model_optimizer = None, None
#     return args, local_model, local_model_optimizer, global_model, global_model_optimizer


def load_basic_models_llm_mamba(args, index):
    current_model_type = args.model_list[str(index)]['type']
    model_path = args.model_list[str(index)]['path']

    print('load_basic_models_llm from:', current_model_type)
    args.tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=True)
    args.tokenizer.padding_side = args.padding_side if (args.padding_side in ["left", "right"]) else "left"

    if args.model_architect == 'CLM':
        full_model = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        assert 1 > 2, "task type not supported"

    full_llm = full_model.backbone

    if args.model_architect == 'CLM':
        head_layer = full_model.lm_head
    else:
        head_layer = None
        assert 1 > 2, "task type not supported"

    if args.pad_token == "default":
        if args.tokenizer.pad_token is None:
            args.tokenizer.pad_token = args.tokenizer.eos_token  # ({'pad_token': '[PAD]'}) # args.tokenizer.eos_token #
            pad_id = args.tokenizer.convert_tokens_to_ids(args.tokenizer.eos_token)  #
            full_model.config.pad_token_id = pad_id
        args.pad_token = "default_" + args.tokenizer.pad_token
    else:
        args.tokenizer.pad_token = args.pad_token  # ({'pad_token': '[PAD]'}) # args.tokenizer.eos_token #
        pad_id = args.tokenizer.convert_tokens_to_ids(args.pad_token)  #
        full_model.config.pad_token_id = pad_id

    args.config = full_model.config
    args.generation_config = full_model.generation_config
    args.model_architectures = args.config.architectures
    args.model_embedded_dim = args.config.hidden_size

    all_encoder_num = args.config.num_hidden_layers
    print('all_encoder_num:', all_encoder_num)

    ########### Local Model ###########
    local_model = None
    local_model_optimizer = None
    if index < args.k - 1:
        print('args.local_encoders_num:', args.local_encoders_num)
        local_model = LocalMambaModel(full_llm, num_encoders=args.local_encoders_num)

        if args.finetune_name == "LoRA":
            # print('args.finetune_detail_configs:',args.finetune_detail_configs)
            if args.finetune_detail_configs != None:
                lora_config = LoraConfig(
                    **args.finetune_detail_configs
                )
            else:
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=4,
                    lora_alpha=32,
                    lora_dropout=0.1
                )

            def get_lora_model(model):
                model.enable_input_require_grads()
                peft_model = get_peft_model(model, lora_config)
                return peft_model

            local_model = get_lora_model(local_model)
            print('after lora')
            local_model.print_trainable_parameters()

        encoder_trainable_ids = args.encoder_trainable_ids_list[index]
        print('encoder_trainable_ids = ', encoder_trainable_ids)
        for encoder_id in range(len(local_model.layers)):
            if encoder_id not in encoder_trainable_ids:
                for param in local_model.layers.parameters():
                    param.requires_grad = False

        print('embedding_trainable = ', args.embedding_trainable[0])
        if args.embedding_trainable[0] == False:
            for param in local_model.embeddings.parameters():
                param.requires_grad = False

        if args.finetune_name == "LoRA":
            print('local final trainable param:')
            local_model.print_trainable_parameters()

        local_model = local_model.to(args.device)

        local_trainable_params = list(filter(lambda x: x.requires_grad, local_model.parameters()))
        if len(local_trainable_params)>0:
            local_model_optimizer = torch.optim.Adam(local_trainable_params, lr=args.main_lr)

    ########### Global Model ###########
    global_model = None
    global_model_optimizer = None
    if index == args.k - 1:
        global_encoders_num = all_encoder_num - args.local_encoders_num
        print('global_encoders_num:', global_encoders_num)

        # global part of llama(frozen)
        global_model = GlobalMambaModel(full_llm, num_encoders=global_encoders_num)

        # add Classification Layer(untrainable)
        if args.model_architect == 'CLM':
            global_model = MambaForCausalLM_pretrained(global_model, head_layer,generation_config=full_model.generation_config)
        # elif args.model_architect == 'CLS':
        #     global_model = MambaForCausalLM_pretrained(global_model, head_layer)
        else:
            assert 1 > 2, "task type not supported"

        print(f"global_model parameters: {sum(p.numel() for p in global_model.parameters())}")

        # Freeze Backbone
        for param in global_model.backbone.parameters():
            param.requires_grad = False

        # Head Layer Trainable/Freeze
        if head_layer:  # head layer exists
            print('Global Model : head_layer_trainable = ', args.head_layer_trainable[1])
            for param in global_model.head_layer.parameters():
                param.requires_grad = args.head_layer_trainable[1]
            if args.head_layer_trainable[1]:
                global_model_optimizer = torch.optim.Adam(list(global_model.head_layer.parameters()), lr=args.main_lr)

        global_model = global_model.to(args.device)

    del (full_model)

    return args, local_model, local_model_optimizer, global_model, global_model_optimizer


def load_basic_models_llm_gemma(args, index):
    current_model_type = args.model_list[str(index)]['type']
    model_path = args.model_list[str(index)]['path']

    print('load_basic_models_llm from:', current_model_type)
    args.tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=True)
    args.tokenizer.padding_side = args.padding_side if (args.padding_side in ["left", "right"]) else "left"

    if args.model_architect == 'CLM':
        full_model = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        assert 1 > 2, "task type not supported"

    full_llm = full_model.model

    if args.model_architect == 'CLM':
        head_layer = full_model.lm_head
    else:
        head_layer = None
        assert 1 > 2, "task type not supported"

    if args.pad_token == "default":
        if args.tokenizer.pad_token is None:
            args.tokenizer.pad_token = args.tokenizer.eos_token  # ({'pad_token': '[PAD]'}) # args.tokenizer.eos_token #
            pad_id = args.tokenizer.convert_tokens_to_ids(args.tokenizer.eos_token)  #
            full_model.config.pad_token_id = pad_id
        args.pad_token = "default_" + args.tokenizer.pad_token
    else:
        args.tokenizer.pad_token = args.pad_token  # ({'pad_token': '[PAD]'}) # args.tokenizer.eos_token #
        pad_id = args.tokenizer.convert_tokens_to_ids(args.pad_token)  #
        full_model.config.pad_token_id = pad_id

    args.config = full_model.config
    args.generation_config = full_model.generation_config
    args.model_architectures = args.config.architectures
    args.model_embedded_dim = args.config.hidden_size

    all_encoder_num = args.config.num_hidden_layers
    print('all_encoder_num:', all_encoder_num)

    ########### Local Model ###########
    local_model = None
    local_model_optimizer = None
    if index < args.k - 1:
        print('args.local_encoders_num:', args.local_encoders_num)
        local_model = LocalGemmaModel(full_llm, num_encoders=args.local_encoders_num)

        if args.finetune_name == "LoRA":
            # print('args.finetune_detail_configs:',args.finetune_detail_configs)
            if args.finetune_detail_configs != None:
                lora_config = LoraConfig(
                    **args.finetune_detail_configs
                )
            else:
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=4,
                    lora_alpha=32,
                    lora_dropout=0.1
                )

            def get_lora_model(model):
                model.enable_input_require_grads()
                peft_model = get_peft_model(model, lora_config)
                return peft_model

            local_model = get_lora_model(local_model)
            print('after lora')
            local_model.print_trainable_parameters()

        encoder_trainable_ids = args.encoder_trainable_ids_list[index]
        print('encoder_trainable_ids = ', encoder_trainable_ids)
        for encoder_id in range(len(local_model.layers)):
            if encoder_id not in encoder_trainable_ids:
                for param in local_model.layers.parameters():
                    param.requires_grad = False

        print('embedding_trainable = ', args.embedding_trainable[0])
        if args.embedding_trainable[0] == False:
            for param in local_model.embed_tokens.parameters():
                param.requires_grad = False

        if args.finetune_name == "LoRA":
            print('local final trainable param:')
            local_model.print_trainable_parameters()

        local_model = local_model.to(args.device)

        local_trainable_params = list(filter(lambda x: x.requires_grad, local_model.parameters()))
        if len(local_trainable_params)>0:
            local_model_optimizer = torch.optim.Adam(local_trainable_params, lr=args.main_lr)


    ########### Global Model ###########
    global_model = None
    global_model_optimizer = None
    if index == args.k - 1:
        global_encoders_num = all_encoder_num - args.local_encoders_num
        print('global_encoders_num:', global_encoders_num)

        # global part of llama(frozen)
        global_model = GlobalGemmaModel(full_llm, num_encoders=global_encoders_num)

        # add Classification Layer(untrainable)
        if args.model_architect == 'CLM':
            global_model = GemmaForCausalLM_pretrained(global_model, head_layer,generation_config=full_model.generation_config)
        # elif args.model_architect == 'CLS':
        #     global_model = MambaForCausalLM_pretrained(global_model, head_layer)
        else:
            assert 1 > 2, "task type not supported"

        print(f"global_model parameters: {sum(p.numel() for p in global_model.parameters())}")

        # Freeze Backbone
        for param in global_model.model.parameters():
            param.requires_grad = False

        # Head Layer Trainable/Freeze
        if head_layer:  # head layer exists
            print('Global Model : head_layer_trainable = ', args.head_layer_trainable[1])
            for param in global_model.head_layer.parameters():
                param.requires_grad = args.head_layer_trainable[1]
            if args.head_layer_trainable[1]:
                global_model_optimizer = torch.optim.Adam(list(global_model.head_layer.parameters()), lr=args.main_lr)

        global_model = global_model.to(args.device)

    del (full_model)

    return args, local_model, local_model_optimizer, global_model, global_model_optimizer


def load_basic_models_llm_chatglm(args, index):
    current_model_type = args.model_list[str(index)]['type']
    model_path = args.model_list[str(index)]['path']

    print('load_basic_models_llm from:', current_model_type)
    args.tokenizer = ChatGLMTokenizer.from_pretrained(model_path, do_lower_case=True, trust_remote_code=True)
    args.tokenizer.padding_side = args.padding_side if (args.padding_side in ["left", "right"]) else "left"

    if args.model_architect == 'CLM':
        full_model = ChatGLMForConditionalGeneration.from_pretrained(model_path, trust_remote_code=True)
    else:
        assert 1 > 2, "task type not supported"

    full_llm = full_model.transformer
    head_layer = full_model.transformer.output_layer

    all_encoder_num = full_model.config.num_layers

    if args.model_architect == 'CLM':
        head_layer = full_model.transformer.output_layer
    else:
        head_layer = None
        assert 1 > 2, "task type not supported"

    if args.pad_token == "default":
        if args.tokenizer.pad_token is None:
            args.tokenizer.pad_token = args.tokenizer.eos_token  # ({'pad_token': '[PAD]'}) # args.tokenizer.eos_token #
            pad_id = args.tokenizer.convert_tokens_to_ids(args.tokenizer.eos_token)  #
            full_model.config.pad_token_id = pad_id
        args.pad_token = "default_" + args.tokenizer.pad_token
    else:
        args.tokenizer.pad_token = args.pad_token  # ({'pad_token': '[PAD]'}) # args.tokenizer.eos_token #
        pad_id = args.tokenizer.convert_tokens_to_ids(args.pad_token)  #
        full_model.config.pad_token_id = pad_id

    args.config = full_model.config
    args.generation_config = full_model.generation_config
    args.model_architectures = args.config.architectures
    args.model_embedded_dim = args.config.hidden_size

    all_encoder_num = args.config.num_layers
    print('all_encoder_num:', all_encoder_num)

    ########### Local Model ###########
    local_model = None
    local_model_optimizer = None
    if index < args.k - 1:
        print('args.local_encoders_num:',args.local_encoders_num)
        local_model = LocalChatGLMModel(full_llm, num_encoders = args.local_encoders_num)
        
        del (full_llm)
        del (full_model)

        if args.finetune_name == "LoRA":
            # print('args.finetune_detail_configs:',args.finetune_detail_configs)
            if args.finetune_detail_configs != None:
                lora_config = LoraConfig(
                    **args.finetune_detail_configs
                )
            else:
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=4,
                    lora_alpha=32,
                    lora_dropout=0.1
                )

            def get_lora_model(model):
                model.enable_input_require_grads()
                peft_model = get_peft_model(model, lora_config)
                return peft_model

            local_model = get_lora_model(local_model)
            print('after lora')
            local_model.print_trainable_parameters()

        encoder_trainable_ids = args.encoder_trainable_ids_list[index]
        print('encoder_trainable_ids = ', encoder_trainable_ids)
        for encoder_id in range(len(local_model.encoder.layers)):
            if encoder_id not in encoder_trainable_ids:
                for param in local_model.encoder.layers.parameters():
                    param.requires_grad = False

        print('embedding_trainable = ', args.embedding_trainable[0])
        if args.embedding_trainable[0] == False:
            for param in local_model.embedding.parameters():
                param.requires_grad = False

        if args.finetune_name == "LoRA":
            print('local final trainable param:')
            local_model.print_trainable_parameters()

        local_model = local_model.to(args.device)

        local_trainable_params = list(filter(lambda x: x.requires_grad, local_model.parameters()))
        if len(local_trainable_params)>0:
            local_model_optimizer = torch.optim.Adam(local_trainable_params, lr=args.main_lr)

    ########### Global Model ###########
    global_model = None
    global_model_optimizer = None
    if index == args.k - 1:
        global_encoders_num = all_encoder_num - args.local_encoders_num
        print('global_encoders_num:', global_encoders_num)

        # global part of llama(frozen)
        global_model = GlobalChatGLMModel(full_llm, num_encoders=global_encoders_num)

        # add Classification Layer(untrainable)
        if args.model_architect == 'CLM':
            global_model = ChatGLMForConditionalGeneration_pretrained(global_model, head_layer,generation_config=full_model.generation_config)
        # elif args.model_architect == 'CLS':
        #     global_model = MambaForCausalLM_pretrained(global_model, head_layer)
        else:
            assert 1 > 2, "task type not supported"

        print(f"global_model parameters: {sum(p.numel() for p in global_model.parameters())}")

        # Freeze Backbone
        for param in global_model.transformer.parameters():
            param.requires_grad = False

        # Head Layer Trainable/Freeze
        if head_layer:  # head layer exists
            print('Global Model : head_layer_trainable = ', args.head_layer_trainable[1])
            for param in global_model.head_layer.parameters():
                param.requires_grad = args.head_layer_trainable[1]
            if args.head_layer_trainable[1]:
                global_model_optimizer = torch.optim.Adam(list(global_model.head_layer.parameters()), lr=args.main_lr)

        del (full_llm)
        del (full_model)
        global_model = global_model.to(args.device)



    return args, local_model, local_model_optimizer, global_model, global_model_optimizer


def load_basic_models_llm_mistral(args, index):
    current_model_type = args.model_list[str(index)]['type']
    model_path = args.model_list[str(index)]['path']

    print('load_basic_models_llm from:', current_model_type)
    args.tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=True)
    args.tokenizer.padding_side = args.padding_side if (args.padding_side in ["left", "right"]) else "left"

    if args.model_architect == 'CLM':
        full_model = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        assert 1 > 2, "task type not supported"

    full_llm = full_model.model

    if args.model_architect == 'CLM':
        head_layer = full_model.lm_head
    else:
        head_layer = None
        assert 1 > 2, "task type not supported"

    if args.pad_token == "default":
        if args.tokenizer.pad_token is None:
            args.tokenizer.pad_token = args.tokenizer.eos_token  # ({'pad_token': '[PAD]'}) # args.tokenizer.eos_token #
            pad_id = args.tokenizer.convert_tokens_to_ids(args.tokenizer.eos_token)  #
            full_model.config.pad_token_id = pad_id
        args.pad_token = "default_" + args.tokenizer.pad_token
    else:
        args.tokenizer.pad_token = args.pad_token  # ({'pad_token': '[PAD]'}) # args.tokenizer.eos_token #
        pad_id = args.tokenizer.convert_tokens_to_ids(args.pad_token)  #
        full_model.config.pad_token_id = pad_id

    args.config = full_model.config
    args.generation_config = full_model.generation_config
    args.model_architectures = args.config.architectures
    args.model_embedded_dim = args.config.hidden_size

    all_encoder_num = args.config.num_hidden_layers
    print('all_encoder_num:', all_encoder_num)

    ########### Local Model ###########
    local_model = None
    local_model_optimizer = None
    if index < args.k - 1:
        print('args.local_encoders_num:',args.local_encoders_num)
        local_model = LocalMistralModel(full_llm, num_encoders = args.local_encoders_num)
        
        if args.finetune_name == "LoRA":
            # print('args.finetune_detail_configs:',args.finetune_detail_configs)
            if args.finetune_detail_configs != None:
                lora_config = LoraConfig(
                    **args.finetune_detail_configs
                )
            else:
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=4,
                    lora_alpha=32,
                    lora_dropout=0.1
                )

            def get_lora_model(model):
                model.enable_input_require_grads()
                peft_model = get_peft_model(model, lora_config)
                return peft_model

            local_model = get_lora_model(local_model)
            print('after lora')
            local_model.print_trainable_parameters()

        encoder_trainable_ids = args.encoder_trainable_ids_list[index]
        print('encoder_trainable_ids = ', encoder_trainable_ids)
        for encoder_id in range(len(local_model.layers)):
            if encoder_id not in encoder_trainable_ids:
                for param in local_model.layers.parameters():
                    param.requires_grad = False

        print('embedding_trainable = ', args.embedding_trainable[0])
        if args.embedding_trainable[0] == False:
            for param in local_model.embed_tokens.parameters():
                param.requires_grad = False

        if args.finetune_name == "LoRA":
            print('local final trainable param:')
            local_model.print_trainable_parameters()

        local_model = local_model.to(args.device)

        local_trainable_params = list(filter(lambda x: x.requires_grad, local_model.parameters()))
        if len(local_trainable_params)>0:
            local_model_optimizer = torch.optim.Adam(local_trainable_params, lr=args.main_lr)


    ########### Global Model ###########
    global_model = None
    global_model_optimizer = None
    if index == args.k - 1:
        global_encoders_num = all_encoder_num - args.local_encoders_num
        print('global_encoders_num:', global_encoders_num)

        # global part of llama(frozen)
        global_model = GlobalMistralModel(full_llm, num_encoders=global_encoders_num)

        # add Classification Layer(untrainable)
        if args.model_architect == 'CLM':
            global_model = MistralForCausalLM_pretrained(global_model, head_layer,generation_config=full_model.generation_config)
        else:
            assert 1 > 2, "task type not supported"

        print(f"global_model parameters: {sum(p.numel() for p in global_model.parameters())}")

        # Freeze Backbone
        for param in global_model.model.parameters():
            param.requires_grad = False

        # Head Layer Trainable/Freeze
        if head_layer:  # head layer exists
            print('Global Model : head_layer_trainable = ', args.head_layer_trainable[1])
            for param in global_model.head_layer.parameters():
                param.requires_grad = args.head_layer_trainable[1]
            if args.head_layer_trainable[1]:
                global_model_optimizer = torch.optim.Adam(list(global_model.head_layer.parameters()), lr=args.main_lr)

        global_model = global_model.to(args.device)

    del (full_model)

    return args, local_model, local_model_optimizer, global_model, global_model_optimizer


def load_basic_models_llm(args, index):
    if args.model_type in ['Bert', 'Albert', 'Roberta']:
        args, local_model, local_model_optimizer, global_model, global_model_optimizer = load_basic_models_llm_bert(
            args, index)
    elif args.model_type in ['GPT2']:
        args, local_model, local_model_optimizer, global_model, global_model_optimizer = load_basic_models_llm_gpt2(
            args, index)
    elif args.model_type in ['Llama']:
        args, local_model, local_model_optimizer, global_model, global_model_optimizer = load_basic_models_llm_llama(
            args, index)
    elif args.model_type in ['Mamba']:
        args, local_model, local_model_optimizer, global_model, global_model_optimizer = load_basic_models_llm_mamba(
            args, index)
    elif args.model_type in ['Falcon']:
        args, local_model, local_model_optimizer, global_model, global_model_optimizer = load_basic_models_llm_falcon(
            args, index)
    elif args.model_type in ['Baichuan']:
        args, local_model, local_model_optimizer, global_model, global_model_optimizer = load_basic_models_llm_baichuan(
            args, index)
    elif args.model_type in ['XLNet']:
        args, local_model, local_model_optimizer, global_model, global_model_optimizer = load_basic_models_llm_xlnet(
            args, index)
    elif args.model_type in ['Gemma']:
        args, local_model, local_model_optimizer, global_model, global_model_optimizer = load_basic_models_llm_gemma(
            args, index)
    elif args.model_type in ['ChatGLM']:
        args, local_model, local_model_optimizer, global_model, global_model_optimizer = load_basic_models_llm_chatglm(
            args, index)
    elif args.model_type in ['Mistral']:
        args, local_model, local_model_optimizer, global_model, global_model_optimizer = load_basic_models_llm_mistral(
            args, index)

    elif args.model_type in ['T5']:
        args, local_model, local_model_optimizer, global_model, global_model_optimizer = load_basic_models_llm_t5(args,
                                                                                                                  index)
    else:
        assert 1 > 2, f'{args.model_type} not supported'

    # if 'questionanswering' in args.model_architectures[0].lower():
    #     args.model_architect = 'TQA' # Text-span based Question Answering
    # elif 'classification' in args.model_architectures[0].lower():
    #     args.model_architect = 'CLS' # Classification
    # else:
    #     args.model_architect = 'CLM' # Causal LM

    print(f'Model Architect:{args.model_architectures[0]}  {args.model_architect}')
    return args, local_model, local_model_optimizer, global_model, global_model_optimizer

# def load_basic_models_llm_new(pretrained, task_type, model_type, current_output_dim, is_local, device, padding_side, model_path, main_lr, pad_token,
#                               head_layer_trainable):
#     if model_type in ['Bert', 'Albert', 'Roberta']:
#         local_model, local_model_optimizer, global_model, global_model_optimizer, tokenizer = load_basic_models_llm_bert_new(
#             pretrained, task_type, model_type, current_output_dim, is_local, device, padding_side, model_path, main_lr, pad_token, head_layer_trainable
#         )
#     elif model_type in ['GPT2']:
#         # args, local_model, local_model_optimizer, global_model, global_model_optimizer = load_basic_models_llm_gpt2(args,index)
#         pass
#     elif model_type in ['Llama']:
#         # args, local_model, local_model_optimizer, global_model, global_model_optimizer = load_basic_models_llm_llama(args,index)
#         pass
#     else:
#         assert 1 > 2, 'llm not supported'
#     return local_model, local_model_optimizer, global_model, global_model_optimizer, tokenizer


# def load_basic_models_llm_new(pretrained, task_type, model_type, current_output_dim, is_local, device, padding_side, model_path, main_lr, pad_token,
#                               head_layer_trainable):
#     if model_type in ['Bert', 'Albert', 'Roberta']:
#         local_model, local_model_optimizer, global_model, global_model_optimizer, tokenizer = load_basic_models_llm_bert_new(
#             pretrained, task_type, model_type, current_output_dim, is_local, device, padding_side, model_path, main_lr, pad_token, head_layer_trainable
#         )
#     elif model_type in ['GPT2']:
#         # args, local_model, local_model_optimizer, global_model, global_model_optimizer = load_basic_models_llm_gpt2(args,index)
#         pass
#     elif model_type in ['Llama']:
#         # args, local_model, local_model_optimizer, global_model, global_model_optimizer = load_basic_models_llm_llama(args,index)
#         pass
#     else:
#         assert 1 > 2, 'llm not supported'
#     return local_model, local_model_optimizer, global_model, global_model_optimizer, tokenizer

# def load_models_per_party_new(pretrained, task_type, model_type, current_model_type, current_output_dim, is_local, device, padding_side, model_path, main_lr,
#                               pad_token, head_layer_trainable):
#     if current_model_type in LLM_supported:
#         local_model, local_model_optimizer, global_model, global_model_optimizer, tokenizer = load_basic_models_llm_new(
#             pretrained, task_type, model_type, current_output_dim, is_local, device, padding_side, model_path, main_lr, pad_token, head_layer_trainable
#         )
#         # args, local_model, local_model_optimizer, global_model, global_model_optimizer =
#         #  load_defense_models(args, index, local_model, local_model_optimizer, global_model, global_model_optimizer)

#     encoder = None
#     return local_model, local_model_optimizer, global_model, global_model_optimizer, tokenizer, encoder

# def load_basic_models_llm_bert(args, index):
#     current_model_type = args.model_list[str(index)]['type']
#     current_output_dim = args.model_list[str(index)]['output_dim']
#     task_type = args.task_type
#     model_type = args.model_type
#     pretrained = args.pretrained
#     device = args.device
#     padding_side = args.padding_side
#     model_path = args.model_list[str(index)]['path']
#     main_lr = args.main_lr
#     is_local = args.k - 1 != index
#     pad_token = args.pad_token
#     head_layer_trainable = args.head_layer_trainable
#     encoder_trainable = args.encoder_trainable
#     embedding_trainable = args.embedding_trainable
#     local_encoders_num = args.local_encoders_num

#     local_model, local_model_optimizer, global_model, global_model_optimizer, tokenizer = load_basic_models_llm_bert_new(
#         pretrained, task_type, model_type, current_output_dim, is_local, device, padding_side, model_path, main_lr, pad_token, \
#         head_layer_trainable, encoder_trainable, embedding_trainable, local_encoders_num)

#     args.tokenizer = tokenizer
#     return args, local_model, local_model_optimizer, global_model, global_model_optimizer


# def load_basic_models_llm_bert_new(pretrained, task_type, model_type, current_output_dim, is_local, device, \
#                                    padding_side, model_path, main_lr, pad_token, \
#                                    head_layer_trainable, encoder_trainable,  embedding_trainable, \
#                                    local_encoders_num):
#     print('load_basic_models_llm pretrained:', model_path)
#     tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=True)
#     tokenizer.padding_side = padding_side if (padding_side in ["left", "right"]) else "left"

#     if task_type == 'QuestionAnswering':
#         full_model = AutoModelForQuestionAnswering.from_pretrained(model_path)
#     else:
#         full_model = AutoModelForSequenceClassification.from_pretrained(model_path)

#     # for name, param in full_model.named_parameters():
#     #     print("-----full_model--{}:{}".format(name, param.shape))

#     if model_type == 'Roberta':
#         full_bert = full_model.roberta
#     elif model_type == 'Albert':
#         full_bert = full_model.albert
#     else:  # Bert
#         full_bert = full_model.bert

#     if task_type == 'QuestionAnswering':
#         head_layer = full_model.qa_outputs
#     elif task_type == "SequenceClassification":
#         head_layer = full_model.classifier
#     elif task_type == "CausalLM":
#         head_layer = full_model.cls
#     else:
#         assert 1 > 2, "task type not supported"

#     if pad_token == "default":
#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token  # ({'pad_token': '[PAD]'}) # args.tokenizer.eos_token #
#             pad_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)  #
#             full_model.config.pad_token_id = pad_id
#         pad_token = "default_" + tokenizer.pad_token
#     else:
#         tokenizer.pad_token = pad_token  # ({'pad_token': '[PAD]'}) # args.tokenizer.eos_token #
#         pad_id = tokenizer.convert_tokens_to_ids(pad_token)  #
#         full_model.config.pad_token_id = pad_id

#     config = full_model.config
#     all_encoder_num = config.num_hidden_layers

#     ########### Local Model ###########
#     local_model = None
#     local_model_optimizer = None
#     if is_local:
#         print('=== prepare local model')
#         print('all_encoder_num:',all_encoder_num,'  local_encoders_num:',local_encoders_num)
#         local_model = LocalBertModel(full_bert, local_encoders_num, model_type=model_type)
#         # Freeze Backbone
#         for param in local_model.parameters():
#             param.requires_grad = False
#         local_model = local_model.to(device)
#         print(f"local_model parameters: {sum(p.numel() for p in local_model.parameters())}")

#         local_model_optimizer = None
#         local_trainable_params = []
#         print('Local Model: embedding_trainable = ', embedding_trainable[0])
#         for param in local_model.embeddings.parameters():
#             param.requires_grad = embedding_trainable[0]
#         if embedding_trainable[0]:
#             local_trainable_params.extend(list(local_model.embeddings.parameters()))
#             print('yes embedding')
#         print('Local Model: encoder_trainable = ', encoder_trainable[0])
#         for param in local_model.encoder_layer.parameters():
#             param.requires_grad = encoder_trainable[0]
#         if encoder_trainable[0]:
#             local_trainable_params.extend(list(local_model.encoder_layer.parameters()))
#         if len(local_trainable_params) > 0:
#             local_model_optimizer = torch.optim.Adam(local_trainable_params, lr=main_lr)

#     ########### Global Model ###########
#     global_model = None
#     global_model_optimizer = None
#     if not is_local:
#         print('=== prepare global model')
#         global_encoders_num = all_encoder_num - local_encoders_num
#         print('all_encoder_num:',all_encoder_num,'  global_encoders_num:',global_encoders_num)
#         # global part of bert(frozen)
#         global_bert = GlobalBertModel(full_bert, global_encoders_num, model_type=model_type)
#         # add Classification Layer(untrainable)
#         if task_type == "QuestionAnswering":
#             global_model = BertForQuestionAnswering_pretrained(global_bert, head_layer)
#         elif task_type == "SequenceClassification":
#             global_model = BertForSequenceClassification_pretrained(global_bert, head_layer)
#         elif task_type == "CausalLM":
#             global_model = BertLMHeadModel_pretrained(global_bert, head_layer)
#         else:
#             assert 1 > 2, "task type not supported"
#         print(f"global_model parameters: {sum(p.numel() for p in global_model.parameters())}")

#         # Freeze Backbone
#         for param in global_model.bert.parameters():
#             param.requires_grad = False

#         # Head Layer Trainable/Freeze
#         print('Global Model : head_layer_trainable = ', head_layer_trainable[1])
#         for param in global_model.head_layer.parameters():
#             param.requires_grad = head_layer_trainable[1]
#         if head_layer_trainable[1]:
#             global_model_optimizer = torch.optim.Adam(list(global_model.head_layer.parameters()), lr=main_lr)

#         global_model = global_model.to(device)

#     return local_model, local_model_optimizer, global_model, global_model_optimizer, tokenizer


# def load_defense_models_llm(args, index, local_model, local_model_optimizer, global_model, global_model_optimizer):
#     print('Load Defense models')
#     # no defense at all, set some variables as None
#     args.encoder = None
#     adversarial_model = None
#     adversarial_model_optimizer = None

#     # some defense need model, add here
#     if args.apply_defense == True:
#         # for adversarial training
#         if 'adversarial' in args.defense_name.lower():
#             # add adversarial model for local model
#             if not 'party' in args.defense_configs:
#                 args.defense_configs['party'] = [0]
#                 print('[warning] default passive party selected for applying adversarial training')

#             if not ('adversarial_model_lr' in args.defense_configs):
#                 adversarial_model_lr = args.main_lr
#                 print('[warning] default hyper-parameter mid_lr selected for applying MID')
#             else:
#                 adversarial_model_lr = args.defense_configs['adversarial_model_lr']

#             batch_first = args.defense_configs['batch_first'] if ('batch_first' in args.defense_configs) else True
                
#             if not ('adversarial_model' in args.defense_configs):
#                 model_name = 'Adversarial_Mapping'
#             else:
#                 model_name = args.defense_configs['adversarial_model']

#             if index in args.defense_configs['party']:
#                 print(f'adversarial_model:{model_name} batch_first:{batch_first}')
#                 seq_length = args.defense_configs['seq_length']
#                 embed_dim = self.args.model_embedded_dim  # defense_configs['embed_dim']
            
#                 # embed_dim = args.defense_configs['embed_dim']
#                 adversarial_model = globals()[model_name](seq_length, embed_dim, batch_first).to(args.device)

#                 local_model = local_model.to(args.device)
#                 # update optimizer
#                 adversarial_model_optimizer = torch.optim.Adam(
#                     [{'params': adversarial_model.parameters(), 'lr': adversarial_model_lr}])

#     return args, local_model, local_model_optimizer, global_model, global_model_optimizer, adversarial_model, adversarial_model_optimizer

def load_models_per_party_llm(args, index):
    current_model_type = args.model_list[str(index)]['type']
    val_model = None
    args, local_model, local_model_optimizer, global_model, global_model_optimizer = load_basic_models_llm(args,index)
    return args, local_model, local_model_optimizer, global_model, global_model_optimizer

def load_models_per_party(args, index):
    current_model_type = args.model_list[str(index)]['type']
    val_model = None
    
    args, local_model, local_model_optimizer, global_model, global_model_optimizer = load_basic_models(args, index)
    args, local_model, local_model_optimizer, global_model, global_model_optimizer = load_defense_models(args,index,local_model,local_model_optimizer,global_model,global_model_optimizer)
    # important
    return args, local_model, local_model_optimizer, global_model, global_model_optimizer


if __name__ == '__main__':
    pass
