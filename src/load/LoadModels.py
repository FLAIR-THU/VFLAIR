import sys, os
sys.path.append(os.pardir)

import argparse
import numpy as np
import pickle
from transformers import BertTokenizer, BertModel, BertConfig,PretrainedConfig, BertPreTrainedModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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


from models.LLM_models import *
from models.bottom_models import *
from models.global_models import *
from models.autoencoder import *
from utils.optimizers import MaliciousSGD, MaliciousAdam

MODEL_PATH = {'bert-base-uncased': "/home/DAIR/guzx/.cache/huggingface/hub/bert-base-uncased",
"Bert-sequence-classification": "/home/DAIR/guzx/.cache/huggingface/hub/Bert-sequence-classification",
"toxic-bert": "/home/DAIR/guzx/.cache/huggingface/hub/toxic-bert",
"textattackbert-base-uncased-CoLA": "/home/DAIR/guzx/.cache/huggingface/hub/textattackbert-base-uncased-CoLA",
"textattackbert-base-uncased-SST-2": "/home/DAIR/guzx/.cache/huggingface/hub/textattackbert-base-uncased-SST-2",
"textattackbert-base-uncased-STS-B": "/home/DAIR/guzx/.cache/huggingface/hub/textattackbert-base-cased-STS-B",
"textattackbert-base-uncased-MRPC": "/home/DAIR/guzx/.cache/huggingface/hub/textattackbert-base-uncased-MRPC",
"textattackbert-base-uncased-MNLI": "/home/DAIR/guzx/.cache/huggingface/hub/textattackbert-base-uncased-MNLI",
"textattackbert-base-uncased-QNLI": "/home/DAIR/guzx/.cache/huggingface/hub/textattackbert-base-uncased-QNLI",
"textattackbert-base-uncased-QQP": "/home/DAIR/guzx/.cache/huggingface/hub/textattackbert-base-uncased-QQP",
"textattackbert-base-uncased-WNLI": "/home/DAIR/guzx/.cache/huggingface/hub/textattackbert-base-uncased-WNLI",
"textattackbert-base-uncased-RTE": "/home/DAIR/guzx/.cache/huggingface/hub/textattackbert-base-uncased-RTE",

}
ROBERTA = ["bertweet-base-sentiment-analysis"]
LLM_supported = ['bert-base-uncased','Bert-sequence-classification',"toxic-bert",\
"textattackbert-base-uncased-CoLA","textattackbert-base-uncased-SST-2","textattackbert-base-uncased-STS-B",\
"textattackbert-base-uncased-MRPC","textattackbert-base-uncased-MNLI","textattackbert-base-uncased-QNLI",\
"textattackbert-base-uncased-QQP","textattackbert-base-uncased-WNLI","textattackbert-base-uncased-RTE"]

def create_model(bottom_model, ema=False, size_bottom_out=10, num_classes=10):
    model = BottomModelPlus(bottom_model,size_bottom_out, num_classes,
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
        args.net_list[ik] = pickle.load(open('.././src/models/model_parameters/'+current_model_type+'/'+current_model_path+'.pkl',"rb"))
        args.net_list[ik] = args.net_list[ik].to(args.device)
    # important
    return args

def load_basic_models(args,index):
    current_model_type = args.model_list[str(index)]['type']
    print(f"current_model_type={current_model_type}")
    current_input_dim = args.model_list[str(index)]['input_dim'] if 'input_dim' in args.model_list[str(index)] else args.half_dim[index]
    current_hidden_dim = args.model_list[str(index)]['hidden_dim'] if 'hidden_dim' in args.model_list[str(index)] else -1
    current_output_dim = args.model_list[str(index)]['output_dim']
    current_vocab_size = args.model_list[str(index)]['vocab_size'] if 'vocab_size' in args.model_list[str(index)] else -1
    # print(f"index={index}, current_input_dim={current_input_dim}, current_output_dim={current_output_dim}")
    # current_model_path = args.model_list[str(index)]['path']
    # local_model = pickle.load(open('.././model_parameters/'+current_model_type+'/'+current_model_path+'.pkl',"rb"))
    if 'resnet' in current_model_type.lower() or 'lenet' in current_model_type.lower() or 'cnn' in current_model_type.lower() or 'alexnet' in current_model_type.lower():
        local_model = globals()[current_model_type](current_output_dim)
    elif 'gcn' in current_model_type.lower():
        local_model = globals()[current_model_type](nfeat=current_input_dim,nhid=current_hidden_dim,nclass=current_output_dim, device=args.device, dropout=0.0, lr=args.main_lr)
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
    if index == args.k-1:
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


def load_basic_models_llm(args,index):
    current_model_type = args.model_list[str(index)]['type']
    current_output_dim = args.model_list[str(index)]['output_dim']

    if args.pretrained == 0:
        args.tokenizer = BertTokenizer.from_pretrained(MODEL_PATH[current_model_type], do_lower_case=True)
        full_bert = BertModel.from_pretrained(MODEL_PATH[current_model_type])
        config = full_bert.config #print(full_bert.encoder.layer[0])

        ########### Local Model ###########
        local_model = None
        local_model_optimizer = None
        if index < args.k-1:
            local_model = LocalBertModel(full_bert,1)
            # Freeze Backbone
            for param in local_model.parameters():
                param.requires_grad = False
            local_model = local_model.to(args.device)
            print(f"local_model parameters: {sum(p.numel() for p in local_model.parameters())}")
            local_model_optimizer = None

        ########### Global Model ###########
        global_model = None
        global_model_optimizer = None
        if index == args.k-1:
            # global part of bert(frozen)
            global_bert = GlobalBertModel(full_bert,1)
            # add Classification Layer(trainable)
            global_model = GlobalBertClassifier(global_bert, current_output_dim)
            print(f"global_model parameters: {sum(p.numel() for p in global_model.parameters())}")
            # Freeze Backbone
            for param in global_model.backbone.parameters():
                param.requires_grad = False
            # Trainable Part for finetuning
            print(args.model_path)
            if args.model_path != "":
                global_model.trainable_layer.load_state_dict(torch.load(args.model_path))
            for param in global_model.trainable_layer.parameters():
                param.requires_grad = True
            global_model = global_model.to(args.device)
            global_model_optimizer = torch.optim.Adam(list(global_model.trainable_layer.parameters()), lr=args.main_lr)
    else:
        print('load_basic_models_llm pretrained:',current_model_type)
        args.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH[current_model_type], do_lower_case=True)
        full_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH[current_model_type])
        
        # for name, param in full_model.named_parameters():
        #     print("-----full_model--{}:{}".format(name, param.shape))
        if current_model_type in ROBERTA:
            full_bert = full_model.roberta
        else:
            full_bert = full_model.bert
        classifier = full_model.classifier
        config = full_model.config


        ########### Local Model ###########
        local_model = None
        local_model_optimizer = None
        if index < args.k-1:
            local_model = LocalBertModel(full_bert,1)
            # Freeze Backbone
            for param in local_model.parameters():
                param.requires_grad = False
            local_model = local_model.to(args.device)
            print(f"local_model parameters: {sum(p.numel() for p in local_model.parameters())}")
            local_model_optimizer = None

        ########### Global Model ###########
        global_model = None
        global_model_optimizer = None
        if index == args.k-1:
            # global part of bert(frozen)
            global_bert = GlobalBertModel(full_bert,1)
            
            # add Classification Layer(untrainable)
            global_model = GlobalBertClassifier_pretrained(global_bert, classifier)
            print(f"global_model parameters: {sum(p.numel() for p in global_model.parameters())}")
            
            # Freeze Backbone
            for param in global_model.backbone.parameters():
                param.requires_grad = False
            # Classifier already pretrained
            for param in global_model.classifier.parameters():
                param.requires_grad = False
            global_model = global_model.to(args.device)
            global_model_optimizer = None
    return args, local_model, local_model_optimizer, global_model, global_model_optimizer

def load_models_per_party(args, index):
    current_model_type = args.model_list[str(index)]['type']
    val_model = None
    if current_model_type in LLM_supported:
        args, local_model, local_model_optimizer, global_model, global_model_optimizer = load_basic_models_llm(args,index)
        args, local_model, local_model_optimizer, global_model, global_model_optimizer = load_defense_models(args, index, local_model, local_model_optimizer, global_model, global_model_optimizer)
    else:
        args, local_model, local_model_optimizer, global_model, global_model_optimizer = load_basic_models(args,index)
        args, local_model, local_model_optimizer, global_model, global_model_optimizer = load_defense_models(args, index, local_model, local_model_optimizer, global_model, global_model_optimizer)
        # important
    return args, local_model, local_model_optimizer, global_model, global_model_optimizer


if __name__ == '__main__':
    pass
