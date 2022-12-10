import sys, os
sys.path.append(os.pardir)

import json
import argparse
from models.autoencoder import AutoEncoder

def load_configs(config_file_name, args):
    config_file_path = './configs/'+config_file_name+'.json'
    config_file = open(config_file_path,"r")
    config_dict = json.load(config_file)
    # print(config_dict)
    
    # args.main_lr, learning rate for main task
    args.main_lr = config_dict['lr'] if('lr' in config_dict) else 0.001
    # args.main_epochs, iterations for main task
    args.main_epochs = config_dict['epochs'] if('epochs' in config_dict) else 50
    # args.k, number of participants
    args.k = config_dict['k'] if('k' in config_dict) else 2
    # args.batch_size for main task
    args.batch_size = config_dict['batch_size'] if ('batch_size' in config_dict) else 2048
    # args.num_exp number of repeat experiments for main task
    args.num_exp = config_dict['num_exp'] if ('num_exp' in config_dict) else 10
    # # args.early_stop, if use early stop
    # args.early_stop = config_dict['early_stop'] if ('early_stop' in config_dict) else 0
    # args.early_stop_param = config_dict['early_stop_param'] if ('early_stop_param' in config_dict) else 0.0001
    
    
    # args.dataset_split
    args.dataset_split = config_dict['dataset'] if('dataset' in config_dict) else None
    args.num_classes = args.dataset_split['num_classes'] if('num_classes' in args.dataset_split) else 10

    # args.model_list, specify the types of models
    if 'model_list' in config_dict:
        config_model_dict = config_dict['model_list']
        model_dict = {}
        default_dict_element = {'type': 'MLP2', 'path': 'random_14*28_10', 'input_dim': 392, 'output_dim': 10}
        for ik in range(args.k):
            if str(ik) in config_model_dict:
                if 'type' in config_model_dict[str(ik)]:
                    if 'path' in config_model_dict[str(ik)]:
                        model_dict[str(ik)] = config_model_dict[str(ik)]
                    else:
                        model_type_name = config_model_dict[str(ik)]['type']
                        temp = {'type':model_type_name, 'path':'../models/'+model_type_name+'/random'}
                        model_dict[str(ik)] = temp
                else:
                    # if 'path' in config_model_dict[str(ik)]:
                    #     model_type_name = config_model_dict[str(ik)]['path'].split('/')[-2]
                    #     temp = {'type':model_type_name, 'path':config_model_dict[str(ik)]['path']}
                    #     model_dict[str(ik)] = temp
                    # else:
                    #     model_dict[str(ik)] = default_dict_element
                    model_dict[str(ik)] = default_dict_element
            else:
                model_dict[str(ik)] = default_dict_element
        args.model_list = model_dict
        args.apply_trainable_layer = config_model_dict['apply_trainable_layer'] if ('apply_trainable_layer' in config_model_dict) else 0
        args.global_model = config_model_dict['global_model'] if ('global_model' in config_model_dict) else 'ClassificationModelHostHead'
    else:
        default_model_dict = {}
        default_dict_element = {'type': 'MLP2', 'path': '../models/MLP2/random'}
        for ik in range(args.k):
            default_model_dict[str(ik)] = default_dict_element
        args.model_list = default_model_dict
        args.apply_trainable_layer = 0
        args.global_model = 'ClassificationModelHostHead'
    
    if 'attack_methods' in config_dict:
        config_attack_methods_dict = config_dict['attack_methods']
        methods_list = []
        for key in config_attack_methods_dict:
            if config_attack_methods_dict[key] != 0:
                methods_list.append(key)
        # if len(methods_list) == 0:
        #     methods_list.append('BatchLabelReconstruction')
        args.attack_methods = methods_list
    else:
        # args.attack_methods = ['BatchLabelReconstruction']
        args.attack_methods = []
    
    # attack_configs
    if 'attack_configs' in config_dict:
        config_attack_config_dict = config_dict['attack_configs']
        attack_config_list = []
        for attack_method in args.attack_methods:
            attack_config_list.append((config_attack_config_dict[attack_method] \
                                        if attack_method in config_attack_config_dict else (attack_method+'_configs')))
        args.attack_config_list = attack_config_list
    else:
        attack_config_list = []
        for attack_method in args.attack_methods:
            attack_config_list.append((attack_method+'_configs'))
        args.attack_config_list = attack_config_list

    # defense args initialization
    args.apply_laplace = 0
    args.apply_gaussian = 0
    args.dp_strength = 0.0
    args.apply_grad_spar = 0
    args.grad_spars = 0
    args.apply_discrete_gradients = 0
    args.discrete_bins = 0
    args.apply_encoder = 0
    args.ae_lambda = 0.0
    args.encoder = None
    args.apply_marvell = 0
    args.marvell_s = 0
    if 'defense_methods' in config_dict:
        config_defense_methods_dict = config_dict['defense_methods']
        methods_list = []
        for key in config_defense_methods_dict:
            if config_defense_methods_dict[key] != 0:
                methods_list.append(key)
        if len(methods_list) == 0:
            methods_list.append('NoDefense')
        args.defense_methods = methods_list
    else:
        args.defense_methods = ['NoDefense']
        # args.defense_methods = []
    
    # defense_configs
    if 'defense_configs' in config_dict:
        config_defense_config_dict = config_dict['defense_configs']
        defense_config_list = []
        for defense_method in args.defense_methods:
            defense_config_list.append((config_defense_config_dict[defense_method] \
                                        if defense_method in config_defense_config_dict else (defense_method+'_configs')))
        args.defense_config_list = defense_config_list
    else:
        defense_config_list = []
        for defense_method in args.defense_methods:
            defense_config_list.append((defense_method+'_configs'))
        args.defense_config_list = defense_config_list

    # important
    return args


def load_attack_configs(config_file_name, attack_name, args):
    config_file_path = './configs/attacks/'+config_file_name+'.json'
    config_file = open(config_file_path,"r")
    config_dict = json.load(config_file)
    if attack_name == 'MainTaskVFL' or attack_name == "MainTaskVFL_separate":
        # args.attacker_train_config
        if 'attacker_train' in config_dict:
            config_attacker_dict = config_dict['attacker_train']
            args.batch_size = config_attacker_dict['batch_size'] if ('batch_size' in config_attacker_dict) else 2048
            args.epochs = args.main_epochs
            args.lr = config_attacker_dict['lr'] if ('lr' in config_attacker_dict) else 0.01
            args.num_exp = config_attacker_dict['num_exp'] if ('num_exp' in config_attacker_dict) else 10
            args.early_stop = config_attacker_dict['early_stop'] if ('early_stop' in config_attacker_dict) else 0
            args.early_stop_param = config_attacker_dict['early_stop_param'] if ('early_stop_param' in config_attacker_dict) else 0.0001
        else:
            args.batch_size = 2048
            args.epochs = args.main_epochs
            args.lr = 0.01
            args.num_exp = 10
            args.early_stop = 0
            args.early_stop_param = 0.0001
    elif attack_name == 'BatchLabelReconstruction' or attack_name == "DeepLeakageFromGradients":
        # args.attacker_train_config
        if 'attacker_train' in config_dict:
            config_attacker_dict = config_dict['attacker_train']
            args.batch_size = config_attacker_dict['batch_size'] if ('batch_size' in config_attacker_dict) else 2048
            args.epochs = config_attacker_dict['epochs'] if ('epochs' in config_attacker_dict) else 4000
            args.lr = config_attacker_dict['lr'] if ('lr' in config_attacker_dict) else 0.01
            args.num_exp = config_attacker_dict['num_exp'] if ('num_exp' in config_attacker_dict) else 10
            args.early_stop = config_attacker_dict['early_stop'] if ('early_stop' in config_attacker_dict) else 0
            args.early_stop_param = config_attacker_dict['early_stop_param'] if ('early_stop_param' in config_attacker_dict) else 0.0001
        else:
            args.batch_size = 2048
            args.epochs = 4000
            args.lr = 0.01
            args.num_exp = 10
            args.early_stop = 0
            args.early_stop_param = 0.0001
    elif attack_name == 'ReplacementBackdoor':
        if 'attacker_train' in config_dict:
            config_attacker_dict = config_dict['attacker_train']
            args.batch_size = config_attacker_dict['batch_size'] if ('batch_size' in config_attacker_dict) else 2048
            args.epochs = config_attacker_dict['epochs'] if ('epochs' in config_attacker_dict) else 100
            args.lr = config_attacker_dict['lr'] if ('lr' in config_attacker_dict) else 0.01
            args.num_exp = config_attacker_dict['num_exp'] if ('num_exp' in config_attacker_dict) else 10
            args.amplify_rate = config_attacker_dict['amplify_rate'] if ('amplify_rate' in config_attacker_dict) else 10
            args.report_freq = config_attacker_dict['report_freq'] if ('report_freq' in config_attacker_dict) else 100
            args.momentum = config_attacker_dict['momentum'] if ('momentum' in config_attacker_dict) else 0.0
            args.weight_decay = config_attacker_dict['weight_decay'] if ('weight_decay' in config_attacker_dict) else 1e-5
            args.learning_rate_decay_rate = config_attacker_dict['learning_rate_decay_rate'] if ('learning_rate_decay_rate' in config_attacker_dict) else 0.97
            args.decay_period = config_attacker_dict['decay_period'] if ('decay_period' in config_attacker_dict) else 1
            args.worker_thread_number = config_attacker_dict['worker_thread_number'] if ('worker_thread_number' in config_attacker_dict) else 0
            args.use_project_head = config_attacker_dict['use_project_head'] if ('use_project_head' in config_attacker_dict) else 0
            args.explicit_softmax = config_attacker_dict['explicit_softmax'] if ('explicit_softmax' in config_attacker_dict) else 0
            args.seed = config_attacker_dict['seed'] if ('seed' in config_attacker_dict) else 0
            args.early_stop = config_attacker_dict['early_stop'] if ('early_stop' in config_attacker_dict) else 0
            args.early_stop_param = config_attacker_dict['early_stop_param'] if ('early_stop_param' in config_attacker_dict) else 0.0001
        else:
            args.batch_size = 2048
            args.epochs = 100
            args.lr = 0.01
            args.num_exp = 10
            args.amplify_rate = 10
            args.report_freq = 100
            args.momentum = 0.0
            args.weight_decay = 1e-5
            args.learning_rate_decay_rate = 0.97
            args.decay_period = 1
            args.worker_thread_number = 0
            args.use_project_head = 0
            args.explicit_softmax = 0
            args.seed = 0
            args.early_stop = 0
            args.early_stop_param = 0.0001
    else:
        assert attack_name == 'BatchLabelReconstruction', 'Invalid attack method, please double check'
    
    # important
    return args


def load_defense_configs(config_file_name, defense_name, args):
    args.apply_laplace = 0
    args.apply_gaussian = 0
    args.dp_strength = 0.001
    args.apply_grad_spar = 0
    args.grad_spars = 99
    args.apply_discrete_gradients = 0
    args.discrete_gradients_bins = 12
    args.apply_encoder = 0
    args.ae_lambda = 1.0
    args.encoder = None
    args.apply_marvell = 0
    args.marvell_s = 1
    if defense_name != 'NoDefense':
        config_file_path = './configs/defenses/'+config_file_name+'.json'
        config_file = open(config_file_path,"r")
        config_dict = json.load(config_file)
        args.main_lr = config_dict['lr'] if('lr' in config_dict) else args.main_lr
        args.main_epochs = config_dict['epochs'] if('epochs' in config_dict) else args.main_epochs
        if defense_name == 'LaplaceDP' or defense_name == 'GaussianDP':
            if defense_name == 'LaplaceDP':
                args.apply_laplace = 1
            else:
                args.apply_gaussian = 1
            if 'defense_parameters' in config_dict:
                config_defense_dict = config_dict['defense_parameters']
                args.dp_strength = config_defense_dict['dp_strength'] if ('dp_strength' in config_defense_dict) else 0.001
            else:
                args.dp_strength = 0.001
        elif defense_name == 'GradientSparcification':
            args.apply_grad_spar = 1
            if 'defense_parameters' in config_dict:
                config_defense_dict = config_dict['defense_parameters']
                args.grad_spars = config_defense_dict['grad_spars_rate'] if ('grad_spars_rate' in config_defense_dict) else 99
            else:
                args.grad_spars = 99
        elif defense_name == 'DiscreteGradient':
            args.apply_discrete_gradients = 1
            if 'defense_parameters' in config_dict:
                config_defense_dict = config_dict['defense_parameters']
                args.discrete_gradients_bins = config_defense_dict['discrete_gradients_bins'] if ('discrete_gradients_bins' in config_defense_dict) else 12
            else:
                args.discrete_gradients_bins = 12
        elif defense_name == 'ConfusionalAutoEncoder':
            args.apply_encoder = 1
            if 'defense_parameters' in config_dict:
                config_defense_dict = config_dict['defense_parameters']
                args.ae_lambda = config_defense_dict['ae_lambda'] if ('ae_lambda' in config_defense_dict) else 1.0
                encoder_path = config_defense_dict['encoder_path'] if ('encoder_path' in config_defense_dict) else ""
                dim = args.num_class_list[0]
                encoder = AutoEncoder(input_dim=dim, encode_dim=2+dim * 6).to(args.device)
                encoder.load_model(encoder_path, target_device=args.device)
                args.encoder = encoder
            else:
                args.ae_lambda = 1.0
                args.encoder = None
        elif defense_name == 'DiscreteConfusionalAutoEncoder':
            args.apply_discrete_gradients = 1
            args.apply_encoder = 1
            if 'defense_parameters' in config_dict:
                config_defense_dict = config_dict['defense_parameters']
                args.discrete_gradients_bins = config_defense_dict['discrete_gradients_bins'] if ('discrete_gradients_bins' in config_defense_dict) else 12
                args.ae_lambda = config_defense_dict['ae_lambda'] if ('ae_lambda' in config_defense_dict) else 1.0
                encoder_path = config_defense_dict['encoder_path'] if ('encoder_path' in config_defense_dict) else ""
                dim = args.num_class_list[0]
                encoder = AutoEncoder(input_dim=dim, encode_dim=2+dim * 6).to(args.device)
                encoder.load_model(encoder_path, target_device=args.device)
                args.encoder = encoder
            else:
                args.discrete_gradients_bins = 12
                args.ae_lambda = 1.0
                args.encoder = None
        elif defense_name == 'Marvell':
            args.apply_marvell = 1
            if 'defense_parameters' in config_dict:
                config_defense_dict = config_dict['defense_parameters']
                args.marvell_s = config_defense_dict['marvell_s'] if ('marvell_s' in config_defense_dict) else 10
            else:
                args.marvell_s = 10
        else:
            assert defense_name == 'LaplaceDP', 'Invalid attack method, please double check'
    
    # important
    return args


if __name__ == '__main__':
    args = argparse.ArgumentParser("backdoor").parse_args()
    load_configs("default_config",args)