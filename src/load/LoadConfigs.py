import sys, os
sys.path.append(os.pardir)

import json
import argparse

def load_configs(config_file_name, args):
    config_file_path = './configs/'+config_file_name+'.json'
    config_file = open(config_file_path,"r")
    config_dict = json.load(config_file)
    # print(config_dict)
    
    # args.k, number of participants
    args.k = config_dict['k'] if('k' in config_dict) else 2
    
    # args.dataset_split
    args.dataset_split = config_dict['dataset'] if('dataset' in config_dict) else None

    # args.model_list, specify the types of models
    if 'model_list' in config_dict:
        config_model_dict = config_dict['model_list']
        model_dict = {}
        default_dict_element = {'type': 'MLP2', 'path': 'random_14*28_10'}
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
    else:
        default_model_dict = {}
        default_dict_element = {'type': 'MLP2', 'path': '../models/MLP2/random'}
        for ik in range(args.k):
            default_model_dict[str(ik)] = default_dict_element
        args.model_list = default_model_dict
    
    if 'attack_methods' in config_dict:
        config_attack_methods_dict = config_dict['attack_methods']
        methods_list = []
        for key in config_attack_methods_dict:
            if config_attack_methods_dict[key] != 0:
                methods_list.append(key)
        if len(methods_list) == 0:
            methods_list.append('BatchLabelReconstruction')
        args.attack_methods = methods_list
    else:
        args.attack_methods = ['BatchLabelReconstruction']
    
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


    if 'defense_methods' in config_dict:
        config_defense_dict = config_dict['defense_methods']
        defense_methods = []
        args.apply_trainable_layer = config_defense_dict['apply_trainable_layer'] if ('apply_trainable_layer' in config_defense_dict) else 0
        args.apply_laplace = config_defense_dict['apply_laplace'] if ('apply_laplace' in config_defense_dict) else 0
        args.apply_gaussian = config_defense_dict['apply_gaussian'] if ('apply_gaussian' in config_defense_dict) else 0
        args.dp_strength = config_defense_dict['dp_strength'] if ('dp_strength' in config_defense_dict) else 0.0
        args.apply_grad_spar = config_defense_dict['apply_grad_spar'] if ('apply_grad_spar' in config_defense_dict) else 0
        args.grad_spars = config_defense_dict['grad_spars'] if ('grad_spars' in config_defense_dict) else 0.0
        args.apply_encoder = config_defense_dict['apply_encoder'] if ('apply_encoder' in config_defense_dict) else 0
        args.apply_adversarial_encoder = config_defense_dict['apply_adversarial_encoder'] if ('apply_adversarial_encoder' in config_defense_dict) else 0
        args.ae_lambda = config_defense_dict['ae_lambda'] if ('ae_lambda' in config_defense_dict) else 0.1
        args.encoder = config_defense_dict['encoder'] if ('encoder' in config_defense_dict) else None
        args.apply_marvell = config_defense_dict['apply_marvell'] if ('apply_marvell' in config_defense_dict) else 0
        args.marvell_s = config_defense_dict['marvell_s'] if ('marvell_s' in config_defense_dict) else 0
    else:
        args.apply_trainable_layer = 0
        args.apply_laplace = 0
        args.apply_gaussian = 0
        args.dp_strength = 0.0
        args.apply_grad_spar = 0
        args.grad_spars = 0.0
        args.apply_encoder = 0
        args.apply_adversarial_encoder = 0
        args.ae_lambda = 0.1
        args.encoder = None
        args.apply_marvell = 0
        args.marvell_s = 0

    # important
    return args


def load_attack_configs(config_file_name, attack_name, args):
    config_file_path = './configs/attacks/'+config_file_name+'.json'
    config_file = open(config_file_path,"r")
    config_dict = json.load(config_file)
    if attack_name == 'BatchLabelReconstruction':
        # args.attacker_train_config
        if 'attacker_train' in config_dict:
            config_attacker_dict = config_dict['attacker_train']
            args.batch_size = config_attacker_dict['batch_size'] if ('batch_size' in config_attacker_dict) else 64
            args.epochs = config_attacker_dict['epochs'] if ('epochs' in config_attacker_dict) else 4000
            args.lr = config_attacker_dict['lr'] if ('lr' in config_attacker_dict) else 0.01
            args.num_exp = config_attacker_dict['num_exp'] if ('num_exp' in config_attacker_dict) else 10
            args.early_stop = config_attacker_dict['early_stop'] if ('early_stop' in config_attacker_dict) else 0
            args.early_stop_param = config_attacker_dict['early_stop_param'] if ('early_stop_param' in config_attacker_dict) else 0.0001
        else:
            args.batch_size = 64
            args.epochs = 4000
            args.lr = 0.01
            args.num_exp = 10
            args.early_stop = 0
            args.early_stop_param = 0.0001
    else:
        assert attack_name == 'BatchLabelReconstruction', 'Invalid attack method, please double check'
    
    # important
    return args

if __name__ == '__main__':
    args = argparse.ArgumentParser("backdoor").parse_args()
    load_configs("default_config",args)