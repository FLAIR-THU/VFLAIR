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
    # args.Q ,iteration_per_aggregation for FedBCD
    args.Q = config_dict['iteration_per_aggregation'] if ('iteration_per_aggregation' in config_dict) else 1
    # # args.early_stop, if use early stop
    # args.main_early_stop = config_dict['main_early_stop'] if ('main_early_stop' in config_dict) else 0
    # args.main_early_stop_param = config_dict['main_early_stop_param'] if ('main_early_stop_param' in config_dict) else 0.0001
    # # args.num_exp number of repeat experiments for main task
    # args.num_exp = config_dict['num_exp'] if ('num_exp' in config_dict) else 10
    
    # args.dataset_split
    args.dataset_split = config_dict['dataset'] if('dataset' in config_dict) else None
    args.num_classes = args.dataset_split['num_classes'] if('num_classes' in args.dataset_split) else 10

    # args.model_list, specify the types of models
    if 'model_list' in config_dict:
        config_model_dict = config_dict['model_list']
        #print('config_model_dict:',(len(config_model_dict)-2))
        assert ((len(config_model_dict)-2)==args.k), 'please alter party number k, model number should be equal to party number'
        
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
    
    # if attacker appears
    args.apply_attack = False
    args.apply_backdoor = False
    args.apply_mid = False
    args.apply_cae = False
    if 'attack' in config_dict:
        if 'name' in config_dict['attack']:
            args.apply_attack = True
            args.attack_name = config_dict['attack']['name']
            args.attack_configs = config_dict['attack']['parameters'] if('parameters' in config_dict['attack']) else None
            if 'backdoor' in args.attack_name.casefold():
                args.apply_backdoor = True
        else:
            assert 'name' in config_dict['attack'], "missing attack name"
    
    args.apply_defense = False
    if 'defense' in config_dict:
        if 'name' in config_dict['defense']:
            args.apply_defense = True
            args.defense_name = config_dict['defense']['name']
            args.defense_configs = config_dict['defense']['parameters'] if('parameters' in config_dict['defense']) else None
            if 'mid' in args.defense_name.casefold():
                args.apply_mid = True
            elif 'cae' in args.defense_name.casefold():
                args.apply_cae = True
        else:
            assert 'name' in config_dict['defense'], "missing defense name"
    
    # important
    return args


if __name__ == '__main__':
    pass
