import sys, os
sys.path.append(os.pardir)

import json
import argparse
from models.autoencoder import AutoEncoder

TARGETED_BACKDOOR = ['ReplacementBackdoor','ASB']
UNTARGETED_BACKDOOR = ['NoisyLabel','MissingFeature','NoisySample']
LABEL_INFERENCE = ['BatchLabelReconstruction','DirectLabelScoring','NormbasedScoring',\
'DirectionbasedScoring','PassiveModelCompletion','ActiveModelCompletion']
ATTRIBUTE_INFERENCE = ['AttributeInference']
FEATURE_INFERENCE = ['GenerativeRegressionNetwork','ResSFL']

comminication_protocal_list = ['FedBCD_p','FedBCD_s','CELU','Compress']

def load_basic_configs(config_file_name, args):
    config_file_path = './configs/'+config_file_name+'.json'
    config_file = open(config_file_path,"r")
    config_dict = json.load(config_file)
    # print(config_dict)
    
    # args.main_lr, learning rate for main task
    args.main_lr = config_dict['lr'] if('lr' in config_dict) else 0.001
    assert (args.main_lr>0), "main learning rate should be >0"

    # args.main_epochs, iterations for main task
    args.main_epochs = config_dict['epochs'] if('epochs' in config_dict) else 50
    
    # args.early_stop_threshold, early stop max epoch
    args.early_stop_threshold = config_dict['early_stop_threshold'] if('early_stop_threshold' in config_dict) else 5
    
    # args.k, number of participants
    args.k = config_dict['k'] if('k' in config_dict) else 2
    assert (args.k % 1 == 0 and args.k>0), "k should be positive integers"

    # args.batch_size for main task
    args.batch_size = config_dict['batch_size'] if ('batch_size' in config_dict) else 2048
    
    # args.Q ,iteration_per_aggregation for FedBCD
    args.Q = config_dict['iteration_per_aggregation'] if ('iteration_per_aggregation' in config_dict) else 1
    assert (args.Q % 1 == 0 and args.Q>0), "iteration_per_aggregation should be positive integers"
    
    args.comminication_protocal = config_dict['comminication_protocal'] if ('comminication_protocal' in config_dict) else "FedBCD_p"
    assert (args.comminication_protocal in comminication_protocal_list), "comminication_protocal not available"
    print('comminication_protocal:',args.comminication_protocal)
    print('iteration_per_aggregation:',args.Q)
    # args.BCD_type = config_dict['BCD_type'] if ('BCD_type' in config_dict) else "p"
    # assert (args.BCD_type == "s" or args.BCD_type == "p"), "args.BCD_type should be positive s/p"
    
    args.attacker_id = []
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
                    if 'path' in config_model_dict[str(ik)] or (('input_dim' in config_model_dict[str(ik)]) and ('output_dim' in config_model_dict[str(ik)])):
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
    
    # Check: Centralized Training
    if args.k ==1 :
        print('k=1, Launch Centralized Training, All Attack&Defense dismissed, Q set to 1')
        args.apply_attack = False # bli/ns/ds attack
        args.apply_backdoor = False # replacement backdoor attack
        args.apply_nl = False # noisy label attack
        args.apply_ns = False # noisy sample attack
        args.apply_mf = False # missing feature attack
        args.apply_defense = False
        args.apply_mid = False
        args.apply_cae = False
        args.apply_dcae = False
        args.apply_dp = False
        args.Q=1
        # return args

    # if defense appears
    args.apply_defense = False
    args.apply_dp = False
    args.apply_mid = False # mid defense
    args.apply_cae = False # cae defense
    args.apply_dcae = False # dcae defense
    args.apply_adversarial = False # adversarial
    args.bin_size = [None for _ in range(args.k)] # for discrete bins
    args.gradients_res_a = [None for _ in range(args.k)] # for gradient sparsification
    args.apply_dcor = False # distance corrilation
    if 'defense' in config_dict:
        print(config_dict['defense'].keys())
        if 'name' in config_dict['defense']:
            args.apply_defense = True
            args.defense_name = config_dict['defense']['name']
            args.defense_configs = config_dict['defense']['parameters'] if('parameters' in config_dict['defense']) else None
            assert 'party' in config_dict['defense']['parameters'], '[Error] Defense party not specified'
            print(f"in load configs, defense_configs is type {type(args.defense_configs)}")
            print(f"in load configs, defense_configs is type {type(dict(args.defense_configs))}")
            if 'mid' in args.defense_name.casefold():
                args.apply_mid = True
            elif 'cae' in args.defense_name.casefold():
                args.apply_cae = True
                if 'dcae' in args.defense_name.casefold():
                    args.apply_dcae = True
            elif 'adversarial' in args.defense_name.casefold():
                args.apply_adversarial = True
            elif 'distancecorrelation' in args.defense_name.casefold():
                args.apply_dcor = True
            elif ('gaussian' in args.defense_name.casefold()) or ('laplace' in args.defense_name.casefold()):
                args.apply_dp = True
        else:
            assert 'name' in config_dict['defense'], "missing defense name"
    else:
        args.defense_name = 'None'
        args.defense_configs = None
        print('===== No Defense ======')
    # get Info: args.defense_param  args.defense_param_name
    if args.apply_defense == True:
        if args.defense_name in ["CAE", "DCAE", "MID", "DistanceCorrelation", "AdversarialTraining"]:
            args.defense_param = args.defense_configs['lambda']
            args.defense_param_name = 'lambda'
        elif args.defense_name == "GaussianDP" or args.defense_name=="LaplaceDP":
            args.defense_param = args.defense_configs['dp_strength']
            args.defense_param_name = 'dp_strength'
        elif args.defense_name == "GradientSparsification":
            args.defense_param = args.defense_configs['gradient_sparse_rate']
            args.defense_param_name = 'gradient_sparse_rate'
        elif args.defense_name == "GradPerturb":
            args.defense_param = args.defense_configs['perturb_epsilon']
            args.defense_param_name = 'perturb_epsilon'
        else:
            args.defense_param = 'None'
            args.defense_param_name = 'No_Defense'
    else:
        args.defense_param = 'None'
        args.defense_param_name = 'No_Defense'
    
    # Detail Specifications about defense_name
    if args.defense_name=="MID":
        if args.defense_configs['party'] == [0]:
            args.defense_name="MID_Passive"
        else:
            args.defense_name="MID_Active"


    # if there's attack   Mark attack type
    args.attack_num = 0
    args.targeted_backdoor_list = []
    args.targeted_backdoor_index = []
    args.untargeted_backdoor_list = []
    args.untargeted_backdoor_index = []
    args.label_inference_list = []
    args.label_inference_index = []
    args.attribute_inference_list = []
    args.attribute_inference_index = []
    args.feature_inference_list = []
    args.feature_inference_index = []
    args.apply_attack = False
    if 'attack_list' in config_dict:
        if len(config_dict['attack_list'])>0:
            attack_config_dict = config_dict['attack_list']
            args.attack_num = len(attack_config_dict)
            args.apply_attack = True
            for ik in range(args.attack_num):
                if 'name' in attack_config_dict[str(ik)]:
                    _name = attack_config_dict[str(ik)]['name']
                    if _name in TARGETED_BACKDOOR:
                        args.targeted_backdoor_list.append(_name)
                        args.targeted_backdoor_index.append(ik)
                    
                    elif _name in UNTARGETED_BACKDOOR:
                        args.untargeted_backdoor_list.append(_name)
                        args.untargeted_backdoor_index.append(ik)
                    
                    elif _name in LABEL_INFERENCE:
                        args.label_inference_list.append(_name)
                        args.label_inference_index.append(ik)

                    elif _name in ATTRIBUTE_INFERENCE:
                        args.attribute_inference_list.append(_name)
                        args.attribute_inference_index.append(ik)

                    elif _name in FEATURE_INFERENCE:
                        args.feature_inference_list.append(_name)
                        args.feature_inference_index.append(ik)
                else:
                    assert 'name' in attack_config_dict[str(ik)], 'missing attack name'
        else:
            assert len(config_dict['attack_list'])>0, 'empty attack_list'
    else:
        print('===== No Attack ======')
    
    return args   

def load_attack_configs(config_file_name, args, index):
    '''
    load attack[index] in attack_list
    '''
    config_file_path = './configs/'+config_file_name+'.json'
    config_file = open(config_file_path,"r")
    config_dict = json.load(config_file)

    args.attack_type = None
    args.apply_backdoor = False # replacement backdoor attack
    args.apply_nl = False # noisy label attack
    args.apply_ns = False # noisy sample attack
    args.apply_mf = False # missing feature attack
    
    # No Attack
    if index == -1:
        print('No Attack==============================')
        args.attack_name='No_Attack'
        args.attack_param_name = 'None'
        args.attack_param = None
        return args
    
    # init args about attacks
    assert args.apply_attack == True 
    # choose attack[index]
    attack_config_dict = config_dict['attack_list'][str(index)]

    args.attaker_id = attack_config_dict['party'] if('party' in attack_config_dict) else []

    if 'name' in attack_config_dict:
        args.attack_name = attack_config_dict['name']
        args.attack_configs = attack_config_dict['parameters'] if('parameters' in attack_config_dict) else None
        
        if args.attack_name in TARGETED_BACKDOOR:
            args.attack_type = 'targeted_backdoor'
            if 'backdoor' in args.attack_name.casefold():
                args.apply_backdoor = True
            
        elif args.attack_name in UNTARGETED_BACKDOOR:
            args.attack_type = 'untargeted_backdoor'
            if 'noisylabel' in args.attack_name.casefold():
                args.apply_nl = True
            if 'noisysample' in args.attack_name.casefold():
                args.apply_ns = True
            if 'missingfeature' in args.attack_name.casefold():
                args.apply_mf = True

        elif args.attack_name in LABEL_INFERENCE:
            args.attack_type = 'label_inference'

        elif args.attack_name in ATTRIBUTE_INFERENCE:
            args.attack_type = 'attribute_inference'

        elif args.attack_name in FEATURE_INFERENCE:
            args.attack_type = 'feature_inference'

        else:
            assert 0 , 'attack type not supported'
        
        if args.attack_name == 'NoisyLabel':
            args.attack_param_name = 'noise_type'
            args.attack_param = str(attack_config_dict['parameters']['noise_type'])+'_'+str(attack_config_dict['parameters']['noise_rate'])
        elif args.attack_name == 'MissingFeature':
            args.attack_param_name = 'missing_rate'
            args.attack_param = str(attack_config_dict['parameters']['missing_rate'])
        elif args.attack_name == 'BatchLabelReconstruction':
            args.attack_param_name = 'attack_lr'
            args.attack_param = str(attack_config_dict['parameters']['lr'])
        elif args.attack_name == 'NoisySample':
            args.attack_param_name = 'noise_lambda'
            args.attack_param = str(attack_config_dict['parameters']['noise_lambda'])
        else:
            args.attack_param_name = 'None'
            args.attack_param = None
        
    else:
        assert 'name' in attack_config_dict, "missing attack name"

    # Check: Centralized Training
    if args.k ==1:
        print('k=1, Launch Centralized Training, All Attack&Defense dismissed, Q set to 1')
        args.apply_attack = False # bli/ns/ds attack
        args.apply_backdoor = False # replacement backdoor attack
        args.apply_nl = False # noisy label attack
        args.apply_ns = False # noisy sample attack
        args.apply_mf = False # missing feature attack
        args.apply_defense = False
        args.apply_mid = False
        args.apply_cae = False
        args.apply_dcae = False
        args.apply_dp = False
        args.Q=1

    return args

def init_attack_defense(args):
    args.apply_attack = False 
    args.apply_backdoor = False # replacement backdoor attack
    args.apply_nl = False # noisy label attack
    args.apply_ns = False # noisy sample attack
    args.apply_mf = False # missing feature attack
    args.apply_defense = False
    args.apply_mid = False
    args.apply_cae = False
    args.apply_dcae = False
    args.apply_dp = False
    return args
    


if __name__ == '__main__':
    pass
