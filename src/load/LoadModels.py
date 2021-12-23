import sys, os
sys.path.append(os.pardir)

import argparse
import numpy as np
import pickle

from models.vision import *

def load_models(args):
    args.net_list = [None] * args.k
    for ik in range(args.k):
        current_model_type = args.model_list[str(ik)]['type']
        current_model_path = args.model_list[str(ik)]['path']
        args.net_list[ik] = pickle.load(open('.././model_parameters/'+current_model_type+'/'+current_model_path+'.pkl',"rb"))
        args.net_list[ik] = args.net_list[ik].to(args.device)
    
    # if args.model == 'MLP2':
    #     args.net_a = MLP2(np.prod(list(args.gt_data_a.size())[1:]), args.num_classes).to(args.device)
    #     # args.net_a = pickle.load(open('.././model_parameters/MLP2/random.pkl',"rb"))
    #     # args.net_a = args.net_a.to(args.device)
    #     args.net_b = MLP2(np.prod(list(args.gt_data_b.size())[1:]), args.num_classes).to(args.device)
    #     # args.net_b = pickle.load(open('.././model_parameters/MLP2/random.pkl',"rb"))
    #     # args.net_b = args.net_b.to(args.device)
    # elif args.model == 'resnet18':
    #     args.net_a = resnet18(args.num_classes).to(args.device)
    #     args.net_b = resnet18(args.num_classes).to(args.device)

    # important
    return args

if __name__ == '__main__':
    args = argparse.ArgumentParser("backdoor").parse_args()
    args.num_classes = 100
    
    # temp_net = MLP2(14*28, args.num_classes)
    # pickle.dump(temp_net, open('../../model_parameters/MLP2/random_14*28_'+str(args.num_classes)+'.pkl','wb'))
    # loaded_net = pickle.load(open('../../model_parameters/MLP2/random_14*28_'+str(args.num_classes)+'.pkl',"rb"))

    # temp_net = MLP2(16*32, args.num_classes)
    # pickle.dump(temp_net, open('../../model_parameters/MLP2/random_16*32_'+str(args.num_classes)+'.pkl','wb'))
    # loaded_net = pickle.load(open('../../model_parameters/MLP2/random_16*32_'+str(args.num_classes)+'.pkl',"rb"))


    temp_net = resnet18(args.num_classes)
    pickle.dump(temp_net, open('../../model_parameters/resnet18/random_'+str(args.num_classes)+'.pkl','wb'))
    loaded_net = pickle.load(open('../../model_parameters/resnet18/random_'+str(args.num_classes)+'.pkl',"rb"))

    print(temp_net)
    print(loaded_net)

    # if args.model == 'MLP2':
    #     args.net_a = MLP2(np.prod(list(args.gt_data_a.size())[1:]), args.num_classes).to(args.device)
    #     args.net_b = MLP2(np.prod(list(args.gt_data_b.size())[1:]), args.num_classes).to(args.device)
    # elif args.model == 'resnet18':
    #     args.net_a = resnet18(args.num_classes).to(args.device)
    #     args.net_b = resnet18(args.num_classes).to(args.device)