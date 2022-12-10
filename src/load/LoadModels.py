import sys, os
sys.path.append(os.pardir)

import argparse
import numpy as np
import pickle

from models.vision import *
from models.model_templates import *

def load_models(args):
    args.net_list = [None] * args.k
    for ik in range(args.k):
        current_model_type = args.model_list[str(ik)]['type']
        current_model_path = args.model_list[str(ik)]['path']
        args.net_list[ik] = pickle.load(open('.././model_parameters/'+current_model_type+'/'+current_model_path+'.pkl',"rb"))
        args.net_list[ik] = args.net_list[ik].to(args.device)
    # important
    return args

def load_models_per_party(args, index):
    current_model_type = args.model_list[str(index)]['type']
    current_input_dim = args.model_list[str(index)]['input_dim']
    current_output_dim = args.model_list[str(index)]['output_dim']
    # current_model_path = args.model_list[str(index)]['path']
    # local_model = pickle.load(open('.././model_parameters/'+current_model_type+'/'+current_model_path+'.pkl',"rb"))
    if 'resnet' in current_model_type:
        local_model = globals()[current_model_type](current_output_dim)
    else:
        local_model = globals()[current_model_type](current_input_dim,current_output_dim)
    local_model = local_model.to(args.device)
    local_model_optimizer = torch.optim.Adam(list(local_model.parameters()), lr=args.main_lr)
    
    global_model = None
    global_model_optimizer = None
    if index == args.k-1:
        if args.apply_trainable_layer == 0:
            global_model = globals()[args.global_model]()
            global_model = global_model.to(args.device)
            global_model_optimizer = None
        else:
            print("global_model", args.global_model)
            global_model = globals()[args.global_model](args.k*args.num_classes, args.num_classes)
            global_model = global_model.to(args.device)
            global_model_optimizer = torch.optim.Adam(list(global_model.parameters()), lr=args.main_lr)

    # important
    return args, local_model, local_model_optimizer, global_model, global_model_optimizer


if __name__ == '__main__':
    args = argparse.ArgumentParser("backdoor").parse_args()
    args.num_classes = 10
    
    # temp_net = MLP2(28*28, args.num_classes)
    # pickle.dump(temp_net, open('../../model_parameters/MLP2/random_28*28_'+str(args.num_classes)+'.pkl','wb'))
    # loaded_net = pickle.load(open('../../model_parameters/MLP2/random_28*28_'+str(args.num_classes)+'.pkl',"rb"))

    torch.manual_seed(1234)
    def weights_init(m):
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    
    temp_net = LeNet(args.num_classes)
    temp_net.apply(weights_init)
    pickle.dump(temp_net, open('../../model_parameters/LeNet/random_'+str(args.num_classes)+'.pkl','wb'))
    loaded_net = pickle.load(open('../../model_parameters/LeNet/random_'+str(args.num_classes)+'.pkl',"rb"))


    # temp_net = resnet18(args.num_classes)
    # pickle.dump(temp_net, open('../../model_parameters/resnet18/random_'+str(args.num_classes)+'.pkl','wb'))
    # loaded_net = pickle.load(open('../../model_parameters/resnet18/random_'+str(args.num_classes)+'.pkl',"rb"))

    print(temp_net)
    print(loaded_net)

    # if args.model == 'MLP2':
    #     args.net_a = MLP2(np.prod(list(args.gt_data_a.size())[1:]), args.num_classes).to(args.device)
    #     args.net_b = MLP2(np.prod(list(args.gt_data_b.size())[1:]), args.num_classes).to(args.device)
    # elif args.model == 'resnet18':
    #     args.net_a = resnet18(args.num_classes).to(args.device)
    #     args.net_b = resnet18(args.num_classes).to(args.device)