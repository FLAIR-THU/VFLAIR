import sys, os
sys.path.append(os.pardir)

from torch.utils.data import DataLoader

from models.vision import *
from models.model_templates import *
from models.autoencoder import *
from load.LoadDataset import load_dataset_per_party, load_dataset_per_party_backdoor
from load.LoadModels import *
from evaluates.attacks.attack_api import AttackerLoader
from evaluates.defenses.defense_api import DefenderLoader

class Party(object):
    def __init__(self, args, index):
        self.name = 'party#'+str(index+1)
        self.index = index
        self.args = args
        # data for training and testing
        self.half_dim = -1
        self.train_data = None
        self.test_data = None
        self.train_label = None
        self.test_label = None
        self.train_dst = None
        self.test_dst = None
        self.train_loader = None
        self.test_loader = None
        self.local_batch_data = None
        # backdoor poison data and label and target images list
        self.train_poison_data = None
        self.train_poison_label = None
        self.test_poison_data = None
        self.test_poison_label = None
        self.train_target_list = None
        self.test_target_list = None
        # local model
        self.local_model = None
        self.local_model_optimizer = None
        # global_model
        self.global_model = None
        self.global_model_optimizer = None

        # attack and defense
        # self.attacker = None
        self.defender = None
        
        self.prepare_data(args, index)
        self.prepare_model(args, index)
        # self.prepare_attacker(args, index)
        # self.prepare_defender(args, index)

    def prepare_data(self, args, index):
        # prepare raw data for training
        if args.apply_backdoor == True:
            print("in party prepare_data, will prepare poison data for backdooring")
            args, self.half_dim, (self.train_data,self.train_label), (self.test_data,self.test_label), (self.train_poison_data,self.train_poison_label), (self.test_poison_data, self.test_poison_label), self.train_target_list, self.test_target_list = load_dataset_per_party_backdoor(args, index)
        else:
            args, self.half_dim, (self.train_data,self.train_label), (self.test_data,self.test_label) = load_dataset_per_party(args, index)

    def prepare_data_loader(self, batch_size):
        self.train_loader = DataLoader(self.train_dst, batch_size=batch_size)
        self.test_loader = DataLoader(self.test_dst, batch_size=batch_size)

    def prepare_model(self, args, index):
        # prepare model and optimizer
        args, self.local_model, self.local_model_optimizer, self.global_model, self.global_model_optimizer = load_models_per_party(args, index)

    # def prepare_attacker(self, args, index):
    #     if index in args.attack_configs['party']:
    #         self.attacker = AttackerLoader(args, index, self.local_model)

    # def prepare_defender(self, args, index):
    #     if index in args.attack_configs['party']:
    #         self.defender = DefenderLoader(args, index)
    
    def obtain_local_data(self, data):
        self.local_batch_data = data
    
    # def party_attack(self, args, *params):
    #     if self.index in args.attack_configs['party']:
    #         return self.attacker.attack(self.local_batch_data, self.weights_grad_a, *params)

    def local_forward():
        # args.local_model()
        pass

    def local_backward(self, gradient, local_pred):
        # update local model
        self.local_model_optimizer.zero_grad()
        # ########## for passive local mid loss (start) ##########
        if self.args.apply_mid == True and (self.index in self.args.defense_configs['party']) and (self.index < self.args.k-1):
            self.local_model.mid_loss.backward(retain_graph=True)
            self.local_model.mid_loss = torch.empty((1,1)).to(self.args.device)
        # ########## for passive local mid loss (end) ##########
        self.weights_grad_a = torch.autograd.grad(local_pred, self.local_model.parameters(), grad_outputs=gradient, retain_graph=True)
        for w, g in zip(self.local_model.parameters(), self.weights_grad_a):
            if w.requires_grad:
                w.grad = g.detach()        
        self.local_model_optimizer.step()

