import sys, os
sys.path.append(os.pardir)

from torch.utils.data import DataLoader

from models.vision import *
from models.model_templates import *
from models.autoencoder import *
from load.LoadDataset import load_dataset_per_party
from load.LoadModels import *

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
        # local model
        self.local_model = None
        self.local_model_optimizer = None
        # global_model
        self.global_model = None
        self.global_model_optimizer = None
        
        self.prepare_data(args, index)
        self.prepare_model(args, index)

    def prepare_data(self, args, index):
        # prepare raw data for training
        args, self.half_dim, (self.train_data,self.train_label), (self.test_data,self.test_label) = load_dataset_per_party(args, index)

    def prepare_data_loader(self, batch_size):
        self.train_loader = DataLoader(self.train_dst, batch_size=batch_size)
        self.test_loader = DataLoader(self.test_dst, batch_size=batch_size)

    def prepare_model(self, args, index):
        # prepare model and optimizer
        args, self.local_model, self.local_model_optimizer, self.global_model, self.global_model_optimizer = load_models_per_party(args, index)

    def local_forward():
        # args.local_model()
        pass

    def local_backward(self, gradient, local_pred):
        # update local model
        self.local_model_optimizer.zero_grad()
        weights_grad_a = torch.autograd.grad(local_pred, self.local_model.parameters(), grad_outputs=gradient, retain_graph=True)
        for w, g in zip(self.local_model.parameters(), weights_grad_a):
            if w.requires_grad:
                w.grad = g.detach()        
        self.local_model_optimizer.step()

