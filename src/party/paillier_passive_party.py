import sys, os
sys.path.append(os.pardir)
import torch
from torch.utils.data import DataLoader
from party.passive_party import PassiveParty
from dataset.party_dataset import PassiveDataset
from dataset.party_dataset import ActiveDataset


class PaillierPassiveParty(PassiveParty):
    def __init__(self, args, index):
        super().__init__(args, index)
        self.pk = None

    def set_pk(self, pk):
        self.pk = pk

    def give_pred(self):
        # ####### Noisy Sample #########
        if self.args.apply_ns == True and (self.index in self.args.attack_configs['party']):
            scale = self.args.attack_configs['noise_lambda']
            self.local_pred = self.local_model(noisy_sample(self.local_batch_data,scale))
        # ####### Noisy Sample #########
        else:
            self.local_pred = self.local_model(self.local_batch_data)
        self.local_pred_clone = self.local_pred.detach().clone()
        return self.local_pred, self.local_pred_clone

    def prepare_data(self, args, index):
        super().prepare_data(args, index)
        self.train_dst = PassiveDataset(self.train_data)
        self.test_dst = PassiveDataset(self.test_data)
        if self.args.need_auxiliary == 1:
            self.aux_dst = ActiveDataset(self.aux_data, self.aux_label)
            # self.aux_loader = DataLoader(self.aux_dst, batch_size=batch_size,shuffle=True)

    def local_backward(self):
        # update local model
        self.local_model_optimizer.zero_grad()
        # dummy_local_gradient = torch.ones(self.local_gradient.size())
        # torch.autograd.backward(self.local_pred, self.local_pred)
        params = list(self.local_model.parameters())
        params[0].grad = torch.matmul(self.local_gradient.T, self.local_batch_data.reshape(self.local_gradient.shape[0], -1)) / self.local_batch_data.shape[0]
        params[1].grad = torch.sum(self.local_gradient, dim=0) / self.local_batch_data.shape[0]
        self.local_model_optimizer.step()
