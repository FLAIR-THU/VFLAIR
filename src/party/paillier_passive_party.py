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
        self.sk = None

    def set_keypairs(self, pk, sk):
        self.pk = pk
        self.sk = sk

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
        torch.autograd.backward(self.local_pred, self.local_gradient)
        self.local_model_optimizer.step()
