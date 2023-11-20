import random
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

        self.partial_update = args.partial_update
        self.partial_update_num = args.partial_update_num

    def set_pk(self, pk):
        self.pk = pk

    def give_pred(self):
        # ####### Noisy Sample #########
        if self.args.apply_ns == True and (
            self.index in self.args.attack_configs["party"]
        ):
            scale = self.args.attack_configs["noise_lambda"]
            self.local_pred = self.local_model(
                noisy_sample(self.local_batch_data, scale)
            )
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

    def receive_gradient(self, gradient):
        if self.partial_update:
            bd = self.local_batch_data.reshape(gradient.size()[0], -1)
            self.sampled_idx = random.sample(list(range(bd.shape[1])), self.partial_update_num)
            self.local_gradient = [
                torch.matmul(gradient.T, bd[:, self.sampled_idx]),
                torch.sum(gradient, dim=0),
            ]
        else:
            self.local_gradient = [
                torch.matmul(
                    gradient.T,
                    self.local_batch_data.reshape(gradient.size()[0], -1),
                ),
                torch.sum(gradient, dim=0),
            ]
        self.local_batch_size = gradient.size()[0]

        """
        self.local_gradient = [
            torch.matmul(
                gradient.T,
                self.local_batch_data.reshape(gradient.size()[0], -1),
            ),
            torch.sum(gradient, dim=0),
        ]
        self.local_batch_size = gradient.size()[0]
        """

        self.random_masks = []
        for i in range(len(self.local_gradient)):
            mask = torch.randn(self.local_gradient[i].size()).to(
                self.local_gradient[i].device
            )
            self.local_gradient[i] = self.local_gradient[i] + mask
            self.random_masks.append(mask)

    def local_backward(self):
        # update local model
        self.local_model_optimizer.zero_grad()
        params = list(self.local_model.parameters())
        if self.partial_update:
            temp_grad = torch.zeros_like(params[0]).to(params[0].device)
            temp_grad[:, self.sampled_idx] = (
                self.local_gradient[0] - self.random_masks[0]
            ).to(params[0].device)
        else:
            temp_grad = (self.local_gradient[0] - self.random_masks[0]).to(
                params[0].device
            )
        params[0].grad = temp_grad
        params[1].grad = (self.local_gradient[1] - self.random_masks[1]).to(
            params[1].device
        )
        params[0].grad = params[0].grad / self.local_batch_size
        params[1].grad = params[1].grad / self.local_batch_size

        """
        self.local_model_optimizer.zero_grad()
        params = list(self.local_model.parameters())
        params[0].grad = (self.local_gradient[0] - self.random_masks[0]).to(
            params[0].device
        )
        params[1].grad = (self.local_gradient[1] - self.random_masks[1]).to(
            params[1].device
        )
        params[0].grad = params[0].grad / self.local_batch_size
        params[1].grad = params[1].grad / self.local_batch_size
        self.local_model_optimizer.step()
        """
