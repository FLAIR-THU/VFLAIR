import sys, os
sys.path.append(os.pardir)

from party.party import Party
from dataset.party_dataset import PassiveDataset

class PassiveParty(Party):
    def __init__(self, args, index):
        super().__init__(args, index)

    def prepare_data(self, args, index):
        super().prepare_data(args, index)
        self.train_dst = PassiveDataset(self.train_data)
        self.test_dst = PassiveDataset(self.test_data)