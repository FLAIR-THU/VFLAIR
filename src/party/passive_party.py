import sys, os
sys.path.append(os.pardir)
from torch.utils.data import DataLoader
from party.party import Party
from party.llm_party import Party as Party_LLM
from dataset.party_dataset import PassiveDataset, PassiveDataset_LLM
from dataset.party_dataset import ActiveDataset, ActiveDataset_LLM


class PassiveParty(Party):
    def __init__(self, args, index):
        super().__init__(args, index)

    def prepare_data(self, args, index):
        super().prepare_data(args, index)
        # self.train_dst = TensorDataset(train_inputs, train_masks) # the second label is just a place holder
        # self.test_dst = TensorDataset(test_inputs, test_masks) # the second label is just a place holder
        
        self.train_dst = PassiveDataset(self.train_data)
        self.test_dst = PassiveDataset(self.test_data)
        if self.args.need_auxiliary == 1:
            self.aux_dst = ActiveDataset(self.aux_data, self.aux_label)

class PassiveParty_LLM(Party_LLM):
    def __init__(self, args, index):
        super().__init__(args, index)

    def prepare_data(self, args, index):
        super().prepare_data(args, index)
        self.train_dst = PassiveDataset_LLM(args, self.train_data)

        print('Passive self.train_dst:',len(self.train_dst), type(self.train_dst[0]), type(self.train_dst[1]) )

        self.test_dst = PassiveDataset_LLM(args,self.test_data)

        # self.train_dst = PassiveDataset(self.train_data)
        # self.test_dst = PassiveDataset(self.test_data)

        # if self.args.need_auxiliary == 1:
        #     self.aux_dst = ActiveDataset(self.aux_data, self.aux_label)
            # self.aux_loader = DataLoader(self.aux_dst, batch_size=batch_size,shuffle=True)