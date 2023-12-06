import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader


class SimpleDataset(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item_idx):
        data_i, target_i = self.data[item_idx], self.labels[item_idx]
        return torch.tensor(data_i.clone().detach(), dtype=torch.float32), torch.tensor(target_i.clone().detach(), dtype=torch.long)


class PassiveDataset_LLM(Dataset):
    def __init__(self, args, texts ,labels):
        '''
        texts: np.array
        '''
        self.args = args
        self.texts = []
        self.masks = []
        self.token_type_ids = []


        if len( texts.shape) == 1: # input: single sentence
            for _text in texts:
                ids = args.tokenizer(_text, truncation=True, max_length=args.max_sequence, padding='max_length',return_tensors="pt")                                        
                self.texts.append( torch.tensor(ids['input_ids']).squeeze() )
                self.masks.append( torch.tensor(ids['attention_mask']).squeeze() )
                self.token_type_ids.append( torch.tensor(ids['token_type_ids']).squeeze() )

        elif len( texts.shape) == 2: # input: sentence pairs
            for _text in texts:
                try:
                    ids = args.tokenizer(_text[0],_text[1], padding='max_length',  # Pad to max_length
                                truncation='longest_first',  # Truncate to max_length
                                max_length=args.max_sequence,  
                                return_tensors='pt')
                except:
                    print(type(_text[0]),_text[0],type(_text[1]),_text[1])
                    print( '_text:',type(_text),len(_text) )
                    assert 1>2
                self.texts.append( torch.tensor(ids['input_ids']).squeeze() )
                self.masks.append( torch.tensor(ids['attention_mask']).squeeze() )
                self.token_type_ids.append( torch.tensor(ids['token_type_ids']).squeeze() )

        self.texts=torch.tensor( [aa.tolist() for aa in self.texts] )#.to(args.device)

        self.masks=torch.tensor( [aa.tolist() for aa in self.masks] )#.to(args.device)

        self.token_type_ids=torch.tensor( [aa.tolist() for aa in self.token_type_ids] )#.to(args.device)

        self.labels = torch.tensor(labels) #.to(args.device)
        # print('PassiveDataset_LLM with data/label:', self.texts.shape,self.masks.shape, self.labels.shape)

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, item_idx):
        data_i, target_i , mask_i, token_type_ids_i =\
            self.texts[item_idx], self.labels[item_idx], self.masks[item_idx], self.token_type_ids[item_idx]

        if self.args.num_classes == 1:
            return torch.tensor(data_i.clone().detach(), dtype=torch.long),\
                torch.tensor(target_i.clone().detach(), dtype=torch.float32),\
                torch.tensor(mask_i.clone().detach(), dtype=torch.long),\
                torch.tensor(token_type_ids_i.clone().detach(), dtype=torch.long) #torch.float32
        else:
            return torch.tensor(data_i.clone().detach(), dtype=torch.long),\
                torch.tensor(target_i.clone().detach(), dtype=torch.long),\
                torch.tensor(mask_i.clone().detach(), dtype=torch.long),\
                torch.tensor(token_type_ids_i.clone().detach(), dtype=torch.long) #torch.float32


class PassiveDataset(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item_idx):
        data_i= self.data[item_idx]
        return torch.tensor(data_i, dtype=torch.float32), torch.tensor([]*data_i.size()[0])


class ActiveDataset(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item_idx):
        data_i, target_i = self.data[item_idx], self.labels[item_idx]
        return torch.tensor(data_i.clone().detach(), dtype=torch.float32), torch.tensor(target_i.clone().detach(), dtype=torch.long)


class SimpleTwoPartyDataset(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, data_a, data_b, labels):
        self.data_a = data_a
        self.data_b = data_b
        self.labels = labels

    def __len__(self):
        return len(self.data_a)

    def __getitem__(self, item_idx):
        data_a_i, data_b_i, target_i = self.data_a[item_idx], self.data_b[item_idx], self.labels[item_idx]
        return (torch.tensor(data_a_i).float(), torch.tensor(data_b_i).float()), \
               torch.tensor(target_i.numpy(), dtype=torch.long)


def get_dataloaders(train_dataset: SimpleTwoPartyDataset, valid_dataset: SimpleTwoPartyDataset, batch_size=32,
                    num_workers=1):
    mnist_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    mnist_valid_loader = None
    if valid_dataset is not None:
        mnist_valid_loader = DataLoader(valid_dataset, batch_size=batch_size * 2, shuffle=True, num_workers=num_workers)
    return mnist_train_loader, mnist_valid_loader
