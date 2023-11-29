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

# class PassiveDataset_LLM(Dataset):
#     def __init__(self, args, texts ,labels):
#         '''
#         texts: np.array
#         '''
#         self.texts = []
#         self.masks = []

#         for _text in texts:
#             ids = args.tokenizer(_text, truncation=True, max_length=args.max_sequence, padding='max_length',return_tensors="pt")                                        
#             self.texts.append( torch.tensor(ids['input_ids']).squeeze() )
#             self.masks.append( torch.tensor(ids['attention_mask']) )

#         self.texts=torch.tensor( [aa.tolist() for aa in self.texts] )#.to(args.device)

#         self.masks=torch.tensor( [aa.tolist() for aa in self.masks] )#.to(args.device)

#         self.labels = torch.tensor(labels) #.to(args.device)
#         # print('PassiveDataset_LLM with data/label:', self.texts.shape,self.masks.shape, self.labels.shape)

#     def __len__(self):
#         return len(self.labels)


#     def __getitem__(self, item_idx):
#         data_i, target_i , mask_i= self.texts[item_idx], self.labels[item_idx], self.masks[item_idx]
#         return torch.tensor(data_i.clone().detach(), dtype=torch.float32),torch.tensor(target_i.clone().detach(), dtype=torch.long)
#     #torch.tensor(mask_i.clone().detach(), dtype=torch.float32), \


class PassiveDataset_LLM(Dataset):
    def __init__(self, args, texts ,labels):
        '''
        texts: np.array
        '''
        self.texts = []
        self.masks = []

        for _text in texts:
            ids = args.tokenizer(_text, truncation=True, max_length=args.max_sequence, padding='max_length',return_tensors="pt")                                        
            self.texts.append( torch.tensor(ids['input_ids']).squeeze() )
            self.masks.append( torch.tensor(ids['attention_mask']) )

        self.texts=torch.tensor( [aa.tolist() for aa in self.texts] )#.to(args.device)

        self.masks=torch.tensor( [aa.tolist() for aa in self.masks] )#.to(args.device)

        self.labels = torch.tensor(labels) #.to(args.device)
        # print('PassiveDataset_LLM with data/label:', self.texts.shape,self.masks.shape, self.labels.shape)

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, item_idx):
        data_i, target_i , mask_i= self.texts[item_idx], self.labels[item_idx], self.masks[item_idx]
        return torch.tensor(data_i.clone().detach(), dtype=torch.float32),\
            torch.tensor(target_i.clone().detach(), dtype=torch.long),\
            torch.tensor(mask_i.clone().detach(), dtype=torch.float32)


# class PassiveDataset_LLM(Dataset):
#     def __init__(self, args, texts):
        
#         self.texts = []
#         for _text in texts:
#             ids = args.tokenizer(_text, truncation=True, max_length=args.max_sequence, padding='max_length',return_tensors="pt")                                        
#             self.texts.append( torch.tensor(ids['input_ids']).squeeze() )
#         self.texts=torch.tensor( [aa.tolist() for aa in self.texts] )#.to(args.device)

#         self.labels = None
#         print('PassiveDataset_LLM texts:',self.texts.shape)


#     def __len__(self):
#         return len(self.texts)

#     def get_batch_texts(self, idx):
#         # Fetch a batch of inputs
#         return self.texts[idx]

#     def __getitem__(self, item_idx):
#         data_i= self.texts[item_idx]
#         return torch.tensor(data_i, dtype=torch.float32), torch.tensor([]*data_i.size()[0])

#     # def __getitem__(self, idx):
#     #     batch_texts = self.get_batch_texts(idx)
#     #     return batch_texts


# class ActiveDataset_LLM(Dataset):
#     def __init__(self, args, texts ,labels):
#         '''
#         texts: np.array
#         '''
#         self.texts = []
#         self.masks = []

#         for _text in texts:
#             ids = args.tokenizer(_text, truncation=True, max_length=args.max_sequence, padding='max_length',return_tensors="pt")                                        
#             self.texts.append( torch.tensor(ids['input_ids']).squeeze() )
#             self.masks.append( torch.tensor(ids['attention_mask']) )

#         self.texts=torch.tensor( [aa.tolist() for aa in self.texts] )#.to(args.device)

#         self.masks=torch.tensor( [aa.tolist() for aa in self.masks] )#.to(args.device)

#         self.labels = torch.tensor(labels) #.to(args.device)
#         print('ActiveDataset_LLM texts:', self.texts.shape,self.masks.shape, self.labels.shape)

#     def __len__(self):
#         return len(self.labels)


#     def __getitem__(self, item_idx):
#         data_i, target_i , mask_i= self.texts[item_idx], self.labels[item_idx], self.masks[item_idx]
#         return torch.tensor(data_i.clone().detach(), dtype=torch.float32),torch.tensor(target_i.clone().detach(), dtype=torch.long)
#     #torch.tensor(mask_i.clone().detach(), dtype=torch.float32), \


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
