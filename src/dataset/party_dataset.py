import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import string 

from utils.squad_utils import normalize_answer

choices = ["A", "B", "C", "D"]
def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = len(choices)
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx][str(choices[j])])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx]['answer']) #df.iloc[idx, k + 1]
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt_list = []
    if k == -1:
        k = train_df.shape[0]
    # print('k=',k)
    
    for i in range(k):
        prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
        if train_df.iloc[i]['subject'] == subject:
            prompt += format_example(train_df, i)

    prompt_list.append(prompt)
    return prompt_list


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
        self.texts = [] # input_ids
        self.masks = [] # attention_mask
        self.token_type_ids = [] # token_type_ids
        self.labels = []
        self.doc_tokens = []
        self.features = []

        ### special treatment str labels(word prediction)
        if args.task_type == 'QuestionAnswering':
            # labels : bs * [start_position, end_position]
            # texts: bs * [feature]
            for i in range(len(texts)): 
                _feature = texts[i]
                self.texts.append(_feature["input_ids"]) # input_ids
                self.masks.append(_feature["input_mask"]) # input_mask
                self.token_type_ids.append(_feature["segment_ids"]) # segment_ids

                self.labels.append( labels[i] ) # [ [start_position, end_position] ]
                self.doc_tokens.append(  _feature["tokens"] ) # [  _feature.tokens ]
                self.features.append(_feature)
            return

        # print('label:',type(labels[0]),labels[0])
        if type(labels[0]) == str:
            labels = [ args.tokenizer.convert_tokens_to_ids(_label) for _label in labels] 

        ### normal treatment ###
        if len( texts.shape) == 1: # input: single sentence
            for _text in texts:
                # ids = args.tokenizer(_text, truncation=True, max_length=args.max_sequence, padding='max_length',return_tensors="pt")                                        

                ids = args.tokenizer(_text, return_tensors="pt")                                        
                self.texts.append( torch.tensor(ids['input_ids']).squeeze() )
                self.masks.append( torch.tensor(ids['attention_mask']).squeeze() )
                if 'token_type_ids' in list(ids.keys()):
                    self.token_type_ids.append( torch.tensor(ids['token_type_ids']).squeeze() )

        elif len( texts.shape) == 2: # input: sentence pairs
            
            for _text in texts:
                try:
                    ids = args.tokenizer(_text[0],_text[1], padding='max_length', # Pad to max_length
                                truncation='longest_first',  # Truncate to max_length
                                max_length=args.max_sequence,  
                                return_tensors='pt')
                except:
                    assert 1>2
                self.texts.append( torch.tensor(ids['input_ids']).squeeze() )
                self.masks.append( torch.tensor(ids['attention_mask']).squeeze() )
                if 'token_type_ids' in list(ids.keys()):
                    self.token_type_ids.append( torch.tensor(ids['token_type_ids']).squeeze() )
        
        elif len( texts.shape) == 3: # input: sentence pairs
            for _text in texts:
                try:
                    ids = args.tokenizer( list(_text[0]), list(_text[1]), padding='max_length',  # Pad to max_length
                                truncation='longest_first',  # Truncate to max_length
                                max_length=args.max_sequence,  
                                return_tensors='pt')
                    # ids = args.tokenizer( list(_text[0]), list(_text[1]), truncation=False)
                except:
                    assert 1>2
                self.texts.append( torch.tensor(ids['input_ids']).squeeze() )
                self.masks.append( torch.tensor(ids['attention_mask']).squeeze() )
                if 'token_type_ids' in list(ids.keys()):
                    self.token_type_ids.append( torch.tensor(ids['token_type_ids']).squeeze() )

        else:
            print(texts.shape)
            assert 1>2, 'text input shape not supported'

        self.labels = torch.tensor(labels) 

        self.texts=[aa.tolist() for aa in self.texts] 

        self.masks=[aa.tolist() for aa in self.masks] 

        if self.token_type_ids != []:
            self.token_type_ids=[aa.tolist() for aa in self.token_type_ids]


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, item_idx):
        data_i, target_i , mask_i =\
            self.texts[item_idx], self.labels[item_idx], self.masks[item_idx]
        data_i = torch.tensor(data_i, dtype=torch.long).to(self.args.device)
        mask_i = torch.tensor(mask_i, dtype=torch.long).to(self.args.device)
        if self.args.num_classes == 1:
            target_i = torch.tensor(target_i, dtype=torch.float32).to(self.args.device)
        else:
            target_i = torch.tensor(target_i, dtype=torch.long).to(self.args.device)


        if self.token_type_ids == []:
            token_type_ids_i = []
        else:
            token_type_ids_i = self.token_type_ids[item_idx]
            token_type_ids_i = torch.tensor(token_type_ids_i, dtype=torch.long).to(self.args.device)

        if self.doc_tokens == []:
            doc_tokens_i = []
        else:
            doc_tokens_i = self.doc_tokens[item_idx]
        
        if self.features == []:
            features_i = []
        else:
            features_i = self.features[item_idx]
  
        return data_i, target_i, mask_i,  token_type_ids_i, features_i #doc_tokens_i
        
          

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
