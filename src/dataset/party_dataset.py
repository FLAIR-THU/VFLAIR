import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import string 
from utils.squad_utils import normalize_answer
from random import randrange 

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
        self.features = []

        if args.task_type == 'QuestionAnswering':
            # labels : bs * [start_position, end_position]
            # texts: bs * [feature]
            for i in range(len(texts)): 
                _feature = texts[i]
                self.texts.append(_feature["input_ids"]) # input_ids
                self.masks.append(_feature["input_mask"]) # input_mask
                self.token_type_ids.append(_feature["segment_ids"]) # segment_ids

                self.labels.append( labels[i] ) # [ [start_position, end_position] ]
                self.features.append( _feature ) 

            # print('self.features:',type(self.features), len(self.features))
            # print(type(self.features[0]), type(self.features[1]))

        elif args.task_type == 'CausalLM':
            for _text in texts:
                ids = args.tokenizer(_text, \
                    padding=args.padding,truncation=args.truncation ,\
                    max_length=args.max_length,return_tensors='pt') 
            
                self.texts.append( torch.tensor(ids['input_ids']).squeeze() )
                self.masks.append( torch.tensor(ids['attention_mask']).squeeze() )
                if 'token_type_ids' in list(ids.keys()):
                    self.token_type_ids.append( torch.tensor(ids['token_type_ids']).squeeze() )

            self.labels = labels
            self.texts=[aa.tolist() for aa in self.texts] 
            self.masks=[aa.tolist() for aa in self.masks] 
            if self.token_type_ids != []:
                self.token_type_ids=[aa.tolist() for aa in self.token_type_ids]

        elif args.task_type == 'SequenceClassification':
            if len( texts.shape) == 1: # input: single sentence
                for _text in texts:
                    # flag+=1
                    # ids = args.tokenizer(_text, \
                    #     padding=args.padding,truncation=args.truncation ,\
                    #     max_length=args.max_length,return_tensors='pt',add_special_tokens=args.add_special_tokens)
                    # print('std attention_mask:',torch.tensor(ids['attention_mask']).squeeze().shape, torch.tensor(ids['attention_mask']).squeeze())
                    # print('std token_type_ids:',torch.tensor(ids['token_type_ids']).squeeze().shape, torch.tensor(ids['token_type_ids']).squeeze())
                    # decode_text = [args.tokenizer.decode([_tok]) for _tok in torch.tensor(ids['input_ids']).squeeze().tolist() ]
                    # decode_text = " ".join(decode_text)
                    # print('decode_text:',decode_text)

                    if args.padding != "do_not_pad" and args.padding_type == "inside": # [PAD] between [CLS][SEP]
                        text_tokens = args.tokenizer.tokenize(_text)

                        pad_length = max( args.max_length - len(text_tokens), 0 )
                        for _pad in range(pad_length):
                            if args.padding_side == 'right':
                                text_tokens.append( args.tokenizer.pad_token )
                            elif args.padding_side == 'left':
                                text_tokens.insert(0, args.tokenizer.pad_token )
                            elif args.padding_side == 'random':
                                text_tokens.insert(randrange(len(text_tokens)+1), args.tokenizer.pad_token )

                        _text = " ".join(text_tokens)
                        # print('after pad:', _text)

                        ids = args.tokenizer(_text, truncation=args.truncation ,max_length=args.max_length,\
                        return_tensors='pt',add_special_tokens=args.add_special_tokens)

                        # for _pos in range(ids['attention_mask'].shape[1]):
                        #     if ids['input_ids'][0][_pos] == args.tokenizer.pad_token_id:
                        #         ids['attention_mask'][0][_pos] = 0

                    else: # [PAD] outside [CLS][SEP]
                        ids = args.tokenizer(_text, \
                        padding=args.padding,truncation=args.truncation ,\
                        max_length=args.max_length,return_tensors='pt',add_special_tokens=args.add_special_tokens)
                   
                    # decode_text = [args.tokenizer.decode([_tok]) for _tok in torch.tensor(ids['input_ids']).squeeze().tolist() ]
                    # decode_text = " ".join(decode_text)
                    # print('decode_text:',decode_text)
                    # print('attention_mask:',torch.tensor(ids['attention_mask']).squeeze().shape, torch.tensor(ids['attention_mask']).squeeze())
                    # print('token_type_ids:',torch.tensor(ids['token_type_ids']).squeeze().shape, torch.tensor(ids['token_type_ids']).squeeze())

                    self.texts.append( torch.tensor(ids['input_ids']).squeeze() )
                    # avoid performing attention on padding token indices.
                    self.masks.append( torch.tensor(ids['attention_mask']).squeeze() )
                    # Segment token indices to indicate first and second portions of the inputs.
                    if 'token_type_ids' in list(ids.keys()):
                        self.token_type_ids.append( torch.tensor(ids['token_type_ids']).squeeze() )
                    
            elif len( texts.shape) == 2: # input: sentence pairs
                for _text in texts:

                    # if args.padding != "do_not_pad" and args.padding_type == "inside": # [PAD] between [CLS][SEP]
                    #     text_tokens_0 = args.tokenizer.tokenize(_text[0])
                    #     text_tokens_1 = args.tokenizer.tokenize(_text[1])


                    #     pad_length = max( args.max_length - len(text_tokens), 0 )
                    #     for _pad in range(pad_length):
                    #         if args.padding_side == 'right':
                    #             text_tokens.append( args.tokenizer.pad_token )
                    #         elif args.padding_side == 'left':
                    #             text_tokens.insert(0, args.tokenizer.pad_token )
                    #         elif args.padding_side == 'random':
                    #             text_tokens.insert(randrange(len(text_tokens)+1), args.tokenizer.pad_token )

                    #     _text = " ".join(text_tokens)
                    #     # print('after pad:', _text)

                    #     ids = args.tokenizer(_text, truncation=args.truncation ,max_length=args.max_length,\
                    #     return_tensors='pt',add_special_tokens=args.add_special_tokens)

                    #     # for _pos in range(ids['attention_mask'].shape[1]):
                    #     #     if ids['input_ids'][0][_pos] == args.tokenizer.pad_token_id:
                    #     #         ids['attention_mask'][0][_pos] = 0

                    # else: # [PAD] outside [CLS][SEP]
                    #     ids = args.tokenizer(_text[0],_text[1], \
                    #     padding=args.padding,truncation=args.truncation ,\
                    #     max_length=args.max_length,return_tensors='pt',add_special_tokens=args.add_special_tokens)
                   
                    try:
                        ids = args.tokenizer(_text[0],_text[1], \
                        padding=args.padding,truncation=args.truncation ,\
                        max_length=args.max_length,return_tensors='pt')
                    except:
                        assert 1>2

                    self.texts.append( torch.tensor(ids['input_ids']).squeeze() )
                    self.masks.append( torch.tensor(ids['attention_mask']).squeeze() )
                    if 'token_type_ids' in list(ids.keys()):
                        self.token_type_ids.append( torch.tensor(ids['token_type_ids']).squeeze() )
            
            elif len( texts.shape) == 3: # input: sentence pairs
                for _text in texts:
                    try:
                        ids = args.tokenizer( list(_text[0]), list(_text[1]), \
                        padding=args.padding,truncation=args.truncation ,\
                        max_length=args.max_length,return_tensors='pt')
                       
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
        
        if not type(target_i) == str:
            if self.args.num_classes == 1:
                target_i = torch.tensor(target_i, dtype=torch.float32).to(self.args.device)
            else:
                target_i = torch.tensor(target_i, dtype=torch.long).to(self.args.device)

        if self.token_type_ids == []:
            token_type_ids_i = []
        else:
            token_type_ids_i = self.token_type_ids[item_idx]
            token_type_ids_i = torch.tensor(token_type_ids_i, dtype=torch.long).to(self.args.device)
        
        if self.features == []:
            features_i = []
        else:
            features_i = self.features[item_idx]

        # if data_i.shape[1] > self.args.max_sequence:
        # print('fetch data:',data_i.shape, target_i, mask_i.shape)
        # print(type(token_type_ids_i), token_type_ids_i, '  ',type(features_i), features_i )

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
