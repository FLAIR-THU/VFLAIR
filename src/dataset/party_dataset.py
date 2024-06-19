import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import string
from utils.squad_utils import normalize_answer
from random import randrange
from typing import List, Dict, Optional, Sequence, Union
import copy


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
        return torch.tensor(data_i.clone().detach(), dtype=torch.float32), torch.tensor(target_i.clone().detach(),
                                                                                        dtype=torch.long)


class PassiveDataset_LLM(Dataset):
    def __init__(self, args, texts: Union[np.array, List[Dict]] ,labels, split_name='test'):
        '''
        texts: np.array
        '''
        self.args = args
        self.labels = []
        self.features = []
        self.input_dicts = []

        if args.task_type == 'SequenceClassification':
            for _text in texts:
                if len(texts.shape) == 1:  # input: single sentence
                    if args.padding != "do_not_pad" and args.padding_type == "inside":  # [PAD] between [CLS][SEP]
                        text_tokens = args.tokenizer.tokenize(_text)

                        pad_length = max(args.max_length - len(text_tokens), 0)
                        for _pad in range(pad_length):
                            if args.padding_side == 'right':
                                text_tokens.append(args.tokenizer.pad_token)
                            elif args.padding_side == 'left':
                                text_tokens.insert(0, args.tokenizer.pad_token)
                            elif args.padding_side == 'random':
                                text_tokens.insert(randrange(len(text_tokens) + 1), args.tokenizer.pad_token)

                        _text = " ".join(text_tokens)
                        # print('after pad:', _text)

                        ids = args.tokenizer(_text, truncation=args.truncation, max_length=args.max_length, \
                                             return_tensors='pt', add_special_tokens=args.add_special_tokens)

                        # for _pos in range(ids['attention_mask'].shape[1]):
                        #     if ids['input_ids'][0][_pos] == args.tokenizer.pad_token_id:
                        #         ids['attention_mask'][0][_pos] = 0
                    else:  # [PAD] outside [CLS][SEP]
                        ids = args.tokenizer(_text, \
                                             padding=args.padding, truncation=args.truncation, \
                                             max_length=args.max_length, return_tensors='pt',
                                             add_special_tokens=args.add_special_tokens)
                elif len(texts.shape) > 1:  # input: sentence pairs
                    try:
                        ids = args.tokenizer(_text[0], _text[1], \
                                             padding=args.padding, truncation=args.truncation, \
                                             max_length=args.max_length, return_tensors='pt')
                    except:
                        assert 1 > 2
                else:
                    print(texts.shape)
                    assert 1 > 2, 'text input shape not supported'

                self.input_dicts.append(ids)

            print(type(labels[:2]), labels[:2])
            if self.args.num_classes == 1:
                self.labels = torch.tensor(labels, dtype=torch.float32)
            else:
                self.labels = torch.tensor(labels)

        elif args.task_type == 'CausalLM':
            if isinstance(texts[0], Dict):
                self.input_dicts = texts
                self.labels = labels
                return
            if split_name == 'test':
                if isinstance(texts[0],Dict):
                    self.input_dicts=texts
                    self.labels=labels
                else:
                    for i in range(len(texts)):
                        ids = args.tokenizer(texts[i], \
                        padding=args.padding,truncation=args.truncation ,\
                        max_length=args.max_length,return_tensors='pt')

                        if i == 0:
                            print('TEXT:',texts[i])
                            print('text_id:',ids['input_ids'].shape, ids['input_ids'])
                            print('label:',labels[i],args.tokenizer.convert_tokens_to_ids( labels[i] ) )
                            print('-'*25)

                        self.labels.append( args.tokenizer.convert_tokens_to_ids( labels[i] ) )
                        self.input_dicts.append(ids)
            else:
                for i in range(len(texts)):
                    ids = args.tokenizer(texts[i], \
                    padding=args.padding,truncation=args.truncation ,\
                    max_length=args.max_length,return_tensors='pt')

                    if i == 0:
                        print('TEXT:',texts[i])
                        print('text_id:',ids['input_ids'].shape, ids['input_ids'])
                        print('label:',labels[i],args.tokenizer.convert_tokens_to_ids( labels[i] ) )
                        print('-'*25)

                    self.labels.append(ids['input_ids'])#args.tokenizer.convert_tokens_to_ids( labels[i] ) )
                    self.input_dicts.append(ids)
        
        elif args.task_type == 'QuestionAnswering':
            # labels : bs * [start_position, end_position]
            # texts: bs * [feature]
            for i in range(len(texts)):
                _feature = texts[i]
                inputs = {
                    'input_ids': _feature["input_ids"],
                    'attention_mask': _feature["input_mask"],
                    'token_type_ids': _feature["segment_ids"],

                    'feature': {
                        'token_to_orig_map': _feature["token_to_orig_map"],
                        'token_is_max_context': _feature["token_is_max_context"],
                        'len_tokens': len(_feature["tokens"])
                    }
                }
                self.input_dicts.append(inputs)
                self.labels.append( labels[i] )
                # test: [ [start_position, end_position] ]
                # train: [ [ [start_position1,start_position2,start_position3],[end_position1,end_position2,end_position3] ] ]

            print(f'---- {split_name} -----')
            print('self.labels:',self.labels[:3])
            # self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item_idx):

        input_dict = self.input_dicts[item_idx]

        for key_name in input_dict.keys():
            if not isinstance( input_dict[key_name] , dict):
                input_dict[key_name] = torch.tensor(input_dict[key_name]).squeeze().to(self.args.device)

        label = torch.tensor(self.labels[item_idx]).squeeze().to(self.args.device)
        return input_dict, label


class LambadaDataset_LLM(Dataset):
    def __init__(self, args, texts ,labels, split_name='test'):
        '''
        texts: np.array
        '''
        self.args = args
        self.labels = []
        self.features = []
        self.input_dicts = []

        if split_name == 'test':
            for i in range(len(texts)):
                ids = args.tokenizer(texts[i], \
                padding=args.padding,truncation=args.truncation ,\
                max_length=args.max_length,return_tensors='pt')

                self.labels.append( args.tokenizer.convert_tokens_to_ids( labels[i] ) )
                self.input_dicts.append(ids)

                # if i == 0:
                #     print('TEST TEXT:',texts[i])
                #     print('text_id:',ids['input_ids'].shape, ids['input_ids'])
                #     print('label:',self.labels[i] )
                #     print('-'*25)
        else:
            for i in range(len(texts)):
                ids = args.tokenizer(texts[i], \
                padding=args.padding,truncation=args.truncation ,\
                max_length=args.max_length,return_tensors='pt')

                self.labels.append(ids['input_ids'])#args.tokenizer.convert_tokens_to_ids( labels[i] ) )
                self.input_dicts.append(ids)

                # if i == 0:
                #     print('TRAIN TEXT:',texts[i])
                #     print('text_id:',ids['input_ids'].shape, ids['input_ids'])
                #     print('label:',self.labels[i], args.tokenizer.convert_tokens_to_ids( labels[i] ) )
                #     print('-'*25)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item_idx):

        input_dict = self.input_dicts[item_idx]

        for key_name in input_dict.keys():
            if not isinstance(input_dict[key_name], dict):
                input_dict[key_name] = torch.tensor(input_dict[key_name]).squeeze().to(self.args.device)

        label = torch.tensor(self.labels[item_idx]).squeeze().to(self.args.device)
        return input_dict, label



class MMLUDataset_LLM(Dataset):
    def __init__(self, args, texts, labels, split_name='test'):
        '''
        texts: np.array
        '''
        self.args = args
        self.labels = []
        self.features = []
        self.input_dicts = []

        # if args.task_type == 'CausalLM':
        for i in range(len(texts)):
            ids = args.tokenizer(texts[i], \
                                 padding=args.padding, truncation=args.truncation, \
                                 max_length=args.max_length, return_tensors='pt')

            if i == 0:
                print('TEXT:', texts[i])
                print('text_id:', ids['input_ids'].shape, ids['input_ids'])
                print('label:', labels[i], args.tokenizer.convert_tokens_to_ids(labels[i]))
                print('-' * 25)

            self.labels.append(args.tokenizer.convert_tokens_to_ids(labels[i]))
            self.input_dicts.append(ids)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item_idx):

        input_dict = self.input_dicts[item_idx]

        for key_name in input_dict.keys():
            if not isinstance(input_dict[key_name], dict):
                input_dict[key_name] = torch.tensor(input_dict[key_name]).squeeze().to(self.args.device)

        label = torch.tensor(self.labels[item_idx]).squeeze().to(self.args.device)
        return input_dict, label


class AlpacaDataset_LLM(Dataset):
    def __init__(self, args, sources, targets, split_name='train'):
        '''
        texts: np.array
        '''
        self.args = args

        IGNORE_INDEX = args.tokenizer.pad_token_id  # -100

        def _tokenize_fn(strings: Sequence[str], tokenizer) -> Dict:
            """Tokenize a list of strings."""

            tokenized_list = [
                tokenizer(
                    text,
                    return_tensors="pt",
                    padding=args.padding,  # "longest",
                    max_length=args.max_length,  # tokenizer.model_max_length,
                    truncation=args.truncation,  # True,
                )
                for text in strings
            ]

            input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
            attention_mask = [tokenized.attention_mask[0] for tokenized in tokenized_list]

            input_ids_lens = labels_lens = [
                tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
            ]

            return dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                input_ids_lens=input_ids_lens,
                labels_lens=labels_lens,
            )

        def preprocess(
                sources: Sequence[str],
                targets: Sequence[str],
                tokenizer,
        ) -> Dict:
            """Preprocess the data by tokenizing."""
            if split_name == 'train':
                examples = [s + t for s, t in zip(sources, targets)]
                examples_tokenized, targets_tokenized = [_tokenize_fn(strings, tokenizer) for strings in
                                                         (examples, targets)]

                input_ids = examples_tokenized["input_ids"]
                attention_mask = examples_tokenized["attention_mask"]

                labels = copy.deepcopy(input_ids)
                for label, target_len in zip(labels, targets_tokenized["input_ids_lens"]):
                    label[:-target_len] = IGNORE_INDEX
            else:
                inputs_tokenized, targets_tokenized = [_tokenize_fn(strings, tokenizer) for strings in
                                                       (sources, targets)]
                input_ids = inputs_tokenized["input_ids"]
                attention_mask = inputs_tokenized["attention_mask"]
                labels = targets_tokenized["input_ids"]
            # input_ids: prompt +  target
            # label: masked_prompt + target
            return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        data_dict = preprocess(sources, targets, args.tokenizer)

        self.input_dicts = [
            {'input_ids': data_dict['input_ids'][i], 'attention_mask': data_dict['attention_mask'][i]}
            for i in range(len(data_dict['input_ids']))
        ]  # list of input_dicts
        self.labels = data_dict["labels"]  # list of tensor(labels)

        # print(f'=== Dataset Split = {split_name} ===')
        # print('text:',self.input_dicts[0]['input_ids'].shape, self.args.tokenizer.decode(self.input_dicts[0]['input_ids']))
        # print('-'*25)
        # print('label:',self.labels[0].shape, self.args.tokenizer.decode(self.labels[0]))
        # print('='*50)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item_idx):

        input_dict = self.input_dicts[item_idx]

        for key_name in input_dict.keys():
            if not isinstance( input_dict[key_name] , dict):
                input_dict[key_name] = torch.tensor(input_dict[key_name]).squeeze().to(self.args.device)

        label = torch.tensor(self.labels[item_idx]).squeeze().to(self.args.device)
        return input_dict, label

class GSMDataset_LLM(Dataset):
    def __init__(self, args, qns, ans, split_name='train', loss_on_prefix=True):
        self.args=args
        self.qns = qns
        self.ans = ans
        self.loss_on_prefix = loss_on_prefix

        if split_name == 'train':
            self.input_dicts = [ self.args.tokenizer(
                                    self.qns[i]+self.ans[i],return_tensors="pt",
                                    padding=args.padding,#"longest",
                                    max_length=args.max_length, #tokenizer.model_max_length,
                                    truncation=args.truncation, #True,
                                )
                for i in range(len(self.ans))
            ]
            self.labels = [ _input['input_ids'] for _input in self.input_dicts ]
            # print('--- train ---')
            # print(self.input_dicts[0])
            # print(self.labels[0])
        else:
            self.input_dicts = [ self.args.tokenizer(
                                    self.qns[i],return_tensors="pt",
                                    padding=args.padding,#"longest",
                                    max_length=args.max_length, #tokenizer.model_max_length,
                                    truncation=args.truncation, #True,
                                )
                for i in range(len(self.ans))
            ]
            self.labels = [ self.args.tokenizer(
                                    self.ans[i],return_tensors="pt",
                                    padding=args.padding,#"longest",
                                    max_length=args.max_length, #tokenizer.model_max_length,
                                    truncation=args.truncation, #True,
                                )['input_ids']
                for i in range(len(self.ans))
            ]

            # print('--- test ---')
            # print(self.input_dicts[0])
            # print(self.labels[0])

        print(f'=== Dataset Split = {split_name} ===')
        print('text:',self.input_dicts[0]['input_ids'].shape)
        print(self.args.tokenizer.decode( self.input_dicts[0]['input_ids'].squeeze()))
        print('-'*25)
        print('label:',self.labels[0].shape)
        print(self.args.tokenizer.decode(self.labels[0].squeeze()))
        print('='*50)


    def __len__(self):
        return len(self.ans)

    def __getitem__(self, item_idx):

        input_dict = self.input_dicts[item_idx]

        for key_name in input_dict.keys():
            if not isinstance( input_dict[key_name] , dict):
                input_dict[key_name] = torch.tensor(input_dict[key_name]).squeeze().to(self.args.device)

        label = torch.tensor(self.labels[item_idx]).squeeze().to(self.args.device)
        return input_dict, label

    # def __getitem__(self, idx):
    #     qn_tokens = self.qns["input_ids"][idx]
    #     ans_tokens = self.ans["input_ids"][idx]
    #     pad_tokens = [0] * (self.max_len - len(qn_tokens) - len(ans_tokens))
    #     tokens = qn_tokens + ans_tokens + pad_tokens
    #     mask = (
    #         ([int(self.loss_on_prefix)] * len(qn_tokens))
    #         + ([1] * len(ans_tokens))
    #         + ([0] * len(pad_tokens))
    #     )
    #     tokens = th.tensor(tokens)
    #     mask = th.tensor(mask)
    #     return dict(input_ids=tokens, attention_mask=mask)

class MATHDataset_LLM(Dataset):
    def __init__(self, args, sources ,targets, split_name='train'):
        '''
        texts: np.array
        '''
        self.args = args
        
        IGNORE_INDEX = args.tokenizer.pad_token_id #-100
        def _tokenize_fn(strings: Sequence[str], tokenizer) -> Dict:
            """Tokenize a list of strings."""

            tokenized_list = [
                tokenizer(
                    text,
                    return_tensors="pt",
                    padding=args.padding,#"longest",
                    max_length=args.max_length, #tokenizer.model_max_length,
                    truncation=args.truncation, #True,
                )
                for text in strings
            ]

            input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
            attention_mask = [tokenized.attention_mask[0] for tokenized in tokenized_list]

            input_ids_lens = labels_lens = [
                tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
            ]

            return dict(
                input_ids=input_ids,
                attention_mask = attention_mask,
                labels=labels,
                input_ids_lens=input_ids_lens,
                labels_lens=labels_lens,
            )

        def preprocess(
            sources: Sequence[str],
            targets: Sequence[str],
            tokenizer,
        ) -> Dict:
            """Preprocess the data by tokenizing."""
            if split_name == 'train':
                examples = [s + t for s, t in zip(sources, targets)]
                examples_tokenized, targets_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, targets)]

                input_ids = examples_tokenized["input_ids"]
                attention_mask = examples_tokenized["attention_mask"]

                labels = copy.deepcopy(input_ids)
                for label, target_len in zip(labels, targets_tokenized["input_ids_lens"]):
                    label[:-target_len] = IGNORE_INDEX
            else:
                inputs_tokenized, targets_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (sources, targets)]
                # inputs_tokenized = _tokenize_fn(sources, tokenizer) 
                
                # targets_tokenized = 

                # targets_tokenized_list = [
                #     args.tokenizer(
                #         text,
                #         return_tensors="pt",
                #         padding=args.padding,#"longest",
                #         max_length=args.max_length, #tokenizer.model_max_length,
                #         truncation=args.truncation, #True,
                #     )
                #     for text in strings
                # ]
                # targets_tokenized = [tokenized.input_ids[0] for tokenized in targets_tokenized_list]
            
                input_ids = inputs_tokenized["input_ids"]
                attention_mask = inputs_tokenized["attention_mask"]
                labels = targets_tokenized["input_ids"]
            # input_ids: prompt +  target
            # label: masked_prompt + target
            return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        data_dict = preprocess(sources, targets, args.tokenizer)

        self.input_dicts = [
                {'input_ids':data_dict['input_ids'][i], 'attention_mask':data_dict['attention_mask'][i]}
                for i in range( len(data_dict['input_ids']) )
            ] # list of input_dicts
        self.labels = data_dict["labels"] # list of tensor(labels)

        print(f'=== Dataset Split = {split_name} ===')
        print('text:',self.input_dicts[0]['input_ids'].shape, self.args.tokenizer.decode(self.input_dicts[0]['input_ids']))
        print('-'*25)
        print('label:',self.labels[0].shape, self.args.tokenizer.decode(self.labels[0]))
        print('='*50)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item_idx):

        input_dict = self.input_dicts[item_idx]

        for key_name in input_dict.keys():
            if not isinstance(input_dict[key_name], dict):
                input_dict[key_name] = torch.tensor(input_dict[key_name]).squeeze().to(self.args.device)

        label = torch.tensor(self.labels[item_idx]).squeeze().to(self.args.device)
        return input_dict, label


class PassiveDataset_LLM_old(Dataset):
    def __init__(self, args, texts, labels, split_name='test'):
        '''
        texts: np.array
        '''
        self.args = args
        self.texts = []  # input_ids
        self.masks = []  # attention_mask
        self.token_type_ids = []  # token_type_ids
        self.labels = []
        self.features = []

        max_token_len = 0

        if args.task_type == 'QuestionAnswering':
            # labels : bs * [start_position, end_position]
            # texts: bs * [feature]
            for i in range(len(texts)):
                _feature = texts[i]
                self.texts.append(_feature["input_ids"])  # input_ids
                self.masks.append(_feature["input_mask"])  # input_mask
                self.token_type_ids.append(_feature["segment_ids"])  # segment_ids

                self.labels.append(labels[i])  # [ [start_position, end_position] ]
                self.features.append(_feature)

                # print('self.features:',type(self.features), len(self.features))
            # print(type(self.features[0]), type(self.features[1]))

        elif args.task_type == 'CausalLM':
            if args.dataset == 'MMLU':
                flag = 0
                for i in range(len(texts)):
                    ids = args.tokenizer(texts[i], \
                                         padding=args.padding, truncation=args.truncation, \
                                         max_length=args.max_length, return_tensors='pt')

                    self.texts.append(torch.tensor(ids['input_ids']).squeeze())
                    self.masks.append(torch.tensor(ids['attention_mask']).squeeze())
                    if 'token_type_ids' in list(ids.keys()):
                        self.token_type_ids.append(torch.tensor(ids['token_type_ids']).squeeze())

                    if flag == 0:
                        print('TEXT:', texts[i])
                        print(self.texts[-1])
                        print('LABEL:', labels[i])

                        print('-' * 25)
                        flag = flag + 1
                self.labels = labels
                self.texts = [aa.tolist() for aa in self.texts]
                self.masks = [aa.tolist() for aa in self.masks]
                if self.token_type_ids != []:
                    self.token_type_ids = [aa.tolist() for aa in self.token_type_ids]

            else:
                flag = 0
                for i in range(len(texts)):

                    if split_name == 'test':
                        ids = args.tokenizer(texts[i], return_tensors='pt')
                    else:
                        # # truncation
                        # if torch.tensor(ids['input_ids']).squeeze().shape[0] > args.max_length:
                        #     ids['input_ids'] = ids['input_ids'][...,-(args.max_length+1):] # torch.size([max_length])
                        #     ids['attention_mask'] = ids['attention_mask'][...,-(args.max_length+1):] # torch.size([max_length])
                        #     if 'token_type_ids' in list(ids.keys()):
                        #         ids['token_type_ids'] = ids['token_type_ids'][...,-(args.max_length+1):] # torch.size([max_length])
                        # # padding
                        # else:
                        ids = args.tokenizer(texts[i], \
                                             padding=args.padding, truncation=args.truncation, \
                                             max_length=args.max_length, return_tensors='pt')

                    self.texts.append(torch.tensor(ids['input_ids']).squeeze())
                    self.masks.append(torch.tensor(ids['attention_mask']).squeeze())
                    # self.labels.append( int(torch.tensor(ids['input_ids']).squeeze()[-1].item()) )
                    self.labels.append(args.tokenizer.convert_tokens_to_ids(labels[i]))

                    if 'token_type_ids' in list(ids.keys()):
                        self.token_type_ids.append(torch.tensor(ids['token_type_ids']).squeeze())

                    if flag == 0:
                        print('TEXT:', texts[i])

                        print('text_id:', self.texts[-1].shape, self.texts[-1])
                        print('label:', labels[i], self.labels[-1])
                        print('-' * 25)
                        flag = flag + 1

                self.labels = [int(aa) for aa in self.labels]
                self.texts = [aa.tolist() for aa in self.texts]
                self.masks = [aa.tolist() for aa in self.masks]
                if self.token_type_ids != []:
                    self.token_type_ids = [aa.tolist() for aa in self.token_type_ids]

        elif args.task_type == 'SequenceClassification':
            if len(texts.shape) == 1:  # input: single sentence
                for _text in texts:
                    if args.padding != "do_not_pad" and args.padding_type == "inside":  # [PAD] between [CLS][SEP]
                        text_tokens = args.tokenizer.tokenize(_text)

                        pad_length = max(args.max_length - len(text_tokens), 0)
                        for _pad in range(pad_length):
                            if args.padding_side == 'right':
                                text_tokens.append(args.tokenizer.pad_token)
                            elif args.padding_side == 'left':
                                text_tokens.insert(0, args.tokenizer.pad_token)
                            elif args.padding_side == 'random':
                                text_tokens.insert(randrange(len(text_tokens) + 1), args.tokenizer.pad_token)

                        _text = " ".join(text_tokens)
                        # print('after pad:', _text)

                        ids = args.tokenizer(_text, truncation=args.truncation, max_length=args.max_length, \
                                             return_tensors='pt', add_special_tokens=args.add_special_tokens)

                        # for _pos in range(ids['attention_mask'].shape[1]):
                        #     if ids['input_ids'][0][_pos] == args.tokenizer.pad_token_id:
                        #         ids['attention_mask'][0][_pos] = 0

                    else:  # [PAD] outside [CLS][SEP]
                        ids = args.tokenizer(_text, \
                                             padding=args.padding, truncation=args.truncation, \
                                             max_length=args.max_length, return_tensors='pt',
                                             add_special_tokens=args.add_special_tokens)

                        # _len = torch.tensor(ids['input_ids']).shape[1]
                        # max_token_len = max(max_token_len,_len)
                    # print('max_token_len:',max_token_len)

                    self.texts.append(torch.tensor(ids['input_ids']).squeeze())
                    # avoid performing attention on padding token indices.
                    self.masks.append(torch.tensor(ids['attention_mask']).squeeze())
                    # Segment token indices to indicate first and second portions of the inputs.
                    if 'token_type_ids' in list(ids.keys()):
                        self.token_type_ids.append(torch.tensor(ids['token_type_ids']).squeeze())
                print(self.texts[0])
                assert 1 > 2

            elif len(texts.shape) == 2:  # input: sentence pairs
                for _text in texts:

                    try:
                        ids = args.tokenizer(_text[0], _text[1], \
                                             padding=args.padding, truncation=args.truncation, \
                                             max_length=args.max_length, return_tensors='pt')
                    except:
                        assert 1 > 2

                    self.texts.append(torch.tensor(ids['input_ids']).squeeze())
                    self.masks.append(torch.tensor(ids['attention_mask']).squeeze())
                    if 'token_type_ids' in list(ids.keys()):
                        self.token_type_ids.append(torch.tensor(ids['token_type_ids']).squeeze())
                print(self.texts[0])
                assert 1 > 2

            elif len(texts.shape) == 3:  # input: sentence pairs
                for _text in texts:
                    try:
                        ids = args.tokenizer(list(_text[0]), list(_text[1]), \
                                             padding=args.padding, truncation=args.truncation, \
                                             max_length=args.max_length, return_tensors='pt')

                    except:
                        assert 1 > 2
                    self.texts.append(torch.tensor(ids['input_ids']).squeeze())
                    self.masks.append(torch.tensor(ids['attention_mask']).squeeze())
                    if 'token_type_ids' in list(ids.keys()):
                        self.token_type_ids.append(torch.tensor(ids['token_type_ids']).squeeze())

            else:
                print(texts.shape)
                assert 1 > 2, 'text input shape not supported'

            if self.args.num_classes == 1:
                self.labels = torch.tensor(labels, dtype=torch.float32)
            else:
                self.labels = torch.tensor(labels)

            self.texts = [aa.tolist() for aa in self.texts]

            self.masks = [aa.tolist() for aa in self.masks]

            if self.token_type_ids != []:
                self.token_type_ids = [aa.tolist() for aa in self.token_type_ids]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item_idx):
        data_i, target_i, mask_i = \
            self.texts[item_idx], self.labels[item_idx], self.masks[item_idx]
        data_i = torch.tensor(data_i, dtype=torch.long).to(self.args.device)
        mask_i = torch.tensor(mask_i, dtype=torch.long).to(self.args.device)

        if not type(target_i) == str:
            target_i = torch.tensor(target_i, dtype=torch.long).to(self.args.device)
            # if self.args.num_classes == 1:
            #     target_i = torch.tensor(target_i, dtype=torch.float32).to(self.args.device)
            # else:
            #     target_i = torch.tensor(target_i, dtype=torch.long).to(self.args.device)

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

        return data_i, target_i, mask_i, token_type_ids_i, features_i  # doc_tokens_i


class PassiveDataset(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item_idx):
        data_i = self.data[item_idx]
        return torch.tensor(data_i, dtype=torch.float32), torch.tensor([] * data_i.size()[0])


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
        return torch.tensor(data_i.clone().detach(), dtype=torch.float32), torch.tensor(target_i.clone().detach(),
                                                                                        dtype=torch.long)


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
