import os
import random
import csv
import os
from decimal import Decimal
from io import BytesIO
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from matplotlib.ticker import MultipleLocator
from torchvision import models, datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import logging
import copy
import math
import threading
from loguru import logger


def get_size_of(target_tensor):
    # logger.info(f"n: {target_tensor.nelement()}, size: {target_tensor.element_size()}")
    return target_tensor.nelement() * target_tensor.element_size() / (1024 * 1024)  # mb
    # _size = 1
    # for _dim in target_tensor.shape:
    #     _size = _size*_dim
    # return _size*4/(1024*1024) # MB

def get_total_size(tensor_dict:dict):
    total_size=0.0
    for t in tensor_dict.values():
        if isinstance(t, torch.Tensor):
            total_size += get_size_of(t)
    # logger.info(f"total size: {total_size}MB")
    return total_size

# CELU
class Cache(object):
    def __init__(self):
        # batch: pred grad batch_cached_at used_time
        self._cache = {}
        # self._cv = threading.Condition()

    def put(self, batch, act, dev, timestamp):
        self._cache[batch] = [act, dev, timestamp, 0]

    def sample(self, reject_lists):
        # while len(self._cache) == 0:
        ret = random.sample(self._cache.items(), 1)[0]
        if ret[0] not in reject_lists:
            return ret

    def inc(self, batch):  # used once
        if batch in self._cache:
            self._cache[batch][-1] += 1

    def remove(self, batch):
        if batch in self._cache:
            del self._cache[batch]


def ins_weight(V, stale_V, smi_thresh):
    weights = F.cosine_similarity(V, stale_V, dim=1)
    # print(weights.shape) # [2048]

    for i in range(weights.shape[0]):
        if weights[i] < smi_thresh:
            weights[i] = 0
    return weights


# Compress
def compress_pred(args, pred, local_grad, epoch, step):
    comp = args.communication_protocol
    if comp == 'Topk':
        ratio = args.ratio
        # if args.quant_level > 0:
        #     ratio = math.log(args.quant_level,2)/32
        if not (step == 0):  # epoch == 0 and
            # Choose top k elements based on local_grad
            pred = pred.cpu().detach().numpy()
            local_grad = local_grad.cpu().detach().numpy()
            grads = np.abs(local_grad)  # [2048,10]

            num = math.ceil(pred.shape[1] * (1 - ratio))
            # print('changed num:',num)

            for _i in range(pred.shape[0]):
                sorted_indices = np.argsort(grads[_i])
                indices = sorted_indices[-num:]
                pred[_i, indices[:num]] = 0
            # print('pred:',pred.shape,pred[0])
            # print('sorted_indices:',sorted_indices)
            # print('indices:',indices)

            # idx = np.argpartition(grads, num)[:num]
            # indices = idx[np.argsort((grads)[idx])]

            # print('pred:',pred[0])

            pred = torch.from_numpy(pred).float()
        else:
            # If first iteration, do nothing
            pass
    elif comp == 'Quantization':
        if args.vecdim == 1:
            # Scalar quantization
            pred = quantize_scalar(pred.cpu().detach().numpy(),
                                   quant_level=args.quant_level)
        else:
            # Vector quantization
            pred = quantize_vector(pred.cpu().detach().numpy(),
                                   quant_level=args.quant_level, dim=args.vecdim)

    # print('Compress pred:',pred.shape)

    return pred


def quantize_vector(x, quant_min=0, quant_max=1, quant_level=5, dim=2):
    """Uniform vector quantization approach

    Notebook: C2S2_DigitalSignalQuantization.ipynb

    Args:
        x: Original signal
        quant_min: Minimum quantization level
        quant_max: Maximum quantization level
        quant_level: Number of quantization levels
        dim: dimension of vectors to quantize

    Returns:
        x_quant: Quantized signal

        Currently only works for 2 dimensions and 
        quant_levels of 4, 8, and 16.
    """

    dither = np.random.uniform(-(quant_max - quant_min) / (2 * (quant_level - 1)),
                               (quant_max - quant_min) / (2 * (quant_level - 1)),
                               size=np.array(x).shape)
    # Move into 0,1 range:
    x_normalize = x / np.max(x)
    x_normalize = x_normalize + dither

    A2 = latbin.lattice.ALattice(dim, scale=1 / (2 * math.log(quant_level, 2)))
    if quant_level == 4:
        A2 = latbin.lattice.ALattice(dim, scale=1 / 4)
    elif quant_level == 8:
        A2 = latbin.lattice.ALattice(dim, scale=1 / 8.5)
    elif quant_level == 16:
        A2 = latbin.lattice.ALattice(dim, scale=1 / 19)

    for i in range(0, x_normalize.shape[1], dim):
        x_normalize[:, i:(i + dim)] = A2.lattice_to_data_space(
            A2.quantize(x_normalize[:, i:(i + dim)]))

    # Move out of 0,1 range:
    x_normalize = np.max(x) * (x_normalize - dither)
    return torch.from_numpy(x_normalize).float()


def quantize_scalar(x, quant_min=0, quant_max=1, quant_level=5):
    """Uniform quantization approach

    Notebook: C2S2_DigitalSignalQuantization.ipynb

    Args:
        x: Original signal
        quant_min: Minimum quantization level
        quant_max: Maximum quantization level
        quant_level: Number of quantization levels

    Returns:
        x_quant: Quantized signal
    """
    x_normalize = np.array(x)

    # Move into 0,1 range:
    x_normalize = x_normalize / np.max(x)
    x_normalize = np.nan_to_num(x_normalize)

    dither = np.random.uniform(-(quant_max - quant_min) / (2 * (quant_level - 1)),
                               (quant_max - quant_min) / (2 * (quant_level - 1)),
                               size=x_normalize.shape)
    x_normalize = x_normalize + dither

    x_normalize = (x_normalize - quant_min) * (quant_level - 1) / (quant_max - quant_min)
    x_normalize[x_normalize > quant_level - 1] = quant_level - 1
    x_normalize[x_normalize < 0] = 0
    x_normalize_quant = np.around(x_normalize)
    x_quant = (x_normalize_quant) * (quant_max - quant_min) / (quant_level - 1) + quant_min

    # Move out of 0,1 range:
    x_quant = np.max(x) * (x_quant - dither)
    return torch.from_numpy(x_quant).float()


def communication_cost_estimator(batch_size, seq_length, hidden_size, vocab_size, element_weight=4, num_slice=2,
                                 backward_steps=2):
    w_hidden_state = batch_size * seq_length * hidden_size * element_weight
    w_attention_mask = batch_size * seq_length * seq_length * element_weight
    w_logits = batch_size * seq_length * vocab_size * element_weight
    w_grad_of_loss = batch_size * seq_length * vocab_size * element_weight
    w_grad_of_hidden = batch_size * seq_length * hidden_size * element_weight
    if num_slice == 2:
        _backward_cost_by_step = {0: w_grad_of_loss, 1: w_grad_of_hidden}
        _cost = {'f': (w_hidden_state + w_attention_mask + w_logits) / 2 ** 20,
                 'b': sum([_backward_cost_by_step[i] for i in range(backward_steps)]) / 2 ** 20}
    elif num_slice == 3:
        _backward_cost_by_step = {0: w_grad_of_hidden, 1: w_grad_of_hidden}
        _cost = {'f': (w_hidden_state + w_attention_mask) * 2 / 2 ** 20,
                 'b': sum([_backward_cost_by_step[i] for i in range(backward_steps)]) / 2 ** 20}

    else:
        raise ValueError(f"num_slice={num_slice}")
    return _cost
