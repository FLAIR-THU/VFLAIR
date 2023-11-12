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
from PIL import Image
import torchvision.transforms as transforms
from datetime import datetime
import logging
import copy
import math

def compress_pred( pred , local_grad,  epoch ,step ):

    ratio = 0
    # if args.quant_level > 0:
    #     ratio = math.log(args.quant_level,2)/32
    
    if not (epoch == 0 and step == 0):
        # Choose top k elements based on local_grad
        pred = pred.cpu().detach().numpy()
        local_grad = local_grad.cpu().detach().numpy()
        grads = np.abs(local_grad)

        num = math.ceil(pred.shape[1]*(1-ratio))
        
        sorted_indices = np.argsort(grads)
        indices = sorted_indices[-num:]
        # idx = np.argpartition(grads, num)[:num]
        # indices = idx[np.argsort((grads)[idx])]

        pred[:,indices[:num]] = 0
        pred = torch.from_numpy(pred).float()
    else: 
        # If first iteration, do nothing
        pass
    
    # print('Compress pred:',pred.shape)

    return pred 
