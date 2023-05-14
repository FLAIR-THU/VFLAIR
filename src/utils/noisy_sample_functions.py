import numpy as np
import torch
import random

def noisy_sample(origin_sample,scale):
    # print('origin_sample:',origin_sample.size())
    location = 0.0
    # threshold = 0.2  # 1e9
    # norm_factor_a = torch.div(torch.max(torch.norm(origin_sample, dim=1)),
    #                                         threshold + 1e-6).clamp(min=1.0)
    
    # add laplace noise
    dist_a = torch.distributions.laplace.Laplace(location, scale)
    noisy_sample =  origin_sample + dist_a.sample(origin_sample.shape) # (torch.div(origin_sample, norm_factor_a) +
    # print('noisy_sample:',noisy_sample.size())
    return noisy_sample