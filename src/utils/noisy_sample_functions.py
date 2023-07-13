import numpy as np
import torch
import random

def noisy_sample(origin_sample,scale):
    # print('origin_sample:',origin_sample.size()) # [12665, 1, 14, 28]

    location = 0.0
    # threshold = 0.2  # 1e9
    # norm_factor_a = torch.div(torch.max(torch.norm(origin_sample, dim=1)),
    #                                         threshold + 1e-6).clamp(min=1.0)
    noisy_sample = origin_sample

    for i in range(len(origin_sample)):
        # dist_a = torch.distributions.laplace.Laplace(location, scale)
        # noisy_sample[i] =  origin_sample[i] + dist_a.sample(origin_sample[i].shape).to(origin_sample.device)

        noise = np.random.normal(loc=0.0, scale=scale, size=origin_sample[i].shape)
        noise =  np.array(origin_sample[i].cpu()) + noise
        np.putmask(noise, noise>1.0, 1.0)
        np.putmask(noise, noise<0.0, 0.0)

        noisy_sample[i]=torch.tensor(noise).to(origin_sample.device)

    # print('noisy_sample:',len(noisy_sample),noisy_sample[0].size())
    return (noisy_sample).to(origin_sample.device)
