import numpy as np
import torch
import random
import copy


def noisy_sample(origin_sample, scale):
    # print('origin_sample:',origin_sample.size()) # [12665, 1, 14, 28]

    location = 0.0
    # threshold = 0.2  # 1e9
    # norm_factor_a = torch.div(torch.max(torch.norm(origin_sample, dim=1)),
    #                                         threshold + 1e-6).clamp(min=1.0)

    noisy_sample = copy.deepcopy(origin_sample)
    for i in range(len(origin_sample)):
        # dist_a = torch.distributions.laplace.Laplace(location, scale)
        # noisy_sample[i] =  origin_sample[i] + dist_a.sample(origin_sample[i].shape).to(origin_sample.device)

        noise = np.random.normal(loc=location, scale=scale, size=origin_sample[i].shape)
        noise = np.array(origin_sample[i].cpu()) + noise
        np.putmask(noise, noise > 1.0, 1.0)
        np.putmask(noise, noise < 0.0, 0.0)
        noisy_sample[i] = torch.tensor(noise, dtype=origin_sample.dtype).to(origin_sample.device)

    # noise = np.random.normal(loc=0.0, scale=scale, size=origin_sample.shape)
    # noise = np.array(origin_sample.cpu()) + noise
    # np.putmask(noise, noise>1.0, 1.0)
    # np.putmask(noise, noise<0.0, 0.0)
    # noisy_sample = torch.tensor(noise,dtype=origin_sample.dtype).to(origin_sample.device)

    # print('noisy_sample:',len(noisy_sample),noisy_sample[0].size())
    # print(origin_sample.dtype, noisy_sample.dtype)

    # print(noisy_sample[0])
    # print(origin_sample[0])
    return (noisy_sample).to(origin_sample.device)
