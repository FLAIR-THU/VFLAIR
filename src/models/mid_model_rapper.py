import os, sys
sys.path.append(os.pardir)

import torch
import torch.nn as nn
import torch.nn.functional as F

class MID_model(nn.Module):
    def __init__(self, input_dim, output_dim, mid_lambda, bottleneck_scale=1, std_shift=0.5):
        super(MID_model, self).__init__()
        self.bottleneck_scale = bottleneck_scale
        self.input_dim = input_dim
        self.mid_lambda = mid_lambda
        self.std_shift = std_shift
        self.enlarge_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, input_dim*2*bottleneck_scale, bias=True),
            nn.ReLU(inplace=True)
        )
        self.decoder_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim*bottleneck_scale, input_dim*bottleneck_scale*5, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim*bottleneck_scale*5, output_dim, bias=True),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        epsilon = torch.empty((x.size()[0],x.size()[1]*self.bottleneck_scale))
        torch.nn.init.normal(epsilon, mean=0, std=1) # epsilon is initialized
        epsilon = epsilon.to(x.device)
        # # x.size() = (batch_size, class_num)
        x_double = self.enlarge_layer(x)
        mu, std = x_double[:,:self.input_dim*self.bottleneck_scale], x_double[:,self.input_dim*self.bottleneck_scale:]
        # print(f"mu, std={mu},{std}")
        std = F.softplus(std-self.std_shift) # ? F.softplus(std-0.5) F.softplus(std-5)
        z = mu + std * epsilon
        z = z.to(x.device)
        z = self.decoder_layer(z)
        mid_loss = self.mid_lambda * torch.mean(torch.sum((-0.5)*(1+2*torch.log(std)-mu**2 - std**2),1))

        return z, mid_loss


class Passive_local_MID_model(nn.Module):
    def __init__(self, local_model, mid_model):
        super(Passive_local_MID_model, self).__init__()
        assert local_model != None and mid_model != None
        self.local_model = local_model
        self.mid_model = mid_model
        self.mid_loss = None

    def forward(self,x):
        hp = self.local_model(x)
        z, self.mid_loss = self.mid_model(hp)
        return z


class Active_global_MID_model(nn.Module):
    def __init__(self, global_model, mid_model_list):
        super(Active_global_MID_model, self).__init__()
        assert global_model != None and len(mid_model_list) > 0
        self.global_model = global_model
        self.mid_model_list = mid_model_list
        self.mid_loss = None

    def forward(self,x):
        assert len(x)-1 == len(self.mid_model_list)
        z_list = []
        self.mid_loss_list = []
        # give all passive party a mid_model
        for i in range(len(x)-1):
            _z, mid_loss = self.mid_model_list[i](x[i])
            z_list.append(_z)
            self.mid_loss_list.append(mid_loss)
        # active party does not have mid_model
        z_list.append(x[-1])
        z = self.global_model(z_list)
        # print(f"active party mid global model, before_mid={x}, after_mid={z_list}, final_global_aggregation={z}")
        return z
