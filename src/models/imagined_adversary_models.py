import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function

class ImaginedAdversary(nn.Module):
    '''
    input --- intermediate : bs, seq_length, 768(embed_dim)
    output --- embedding : bs, seq_length, 768(embed_dim)
    '''
    def __init__(self, seq_length, embed_dim):
        super(ImaginedAdversary,self).__init__()
        # print('Adversarial_MLP init:',seq_length, embed_dim)
        self.seq_length = seq_length
        self.embed_dim = embed_dim

        self.net1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_length*embed_dim, 80), 
            nn.LayerNorm(80),
            nn.ReLU(),
        )

        self.net2 = nn.Sequential(
            nn.Linear(80, 80), 
            nn.LayerNorm(80),
            nn.ReLU()
        )

        self.net3 = nn.Sequential(
            nn.Linear(80, seq_length*embed_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        origin_shape = x.shape
        # print('x:',x.shape,origin_shape)

        x = torch.tensor(x,dtype=torch.float32)
        x1 = self.net1(x)
        # print('x1:',x1.shape)

        x2 = self.net2(x1)
        # print('x2:',x2.shape)

        x3 = self.net3(x2)
        # print('x3:',x3.shape)

        x3 = x3.reshape(origin_shape)
        return x3

class Adversarial_Mapping(nn.Module):
    '''
    input --- intermediate : bs, seq_length, 768(embed_dim)
    output --- embedding : bs, seq_length, 768(embed_dim)
    '''
    def __init__(self, seq_length, embed_dim):
        super(Adversarial_Mapping,self).__init__()
        # print('Adversarial_MLP init:',seq_length, embed_dim)
        self.seq_length = seq_length
        self.embed_dim = embed_dim

        self.net1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_length*embed_dim, 80), 
            nn.LayerNorm(80),
            nn.ReLU(),
        )

        self.net2 = nn.Sequential(
            nn.Linear(80, 80), 
            nn.LayerNorm(80),
            nn.ReLU()
        )

        self.net3 = nn.Sequential(
            nn.Linear(80, seq_length*embed_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        origin_shape = x.shape
        # print('x:',x.shape,origin_shape)

        x = torch.tensor(x,dtype=torch.float32)
        x1 = self.net1(x)
        # print('x1:',x1.shape)

        x2 = self.net2(x1)
        # print('x2:',x2.shape)

        x3 = self.net3(x2)
        # print('x3:',x3.shape)

        x3 = x3.reshape(origin_shape)
        return x3
