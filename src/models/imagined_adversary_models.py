import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function


class ImaginedAdversary_MLP3(nn.Module):
    '''
    input --- intermediate : bs, seq_length, 768(embed_dim)
    output --- embedding : bs, seq_length, 768(embed_dim)
    '''

    def __init__(self, seq_length, embed_dim, hidden_size=80):
        super(ImaginedAdversary_MLP3, self).__init__()
        # print('Adversarial_MLP init:',seq_length, embed_dim)
        self.seq_length = seq_length
        self.embed_dim = embed_dim

        self.net1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_length * embed_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        self.net2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )

        self.net3 = nn.Sequential(
            nn.Linear(hidden_size, seq_length * embed_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        origin_shape = x.shape
        # print('x:',x.shape,origin_shape)

        x = torch.tensor(x, dtype=torch.float32)
        x1 = self.net1(x)
        # print('x1:',x1.shape)

        x2 = self.net2(x1)
        # print('x2:',x2.shape)

        x3 = self.net3(x2)
        # print('x3:',x3.shape)

        x3 = x3.reshape(origin_shape)
        return x3
