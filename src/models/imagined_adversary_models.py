import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function


class ImaginedAdversary_MLP3(nn.Module):
    '''
    input --- intermediate : bs, seq_length, 768(embed_dim)
    output --- embedding : bs, seq_length, 768(embed_dim)
    '''

    def __init__(self, seq_length, embed_dim ,hidden_size=80):
        super(ImaginedAdversary_MLP3, self).__init__()
        # print('Adversarial_MLP init:',seq_length, embed_dim)
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        # self.batch_first = batch_first

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
        origin_dtype = x.dtype

        # print('=== ia model ===')
        # print('x raw:',x.shape,x.dtype)
        # print('origin_shape:',origin_shape)

        if origin_shape[1] != self.seq_length:
            x = x.transpose(0,1) # should be [bs, seq_len, embed_dim]
            # print('x after:',x.shape,x.dtype)

        x = torch.tensor(x, dtype=torch.float32)
        x1 = self.net1(x)
        # print('x1:',x1.shape)

        x2 = self.net2(x1)
        # print('x2:',x2.shape)

        x3 = self.net3(x2)
        x3 = x3.reshape(origin_shape)
        # x3 = torch.tensor(x,dtype=origin_dtype)

        # if not self.batch_first:
        #     x3 = x3.transpose(0,1) # should be [bs, seq_len, embed_dim]

        # print('final x3:',x3.shape,x3.dtype)
        # print('=== ia model ===')
        return x3
