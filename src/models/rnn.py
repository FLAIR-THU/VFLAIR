import os, sys

sys.path.append(os.pardir)

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):

    def __init__(self, vocab_size, output_dim, embedding_dim=100, hidden_dim=128):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.Ws = nn.Parameter(torch.Tensor(hidden_dim, output_dim))
        nn.init.uniform_(self.Ws, -0.1, 0.1)

    def forward(self, x):
        x = self.embedding(x.long())
        # x = pack_padded_sequence(x, x_len)
        H, (h_n, c_n) = self.lstm(x)
        h_n = torch.squeeze(h_n)
        res = torch.matmul(h_n, self.Ws)
        y = F.softmax(res, dim=1)
        # y.size(batch_size, output_dim)
        return y
