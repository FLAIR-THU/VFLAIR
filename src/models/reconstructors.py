import torch
import torch.nn as nn
import torch.nn.functional as F


class Reconstructor(nn.Module):
    def __init__(self, input_dim):
        super(Reconstructor, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, input_dim << 2, bias=True),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(input_dim << 2, input_dim << 2, bias=True),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(input_dim << 2, input_dim << 1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Linear(input_dim << 1, input_dim, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
