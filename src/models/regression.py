import os, sys

sys.path.append(os.pardir)

import torch
import torch.nn as nn
import torch.nn.functional as F


# For diabetes dataset
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        out = self.layer(x)  # F.sigmoid(self.layer(x))
        return out

class LogisticRegressionModel_Flatten(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel_Flatten, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, output_dim, bias=True),
        )
        torch.nn.init.xavier_uniform_(self.layer1[1].weight)
        torch.nn.init.zeros_(self.layer1[1].bias)

    def forward(self, x):
        x = self.layer1(x)
        return x
