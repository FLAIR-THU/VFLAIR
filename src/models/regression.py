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
        out = self.layer(x) #F.sigmoid(self.layer(x))
        return out

class LogisticRegressionModel_Normalized(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel_Normalized, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        out = self.layer(x)
        # out = (out - torch.mean(out)) / (torch.std(out) + 1e-16)
        return out
