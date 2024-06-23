import os, sys

sys.path.append(os.pardir)

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP2_ReLu(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP2_ReLu, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 32, bias=True),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(32, output_dim, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class MLP2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 32, bias=True),
            nn.ReLU(inplace=True)
        )
        torch.nn.init.xavier_uniform_(self.layer1[1].weight)
        torch.nn.init.zeros_(self.layer1[1].bias)

        self.layer2 = nn.Sequential(
            nn.Linear(32, output_dim, bias=True),
            # nn.ReLU(inplace=True)
        )
        # torch.nn.init.xavier_uniform_(self.layer2[0].weight)
        # torch.nn.init.zeros_(self.layer2[0].bias)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


# for BreastCancer dataset
class MLP2_128(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP2_128, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 128, bias=True),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(128, output_dim, bias=True)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


# for Attribute Inference attack
class MLP2_scalable(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(MLP2_scalable, self).__init__()
        self.drop_out_rate = 0.2
        # self.drop_out_rate = 0.1
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(self.drop_out_rate),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(self.drop_out_rate),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


# for Attribute Inference attack with language/text data
class MLP4_dropout(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP3_dropout, self).__init__()
        self.drop_out_rate = 0.2
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(self.drop_out_rate),
        )
        # torch.nn.init.xavier_uniform_(self.layer1[0].weight)
        # torch.nn.init.zeros_(self.layer1[0].bias)

        self.layer2 = nn.Sequential(
            nn.Linear(256, 128, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(self.drop_out_rate),
        )
        # torch.nn.init.xavier_uniform_(self.layer2[0].weight)
        # torch.nn.init.zeros_(self.layer2[0].bias)

        self.layer3 = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(self.drop_out_rate),
        )

        self.layer4 = nn.Sequential(
            nn.Linear(64, output_dim, bias=True),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm1d(output_dim),
            # nn.Dropout(self.drop_out_rate),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class MLP3(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP3, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 64, bias=True),
            nn.ReLU(inplace=True)
        )
        # torch.nn.init.xavier_uniform_(self.layer1[0].weight)
        # torch.nn.init.zeros_(self.layer1[0].bias)

        self.layer2 = nn.Sequential(
            nn.Linear(64, 16, bias=True),
            nn.ReLU(inplace=True)
        )
        # torch.nn.init.xavier_uniform_(self.layer2[0].weight)
        # torch.nn.init.zeros_(self.layer2[0].bias)

        self.layer3 = nn.Sequential(
            nn.Linear(16, output_dim, bias=True),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# for Nursery dataset
class MLP3_Nursery(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP3_Nursery, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, output_dim)
        )

    def forward(self, x):
        out = self.layer(x)
        return out

# for adult income dataset
class MLP4(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP4, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        out = self.layer(x)
        return out


# for Credit dataset
class MLP4_Credit(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP4_Credit, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, output_dim)
        )

    def forward(self, x):
        out = self.layer(x)
        return out

# For news20 dataset
class MLP5(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(MLP5, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        out = self.layer(x)
        return out


# For avazu and criteo dataset
class MLP3_256_dense(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP3_256_dense, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        out = self.layer(x)
        return out


# For avazu and criteo dataset
class MLP3_256_sparse(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP3_256_sparse, self).__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(1000000, 16) for _ in range(input_dim)])
        self.layer = nn.Sequential(
            nn.Linear(16 * input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        embeddings = [embedding(x[:, i].long()) for i, embedding in enumerate(self.embeddings)]
        x = torch.cat(embeddings, dim=1)
        # print(torch.min(x.long()), torch.max(x.long()))
        # x = self.embedding(x.long())
        # x = x.view(x.size(0), -1)
        out = self.layer(x)
        return out
