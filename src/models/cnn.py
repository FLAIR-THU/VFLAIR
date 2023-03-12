import os, sys
sys.path.append(os.pardir)

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 20, 5)
        self.fc1 = nn.Linear(20 * 1 * 5, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 20 * 1 * 5)
        x = self.fc1(x)
        x = F.relu(x)
        return x


class LeNet(nn.Module):
    def __init__(self, output_dim):
        self.output_dim = output_dim
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(768, self.output_dim)
        )
        
    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        return out


class LeNet2(nn.Module):
    def __init__(self, classes = 2):
        super(LeNet2, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5//2, stride=2),
            act(), # 16 * 8 * 12
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(), # 8 * 4 * 12
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(), # 8 * 4 * 12
        )
        self.fc = nn.Sequential(
            nn.Linear(384, classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        return out


class LeNet3(nn.Module):
    def __init__(self, classes=2):
        super(LeNet3, self).__init__()
        # act = nn.Sigmoid
        act = nn.LeakyReLU
        # act = nn.ReLU
        padding_1 = 1
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=18, kernel_size=3, padding=padding_1, stride=1),
            act(),  # 128 * 64 * 12
            nn.Conv2d(in_channels=18, out_channels=36, kernel_size=3, padding=padding_1, stride=1),
            act(),  # 64 * 32 * 12
            nn.Conv2d(in_channels=36, out_channels=64, kernel_size=3, stride=1),
            act(),  # 64 * 32 * 12
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            act(),  # 64 * 32 * 12
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.fc = nn.Sequential(
            # nn.Linear(64 * 16 * 16, classes)
            # nn.Linear(25088, classes)
            nn.Linear(10752, classes)

        )

    def forward(self, x):
        out = self.body(x)
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        # print("out: ", out.size())
        out = self.fc(out)
        return out


class LeNet5(nn.Module):

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        # act = nn.Tanh
        act = nn.ReLU
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1),
            act(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1),
            act(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=60, kernel_size=3, stride=1, padding=1),
            act(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=480, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=n_classes),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


if __name__ == '__main__':
    from torchsummary import summary

    net = LeNet5(10)
    summary(net, (3, 16, 32))
