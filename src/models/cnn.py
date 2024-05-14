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


# LeNet (LeCun et al., 1998) variant: three 3×3 convolutional and 2×2 max-pooling layers
# with 16, 32, and 64 filters, followed by two FC layers with 128 and 64 hidden units

class LeNet_LeCun(nn.Module):
    def __init__(self, output_dim, bn=True):
        self.output_dim = output_dim
        super(LeNet_LeCun, self).__init__()
        act = nn.ReLU
        self.drop_rate = 0.2
        # [debug] in LeNet_LeCun, input.shape=torch.Size([128, 3, 50, 25])
        # [debug] in LeNet_LeCun, conv1.out.shape=torch.Size([128, 16, 24, 11])
        # [debug] in LeNet_LeCun, conv2.out.shape=torch.Size([128, 32, 11, 4])
        # [debug] in LeNet_LeCun, conv3.out.shape=torch.Size([128, 64, 4, 1])
        # [debug] in LeNet_LeCun, out.view(-1).shape=torch.Size([128, 256])
        self.body1 = nn.Sequential(
            # input [batch_size, 3, 50, 25]
            nn.Conv2d(3, 16, kernel_size=3, padding=(0, 1), stride=1, bias=False),
            # [batch_size, 3, 48, 24]
            act(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # [batch_size, 16, 24, 12]
            nn.Dropout(self.drop_rate),
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=0, stride=1, bias=False),
            # [batch_size, 32, 22, 10]
            act(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # [batch_size, 32, 11, 5]
            nn.Dropout(self.drop_rate),
        )
        self.body3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=(1, 1), stride=1, bias=False),
            # [batch_size, 64, 10, 4]
            act(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # [batch_size, 64, 5, 2]
            nn.Dropout(self.drop_rate),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 2, 128, bias=False),
            act(),
            nn.BatchNorm1d(128),
            nn.Dropout(self.drop_rate),
            nn.Linear(128, self.output_dim, bias=False),
            act(),
            nn.BatchNorm1d(self.output_dim),
            nn.Dropout(self.drop_rate),
        )
        # # [debug] in LeNet_LeCun, input.shape=torch.Size([128, 3, 50, 25])
        # # [debug] in LeNet_LeCun, conv1.out.shape=torch.Size([128, 16, 24, 11])
        # # [debug] in LeNet_LeCun, conv2.out.shape=torch.Size([128, 32, 11, 4])
        # # [debug] in LeNet_LeCun, conv3.out.shape=torch.Size([128, 64, 4, 1])
        # # [debug] in LeNet_LeCun, out.view(-1).shape=torch.Size([128, 256])
        # self.body1 = nn.Sequential(
        #     # input [batch_size, 3, 50, 50]
        #     nn.Conv2d(3, 16, kernel_size=3, padding=(0,0), stride=1, bias=False),
        #     # [batch_size, 3, 48, 48]
        #     act(),
        #     nn.BatchNorm2d(16),
        #     nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        #     # [batch_size, 16, 24, 24]
        #     nn.Dropout(self.drop_rate),
        # )
        # self.body2 = nn.Sequential(        
        #     nn.Conv2d(16, 32, kernel_size=3, padding=(0,0), stride=1, bias=False),
        #     # [batch_size, 32, 22, 22]
        #     act(),
        #     nn.BatchNorm2d(32),
        #     nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        #     # [batch_size, 32, 11, 11]
        #     nn.Dropout(self.drop_rate),
        # )
        # self.body3 = nn.Sequential(      
        #     nn.Conv2d(32, 64, kernel_size=3, padding=(1,1), stride=1, bias=False),
        #     # [batch_size, 64, 10, 10]
        #     act(),
        #     nn.BatchNorm2d(64),
        #     nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        #     # [batch_size, 64, 5, 5]
        #     nn.Dropout(self.drop_rate),
        # )
        # self.fc = nn.Sequential(
        #     nn.Linear(64*5*5, 128, bias=False),
        #     act(),
        #     nn.BatchNorm1d(128),
        #     nn.Dropout(self.drop_rate),
        #     nn.Linear(128, self.output_dim, bias=False),
        #     act(),
        #     nn.BatchNorm1d(self.output_dim),
        #     nn.Dropout(self.drop_rate),
        # )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        # print(f"[debug] in LeNet_LeCun, input.shape={x.shape}")
        out = self.body1(x)
        # print(f"[debug] in LeNet_LeCun, conv1.out.shape={out.shape}")
        out = self.body2(out)
        # print(f"[debug] in LeNet_LeCun, conv2.out.shape={out.shape}")
        out = self.body3(out)
        # print(f"[debug] in LeNet_LeCun, conv3.out.shape={out.shape}")
        out = out.view(out.size(0), -1)
        # print(f"[debug] in LeNet_LeCun, out.view(-1).shape={out.shape}")

        out = self.fc(out)
        # print(f"[debug] in LeNet_LeCun, fc.out.shape={out.shape}")
        return out


class CNN_3(nn.Module):
    def __init__(self, output_dim, bn=True):
        self.output_dim = output_dim
        print(f"[debug] CNN_3 has output_dim={output_dim}")
        super(CNN_3, self).__init__()
        act = nn.ReLU
        self.drop_rate = 0.25
        # # [debug] in LeNet_LeCun, input.shape=torch.Size([128, 3, 50, 25])
        # # [debug] in LeNet_LeCun, conv1.out.shape=torch.Size([128, 16, 24, 11])
        # # [debug] in LeNet_LeCun, conv2.out.shape=torch.Size([128, 32, 11, 4])
        # # [debug] in LeNet_LeCun, conv3.out.shape=torch.Size([128, 64, 4, 1])
        # # [debug] in LeNet_LeCun, out.view(-1).shape=torch.Size([128, 256])
        # self.body1 = nn.Sequential(
        #     # input [batch_size, 3, 50, 50]
        #     nn.Conv2d(3, 16, kernel_size=3, padding=(0,0), stride=1, bias=False),
        #     # [batch_size, 3, 48, 48]
        #     act(),
        #     nn.BatchNorm2d(16),
        #     nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        #     # [batch_size, 16, 24, 24]
        #     nn.Dropout(self.drop_rate),
        # )
        # self.body2 = nn.Sequential(        
        #     nn.Conv2d(16, 32, kernel_size=3, padding=(0,0), stride=1, bias=False),
        #     # [batch_size, 32, 22, 22]
        #     act(),
        #     nn.BatchNorm2d(32),
        #     nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        #     # [batch_size, 32, 11, 11]
        #     nn.Dropout(self.drop_rate),
        # )
        # self.body3 = nn.Sequential(      
        #     nn.Conv2d(32, 32, kernel_size=3, padding=(1,1), stride=1, bias=False),
        #     # [batch_size, 64, 10, 10]
        #     act(),
        #     nn.BatchNorm2d(32),
        #     nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        #     # [batch_size, 64, 5, 5]
        #     nn.Dropout(self.drop_rate),
        # )
        # self.fc = nn.Sequential(
        #     nn.Linear(64*5*5, self.output_dim, bias=False),
        #     act(),
        #     nn.BatchNorm1d(self.output_dim),
        #     nn.Dropout(0.5),
        # )
        self.body1 = nn.Sequential(
            # input [batch_size, 3, 50, 50]
            nn.Conv2d(3, 16, kernel_size=3, padding=(0, 0), stride=1, bias=False),
            # [batch_size, 3, 48, 48]
            act(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # [batch_size, 16, 24, 24]
            nn.Dropout(self.drop_rate),
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=(0, 0), stride=1, bias=False),
            # [batch_size, 32, 22, 22]
            act(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # [batch_size, 32, 11, 11]
            nn.Dropout(self.drop_rate),
        )
        self.body3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=(1, 1), stride=1, bias=False),
            # [batch_size, 64, 10, 10]
            act(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # [batch_size, 64, 5, 5]
            nn.Dropout(self.drop_rate),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 5 * 5, self.output_dim, bias=False),
            act(),
            nn.BatchNorm1d(self.output_dim),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        # print(f"[debug] in CNN_3, input.shape={x.shape}")
        out = self.body1(x)
        # print(f"[debug] in CNN_3, conv1.out.shape={out.shape}")
        out = self.body2(out)
        # print(f"[debug] in CNN_3, conv2.out.shape={out.shape}")
        out = self.body3(out)
        # print(f"[debug] in CNN_3, conv3.out.shape={out.shape}")
        out = out.view(out.size(0), -1)
        # print(f"[debug] in CNN_3, out.view(-1).shape={out.shape}")

        out = self.fc(out)
        # print(f"[debug] in CNN_3, fc.out.shape={out.shape}")
        return out


class LeNet(nn.Module):
    def __init__(self, output_dim):
        self.output_dim = output_dim
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
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
    def __init__(self, classes=2):
        super(LeNet2, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),  # 16 * 8 * 12
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),  # 8 * 4 * 12
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),  # 8 * 4 * 12
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


# changed from AlexNet in https://github.com/dansuh17/alexnet-pytorch/blob/d0c1b1c52296ffcbecfbf5b17e1d1685b4ca6744/model.py#L40
class AlexNet(nn.Module):
    """
    Neural network model consisting of layers propsed by AlexNet paper.
    """

    def __init__(self, output_dim=1000):
        """
        Define and allocate layers for this neural net.

        Args:
            output_dim (int): dim of output, usually the number of classes to predict with this model
        """
        super().__init__()
        # input size should be : (b x 3 x 128 x 128)
        # The image in the original paper states that width and height are 224 pixels, but
        # the dimensions after first convolution layer do not lead to 55 x 55.
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4),  # (b x 96 x 32 x 16)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # section 3.3
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(48, 64, 5, padding=2),  # (b x 256 x 27 x 27)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
        )
        self.net3 = nn.Sequential(
            nn.Conv2d(64, 96, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            # nn.Conv2d(96, 96, 3, padding=1),  # (b x 384 x 13 x 13)
            # nn.ReLU(),
            nn.Conv2d(96, 64, 3, padding=1),  # (b x 256 x 13 x 13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
        )
        # classifier is just a name for linear layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256 * 6 * 6), out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=output_dim),
        )
        self.init_bias()  # initialize bias

    def init_bias(self):
        for layer in self.net1:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        for layer in self.net2:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        for layer in self.net3:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers
        nn.init.constant_(self.net2[0].bias, 1)
        nn.init.constant_(self.net3[2].bias, 1)
        # nn.init.constant_(self.net3[4].bias, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        print(f"[debug] in AlexNet, input.shape={x.shape}")
        out = self.net1(x)
        print(f"[debug] in AlexNet, conv1.out.shape={out.shape}")
        out = self.net2(out)
        print(f"[debug] in AlexNet, conv2.out.shape={out.shape}")
        out = self.net3(out)
        print(f"[debug] in AlexNet, conv3.out.shape={out.shape}")
        out = out.view(out.size(0), -1)
        print(f"[debug] in CNN_3, out.view(-1).shape={out.shape}")

        return self.classifier(out)


if __name__ == '__main__':
    from torchsummary import summary

    net = LeNet5(10)
    summary(net, (3, 16, 32))
