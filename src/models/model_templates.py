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


class MLP2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP2, self).__init__()
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


class ClassificationModelHost(nn.Module):

    def __init__(self, local_model):
        super().__init__()
        self.local_model = local_model

    def forward(self, input_X):
        z = self.local_model(input_X)
        return z


    def get_prediction(self, z0, z1):
        return z0+z1

    def load_local_model(self, load_path, device):
        self.local_model.load_state_dict(torch.load(load_path, map_location=device))


class ClassificationModelHostHead(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, z0, z1):
        out = z0.add(z1)
        return out


class ClassificationModelHostHeadWithSoftmax(nn.Module):

    def __init__(self):
        super().__init__()
        self.softmax = nn.LogSoftmax()

    def forward(self, z0, z1):
        out = z0.add(z1)
        return self.softmax(out)


class ClassificationModelHostTrainable(nn.Module):

    def __init__(self, local_model, hidden_dim, num_classes):
        super().__init__()
        self.local_model = local_model
        self.classifier_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_X):
        z = self.local_model(input_X).flatten(start_dim=1)
        return z

    def get_prediction(self, z0, z1):
        out = torch.cat([z0, z1], dim=1)
        return self.classifier_head(out)


class ClassificationModelHostTrainableHead(nn.Module):

    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.classifier_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, z0, z1):
        out = torch.cat([z0, z1], dim=1)
        return self.classifier_head(out)


class ClassificationModelHostHeadTrainable(nn.Module):

    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.classifier_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, z0, z1):
        out = torch.cat([z0, z1], dim=1)
        return self.classifier_head(out)


class ClassificationModelGuest(nn.Module):

    def __init__(self, local_model):#), hidden_dim, num_classes):
        super().__init__()
        self.local_model = local_model

    def forward(self, input_X):
        z = self.local_model(input_X).flatten(start_dim=1)
        return z


if __name__ == '__main__':
    from torchsummary import summary

    net = LeNet5(10)
    summary(net, (3, 16, 32))