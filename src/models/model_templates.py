import torch
import torch.nn as nn
import torch.nn.functional as F

# for BreastCancer dataset
class MLP2_128(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP2_128, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 128, bias=True),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(128, output_dim, bias=True)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

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

# For diabetes dataset
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        out = self.layer(x)
        return out


# For news20 dataset
class MLP5(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 64):
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
    
    
class LSTM(nn.Module):
 
    def __init__(self, vocab_size, output_dim, embedding_dim=100, hidden_dim=128):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_dim, num_layers = 1,
                            bidirectional = True, batch_first = True)
        self.Ws = nn.Parameter(torch.Tensor(hidden_dim, output_dim))
        nn.init.uniform_(self.Ws, -0.1, 0.1)
  
    def forward(self, x):
        x = self.embedding(x.long())
        #x = pack_padded_sequence(x, x_len)
        H, (h_n, c_n) = self.lstm(x)
        h_n = torch.squeeze(h_n)
        res = torch.matmul(h_n, self.Ws)
        y = F.softmax(res, dim=1)
        # y.size(batch_size, output_dim)
        return y


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

    def forward(self, z_list):
        out = z_list[0]
        for i in range(len(z_list)-1):
            out = out.add(z_list[i+1])
        return out


class ClassificationModelHostHeadWithSoftmax(nn.Module):

    def __init__(self):
        super().__init__()
        self.softmax = nn.LogSoftmax()

    def forward(self, z_list):
        out = z_list[0]
        for i in range(len(z_list)-1):
            out = out.add(z_list[i+1])
        return self.softmax(out)


class ClassificationModelHostTrainable(nn.Module):

    def __init__(self, local_model, hidden_dim, num_classes):
        super().__init__()
        self.local_model = local_model
        self.classifier_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_X):
        z = self.local_model(input_X).flatten(start_dim=1)
        return z

    def get_prediction(self,  z_list):
        out = torch.cat(z_list, dim=1)
        return self.classifier_head(out)


class ClassificationModelHostTrainableHead(nn.Module):

    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.classifier_head = nn.Linear(hidden_dim, num_classes)

    def forward(self,  z_list):
        out = torch.cat(z_list, dim=1)
        return self.classifier_head(out)


class ClassificationModelHostTrainableHead3(nn.Module):

    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.fc1_top = nn.Linear(hidden_dim, hidden_dim)
        self.fc2_top = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3_top = nn.Linear(hidden_dim//2, num_classes)
        # self.apply(weights_init)

    def forward(self, z_list):
        out = torch.cat(z_list, dim=1)
        x = F.relu(out)
        x = self.fc1_top(x)
        x = F.relu(x)
        x = self.fc2_top(x)
        x = F.relu(x)
        x = self.fc3_top(x)
        return x


class ClassificationModelGuest(nn.Module):

    def __init__(self, local_model):#), hidden_dim, num_classes):
        super().__init__()
        self.local_model = local_model

    def forward(self, input_X):
        z = self.local_model(input_X).flatten(start_dim=1)
        return z



class Backdoor_ClassificationModelHost(nn.Module):

    def __init__(self, local_model):
        super().__init__()
        self.local_model = local_model

    def forward(self, input_X):
        z = self.local_model(input_X)
        return z


    def get_prediction(self, z_list):
        result = z_list[0]
        for i in range(len(z_list)-1):
            result += z_list[i+1]
        return result

    def load_local_model(self, load_path, device):
        self.local_model.load_state_dict(torch.load(load_path, map_location=device))


class Backdoor_ClassificationModelHostHead(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, z_list):
        out = z_list[0]
        for i in range(len(z_list)-1):
            out = out.add(z_list[i+1])
        return out


class Backdoor_ClassificationModelHostHeadWithSoftmax(nn.Module):

    def __init__(self):
        super().__init__()
        self.softmax = nn.LogSoftmax()

    def forward(self, z_list):
        out = z_list[0]
        for i in range(len(z_list)-1):
            out = out.add(z_list[i+1])
        # out = z0.add(z1)
        return self.softmax(out)


class Backdoor_ClassificationModelHostTrainable(nn.Module):

    def __init__(self, local_model, hidden_dim, num_classes):
        super().__init__()
        self.local_model = local_model
        self.classifier_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_X):
        z = self.local_model(input_X).flatten(start_dim=1)
        return z

    def get_prediction(self, z_list):
        out = torch.cat(z_list, dim=1)
        return self.classifier_head(out)


class Backdoor_ClassificationModelHostTrainableHead(nn.Module):

    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.classifier_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, z_list):
        out = torch.cat(z_list, dim=1)
        return self.classifier_head(out)


class Backdoor_ClassificationModelHostHeadTrainable(nn.Module):

    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.classifier_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, z_list):
        out = torch.cat(z_list, dim=1)
        return self.classifier_head(out)


class Backdoor_ClassificationModelGuest(nn.Module):

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
