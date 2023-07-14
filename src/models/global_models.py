import torch
import torch.nn as nn
import torch.nn.functional as F


import torch.nn.init as init
def weights_init(m):
    # classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class TopModelForCifar10(nn.Module):
    def __init__(self, hidden_dim=20, num_classes=10):
        super(TopModelForCifar10, self).__init__()
        self.fc1top = nn.Linear(20, 20)
        self.fc2top = nn.Linear(20, 10)
        self.fc3top = nn.Linear(10, 10)
        self.fc4top = nn.Linear(10, 10)
        self.bn0top = nn.BatchNorm1d(20)
        self.bn1top = nn.BatchNorm1d(20)
        self.bn2top = nn.BatchNorm1d(10)
        self.bn3top = nn.BatchNorm1d(10)

        self.apply(weights_init)

    def forward(self, input_list):
        input_tensor_top_model_a, input_tensor_top_model_b = input_list[0], input_list[1]
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))
        return F.log_softmax(x, dim=1)


class ActivePartyWithoutTrainableLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_a, pred_b):
        pred = pred_a + pred_b
        return pred


class ActivePartyWithTrainableLayer(nn.Module):

    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.classifier_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, pred_a, pred_b):
        out = torch.cat([pred_a, pred_b], dim=1) # out = [dim0, dim1_a+dim1_b], dim0 is number of samples
        return self.classifier_head(out)


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

