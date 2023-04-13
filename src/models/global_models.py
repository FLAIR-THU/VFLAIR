import torch
import torch.nn as nn
import torch.nn.functional as F

class InferenceHead(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layer=1, activation_func_type='ReLU', use_bn=True):
        super().__init__()
        dict_activation_func_type = {'ReLU': F.relu, 'Sigmoid': F.sigmoid, 'None': None}
        self.activation_func = dict_activation_func_type[activation_func_type]
        self.num_layer = num_layer
        self.use_bn = use_bn

        self.fc_1 = nn.Linear(input_size, hidden_size, bias=True)
        self.bn_1 = nn.BatchNorm1d(input_size)
        self.fc_1.apply(weights_init_ones)

        self.fc_2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.bn_2 = nn.BatchNorm1d(input_size)
        self.fc_2.apply(weights_init_ones)

        self.fc_3 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.bn_3 = nn.BatchNorm1d(input_size)
        self.fc_3.apply(weights_init_ones)

        self.fc_4 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.bn_4 = nn.BatchNorm1d(input_size)
        self.fc_4.apply(weights_init_ones)

        self.fc_final = nn.Linear(hidden_size, num_classes, bias=True)
        self.bn_final = nn.BatchNorm1d(input_size)
        self.fc_final.apply(weights_init_ones)

    def forward(self, x):
        if self.num_layer >= 2:
            if self.use_bn:
                x = self.bn_1(x)
            if self.activation_func:
                x = self.activation_func(x)
            x = self.fc_1(x)

        if self.num_layer >= 3:
            if self.use_bn:
                x = self.bn_2(x)
            if self.activation_func:
                x = self.activation_func(x)
            x = self.fc_2(x)

        if self.num_layer >= 4:
            if self.use_bn:
                x = self.bn_3(x)
            if self.activation_func:
                x = self.activation_func(x)
            x = self.fc_3(x)

        if self.num_layer >= 5:
            if self.use_bn:
                x = self.bn_4(x)
            if self.activation_func:
                x = self.activation_func(x)
            x = self.fc_4(x)

        if self.use_bn:
            x = self.bn_final(x)
        if self.activation_func:
            x = self.activation_func(x)
        x = self.fc_final(x)

        return x

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

