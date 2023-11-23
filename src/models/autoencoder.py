import torch
import torch.nn.functional as F
from torch import nn
from utils.basic_functions import sharpen


def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal(m.weight.data)
        # if m.bias is not None:
        #     m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight.data, 0, 0.01)
            m.bias.data.zero_()


class AutoEncoder(nn.Module):
    def __init__(self, input_dim=2, encode_dim=3):
        super(AutoEncoder, self).__init__()
        self.d = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encode_dim ** 2),
            nn.ReLU(),
            nn.Linear(encode_dim ** 2, input_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, encode_dim ** 2),
            nn.ReLU(),
            nn.Linear(encode_dim ** 2, input_dim),
            nn.Softmax(dim=1)
        )
        initialize_weights(self)

    def decode(self,d_y):
        return self.decoder(d_y)
        
    def forward(self, x):
        z = self.encoder(x.view(-1, self.d))
        d_y = F.softmax(z, dim=1)
        d_y = sharpen(d_y, T=1.0)
        return self.decoder(d_y), d_y
        # return self.decoder(z), d_y

    def load_model(self, model_full_name, target_device='cuda:0'):
        self.load_state_dict(torch.load(model_full_name,map_location=target_device))

    def save_model(self, model_full_name):
        torch.save(self.state_dict(), model_full_name)


class AutoEncoder_large(nn.Module):
    def __init__(self, real_dim=100,input_dim=20, encode_dim=122 ):
        super(AutoEncoder_large, self).__init__()
        self.d = input_dim
        self.real_dim = real_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encode_dim ** 2),
            nn.ReLU(),
            nn.Linear(encode_dim ** 2, input_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, encode_dim ** 2),
            nn.ReLU(),
            nn.Linear(encode_dim ** 2, input_dim),
            nn.Softmax(dim=1)
        )
        initialize_weights(self)

    def decode(self,d_y):
        return self.decoder(d_y.view(-1, self.d)).view(-1,self.real_dim)

    def forward(self, x):
        z = self.encoder(x.view(-1, self.d))
        z = z.view(-1,self.real_dim)
        d_y = F.softmax(z, dim=1)
        d_y = sharpen(d_y, T=1.0)
        return self.decoder(d_y.view(-1, self.d)).view(-1,self.real_dim), d_y
        # return self.decoder(z), d_y

    def load_model(self, model_full_name, target_device='cuda:0'):
        self.load_state_dict(torch.load(model_full_name,map_location=target_device))

    def save_model(self, model_full_name):
        torch.save(self.state_dict(), model_full_name)


class AutoEncoder_extend(nn.Module):
    def __init__(self, input_dim=2, encode_dim=3, extend_rate=2):
        super(AutoEncoder_extend, self).__init__()
        self.d = input_dim
        self.extend_d = encode_dim*extend_rate
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encode_dim ** 2),
            nn.ReLU(),
            nn.Linear(encode_dim ** 2, input_dim)
        )

        self.extention = nn.Sequential(
            nn.Linear(input_dim, self.extend_d)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.extend_d, encode_dim ** 2),
            nn.ReLU(),
            nn.Linear(encode_dim ** 2, input_dim),
            nn.Softmax(dim=1)
        )
        initialize_weights(self)

    def forward(self, x):
        z = self.encoder(x.view(-1, self.d))
        d_y = F.softmax(z, dim=1)
        d_y = sharpen(d_y, T=1.0)
        return self.decoder(self.extention(d_y)), d_y
        # return self.decoder(self.extendtion(z)), d_y

    def load_model(self, model_full_name):
        self.load_state_dict(torch.load(model_full_name))

    def save_model(self, model_full_name):
        torch.save(self.state_dict(), model_full_name)


class AutoEncoder_adversarial(nn.Module):
    def __init__(self, input_dim=2, encode_dim=3, encoded_dim=2):
        super(AutoEncoder_adversarial, self).__init__()
        self.d = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encode_dim),
            nn.ReLU(),
            nn.Linear(encode_dim, encoded_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoded_dim, encode_dim),
            nn.ReLU(),
            nn.Linear(encode_dim, input_dim),
            nn.Softmax(dim=1)
        )
        initialize_weights(self)
    
    def forward(self, x):
        # print(x.shape)
        z = self.encoder(x.view(-1, self.d))
        z = F.softmax(z, dim=1)
        z = sharpen(z, T=1.0)
        # print(z.shape)
        return self.decoder(z), z # decoder(encoder(x)), encoder(x)

    def load_model(self, model_full_name):
        self.load_state_dict(torch.load(model_full_name))

    def save_model(self, model_full_name):
        torch.save(self.state_dict(), model_full_name)


class AutoEncoder2(nn.Module):
    def __init__(self, input_dim=2, encode_dim=3):
        super(AutoEncoder2, self).__init__()
        self.d = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encode_dim * 100),
            nn.ReLU(),
            # nn.Linear(encode_dim * 40, encode_dim),
            # nn.ReLU(),
            nn.Linear(encode_dim * 100, input_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, encode_dim * 100),
            nn.ReLU(),
            # nn.Linear(encode_dim * 40, encode_dim),
            # nn.ReLU(),
            nn.Linear(encode_dim * 100, input_dim)
        )
        initialize_weights(self)

    def forward(self, x):
        z = self.encoder(x.view(-1, self.d))
        d_y = F.softmax(z, dim=1)
        d_y = sharpen(d_y, T=0.5)
        return self.decoder(d_y), d_y

    def load_model(self, model_full_name):
        self.load_state_dict(torch.load(model_full_name))

    def save_model(self, model_full_name):
        torch.save(self.state_dict(), model_full_name)


class AutoEncoder3(nn.Module):
    def __init__(self, input_dim=2, encode_dim=3):
        super(AutoEncoder3, self).__init__()
        self.d = input_dim
        print(input_dim)
        print(encode_dim * 60)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encode_dim * 30),
            nn.ReLU(),
            nn.Linear(encode_dim * 30, encode_dim),
            nn.ReLU(),
            nn.Linear(encode_dim, input_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, encode_dim ** 30),
            nn.ReLU(),
            nn.Linear(encode_dim ** 30, encode_dim),
            nn.ReLU(),
            nn.Linear(encode_dim, input_dim),
            nn.Softmax(dim=1)
        )
        initialize_weights(self)

    def forward(self, x):
        z = self.encoder(x.view(-1, self.d))
        d_y = F.softmax(z, dim=1)
        return self.decoder(d_y), d_y

    def load_model(self, model_full_name):
        self.load_state_dict(torch.load(model_full_name))

    def save_model(self, model_full_name):
        torch.save(self.state_dict(), model_full_name)
