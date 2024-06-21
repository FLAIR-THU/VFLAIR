import os, sys

sys.path.append(os.pardir)

import torch
import torch.nn as nn
import torch.nn.functional as F


class MIDCLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''

    def __init__(self, x_dim, y_dim, hidden_size):
        super(MIDCLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        # print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples) ** 2 / 2. / logvar.exp()

        prediction_1 = mu.unsqueeze(1)  # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)  # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2. / logvar.exp()

        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    def loglikeli(self, x_samples, y_samples):  # unnormalized loglikelihood
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class MIDModelCNN_MaxUnpool2d(nn.Module):
    def __init__(self, seq_length, embed_dim, mid_lambda, bottleneck_scale=1, std_shift=0.5):
        super(MIDModelCNN_MaxUnpool2d, self).__init__()
        self.bottleneck_scale = bottleneck_scale

        self.input_dim = (seq_length // 2) * (embed_dim // 2)
        self.output_dim = seq_length * embed_dim

        self.embed_dim = embed_dim
        self.seq_length = seq_length

        self.mid_lambda = mid_lambda
        self.std_shift = std_shift

        # bs, seq_len, 768
        self.enlarge_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
            nn.Linear(self.input_dim, 2 * self.input_dim * self.bottleneck_scale, bias=True),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, stride=2)

        self.decoder_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )

    def forward(self, x):
        # print('== MID Model Forward ==')
        # print('x:',x.shape) # bs, 30 ,768
        input_shape = x.shape

        x = x.unsqueeze(1)
        # print('x unsqueeze:',x.shape)

        x_double = self.enlarge_layer(x)
        # print('x_double:',x_double.shape) # bs, 11520 =30/2 * 768/2

        mu, std = x_double[:, :self.input_dim * self.bottleneck_scale], \
            x_double[:, self.input_dim * self.bottleneck_scale:]
        # print(f"mu, std={mu.shape},{std.shape}") # bs, 23040   bs, 23040
        std = F.softplus(std - self.std_shift)  # ? F.softplus(std-0.5) F.softplus(std-5)
        # print("std:",std.shape)  # bs, 23040
        epsilon = torch.empty((x.size()[0], self.input_dim * self.bottleneck_scale))
        torch.nn.init.normal(epsilon, mean=0, std=1)  # epsilon is initialized
        epsilon = epsilon.to(x.device)
        # print('epsilon:',epsilon.shape) # bs, 11520 =30/2 * 768/2

        z = mu + std * epsilon
        z = z.to(x.device)
        # print('======== After Enlarge z:',z.shape) # bs, seq_leb//2 * embed_dim//2 5760

        z = z.reshape([x.size()[0], 1, self.seq_length // 2, self.embed_dim // 2])
        # print('recovered_z:',z.shape) # bs, seq_leb//2 , embed_dim//2  

        size = z.size()
        _, indices = self.pool(torch.empty(size[0], size[1], size[2] * 2, size[3] * 2))
        z = self.unpool(z, indices.to(x.device))
        # print('unpooled z:',z.shape) 

        z = self.decoder_layer(z)
        # print('decoded z:',z.shape) 

        z = z.reshape(input_shape)
        # print('reshape z:',z.shape) 

        mid_loss = self.mid_lambda * torch.mean(
            torch.sum((-0.5) * (1 + 2 * torch.log(std) - mu ** 2 - std ** 2), 1)) / (input_shape[1] * input_shape[2])
        # print('mid_loss:',mid_loss)

        # print('== MID Model Forward Over ==')

        return z, mid_loss


class MIDModelCNN_ConvTranspose2d(nn.Module):
    def __init__(self, seq_length, embed_dim, mid_lambda, bottleneck_scale=1, std_shift=0.5):
        super(MIDModelCNN_ConvTranspose2d, self).__init__()
        self.bottleneck_scale = bottleneck_scale

        self.input_dim = (seq_length // 2) * (embed_dim // 2)
        self.output_dim = seq_length * embed_dim

        self.embed_dim = embed_dim
        self.seq_length = seq_length

        self.mid_lambda = mid_lambda
        self.std_shift = std_shift

        # bs, seq_len, 768
        self.enlarge_layer = nn.Sequential(
            # 输入[1,28,28]
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(self.input_dim, 2 * self.input_dim * self.bottleneck_scale, bias=True),
            nn.ReLU(inplace=True)
        )

        self.decoder_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # print('== MID Model Forward ==')
        # print('x:',x.shape) # bs, 30 ,768
        input_shape = x.shape
        x = x.unsqueeze(1)
        # print('x unsqueeze:',x.shape)

        x_double = self.enlarge_layer(x)
        # print('x_double:',x_double.shape) # bs, 11520 =30/2 * 768/2

        mu, std = x_double[:, :self.seq_length * self.embed_dim * self.bottleneck_scale // 4], \
            x_double[:, self.seq_length * self.embed_dim * self.bottleneck_scale // 4:]
        # print(f"mu, std={mu.shape},{std.shape}") # bs, 23040   bs, 23040

        std = F.softplus(std - self.std_shift)  # ? F.softplus(std-0.5) F.softplus(std-5)
        # print("std:",std.shape)  # bs, 23040

        epsilon = torch.empty((x.size()[0], self.seq_length * self.embed_dim * self.bottleneck_scale // 4))
        torch.nn.init.normal(epsilon, mean=0, std=1)  # epsilon is initialized
        epsilon = epsilon.to(x.device)
        # print('epsilon:',epsilon.shape) # bs, 11520 =30/2 * 768/2

        z = mu + std * epsilon
        z = z.to(x.device)
        # print('======== After Enlarge z:',z.shape) # bs, seq_leb//2 * embed_dim//2 5760

        recovered_z = z.reshape([x.size()[0], 1, self.seq_length // 2, self.embed_dim // 2])
        # print('recovered_z:',recovered_z.shape) # bs, seq_leb//2 , embed_dim//2  

        z = self.decoder_layer(recovered_z)
        # print('decoded z:',z.shape) # bs, 23040

        z = z.reshape(input_shape)
        # print('reshape z:',z.shape) # bs, 23040

        mid_loss = self.mid_lambda * torch.mean(
            torch.sum((-0.5) * (1 + 2 * torch.log(std) - mu ** 2 - std ** 2), 1)) / (input_shape[1] * input_shape[2])
        # print('mid_loss:',mid_loss)

        # print('== MID Model Forward Over ==')

        return z, mid_loss


class MIDModel_SqueezeLinear(nn.Module):
    def __init__(self, seq_length, embed_dim, mid_lambda, squeeze_dim=124, bottleneck_scale=1, std_shift=0.5):
        super(MIDModel_SqueezeLinear, self).__init__()
        self.bottleneck_scale = bottleneck_scale
        self.input_dim = embed_dim
        self.output_dim = seq_length * embed_dim
        self.squeeze_dim = squeeze_dim

        self.seq_length = seq_length

        self.mid_lambda = mid_lambda
        self.std_shift = std_shift

        self.squeeze_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.squeeze_dim, bias=True),
            nn.ReLU(inplace=True)
        )
        self.enlarge_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_length * self.squeeze_dim, seq_length * self.squeeze_dim * 2 * bottleneck_scale, bias=True),
            nn.ReLU(inplace=True)
        )
        self.decoder_layer = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(seq_length * self.squeeze_dim * bottleneck_scale, self.output_dim, bias=True),
            nn.ReLU(inplace=True)
        )

        torch.nn.init.xavier_uniform_(self.squeeze_layer[0].weight)
        torch.nn.init.zeros_(self.squeeze_layer[0].bias)

        torch.nn.init.xavier_uniform_(self.enlarge_layer[1].weight)
        torch.nn.init.zeros_(self.enlarge_layer[1].bias)

        torch.nn.init.xavier_uniform_(self.decoder_layer[0].weight)
        torch.nn.init.zeros_(self.decoder_layer[0].bias)

    def forward(self, x):
        # print('== MID Model Forward ==')
        # print('x:',x.shape) # bs, 30 ,768
        input_shape = x.shape

        # epsilon = torch.empty((x.size()[0],x.size()[1]*self.bottleneck_scale))

        epsilon = torch.empty((x.size()[0], self.seq_length * self.squeeze_dim * self.bottleneck_scale))
        torch.nn.init.normal(epsilon, mean=0, std=1)  # epsilon is initialized
        epsilon = epsilon.to(x.device)
        # print('epsilon:',epsilon.shape) # bs, 30

        # print(f"[debug] in mid model, x.shape={x.shape}")
        # # x.size() = (batch_size, class_num)
        x = self.squeeze_layer(x)
        # print('x squeeze:',x.shape) # bs, 46080=30*768

        x_double = self.enlarge_layer(x)
        # print('x_double:',x_double.shape) # bs, 46080=30*768

        mu, std = x_double[:, :self.seq_length * self.squeeze_dim * self.bottleneck_scale], \
            x_double[:, self.seq_length * self.squeeze_dim * self.bottleneck_scale:]
        # print(f"mu, std={mu.shape},{std.shape}") # bs, 23040   bs, 23040

        std = F.softplus(std - self.std_shift)  # ? F.softplus(std-0.5) F.softplus(std-5)
        # print("std:",std.shape)  # bs, 23040

        z = mu + std * epsilon
        z = z.to(x.device)

        # print('z:',z.shape) # bs, 23040

        z = self.decoder_layer(z)
        # print('decoded z:',z.shape) # bs, 23040

        z = z.reshape(input_shape)

        # print('reshape z:',z.shape) # bs, 23040

        mid_loss = self.mid_lambda * torch.mean(
            torch.sum((-0.5) * (1 + 2 * torch.log(std) - mu ** 2 - std ** 2), 1)) / (input_shape[1] * input_shape[2])
        # print('mid_loss:',mid_loss)

        # print('== MID Model Forward Over ==')

        return z, mid_loss


class MIDModel_Linear(nn.Module):
    def __init__(self, seq_length, embed_dim, mid_lambda, bottleneck_scale=1, std_shift=0.5):
        super(MIDModel_Linear, self).__init__()
        self.bottleneck_scale = bottleneck_scale
        self.input_dim = embed_dim
        self.output_dim = embed_dim

        self.mid_lambda = mid_lambda
        self.std_shift = std_shift

        # self.drop_out_p = 0.2

        self.enlarge_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim * 2 * bottleneck_scale, bias=True),
            nn.ReLU(inplace=True)
        )
        self.decoder_layer = nn.Sequential(
            nn.Linear(self.input_dim * bottleneck_scale, self.output_dim, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # print('== MID Model Forward ==')
        # print('x:',x.shape) # bs, 30 ,768
        x = torch.tensor(x, dtype=torch.float32)
        input_shape = x.shape

        # epsilon = torch.empty((x.size()[0],x.size()[1]*self.bottleneck_scale))

        epsilon = torch.empty((x.size()[0], x.size()[1], x.size()[2] * self.bottleneck_scale))
        torch.nn.init.normal(epsilon, mean=0, std=1)  # epsilon is initialized
        epsilon = epsilon.to(x.device)
        # print('epsilon:',epsilon.shape) # bs, 30, 768

        # print(f"[debug] in mid model, x.shape={x.shape}")
        # # x.size() = (batch_size, class_num)
        x_double = self.enlarge_layer(x)
        # print('x_double:',x_double.shape) # bs, 30, 2*768

        mu, std = x_double[:, :, :self.input_dim * self.bottleneck_scale], x_double[:, :,
                                                                           self.input_dim * self.bottleneck_scale:]
        # print(f"mu, std={mu.shape},{std.shape}") # bs, 30, 768*bottleneck_scale   bs, 30, 768*bottleneck_scale

        std = F.softplus(std - self.std_shift)  # ? F.softplus(std-0.5) F.softplus(std-5)
        # print("std:",std.shape)  # bs, 30, 768*bottleneck_scale

        z = mu + std * epsilon
        z = z.to(x.device)

        # print('z:',z.shape) # bs, 30, 768*bottleneck_scale

        z = self.decoder_layer(z)
        # print('decoded z:',z.shape) # bs, 30, 768

        z = z.reshape(input_shape)

        # print('reshape z:',z.shape) # bs, 23040

        mid_loss = self.mid_lambda * torch.mean(
            torch.sum((-0.5) * (1 + 2 * torch.log(std) - mu ** 2 - std ** 2), 1)) / (input_shape[1] * input_shape[2])
        # print('mid_loss:',mid_loss)

        # print('== In mid model ==')
        # mark = 0
        # for name, param in self.enlarge_layer.named_parameters():
        #     if mark == 0:
        #         print(name, param.grad)
        #         mark = mark + 1

        # mid_loss.backward()

        # print('-'*25)
        # mark = 0
        # for name, param in self.enlarge_layer.named_parameters():
        #     if mark == 0:
        #         print(name, param.grad)
        #         mark = mark + 1
        # print('== In mid model ==')

        # assert 1>2

        return z, mid_loss


class MIDModel_PoolLinear(nn.Module):
    def __init__(self, seq_length, embed_dim, mid_lambda, bottleneck_scale=1, std_shift=0.5):
        super(MIDModel_PoolLinear, self).__init__()
        self.bottleneck_scale = bottleneck_scale
        self.input_dim = embed_dim
        self.output_dim = embed_dim

        self.mid_lambda = mid_lambda
        self.std_shift = std_shift

        self.enlarge_layer = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(self.input_dim, self.input_dim * 2 * bottleneck_scale, bias=True),
            nn.ReLU(inplace=True)
        )
        self.decoder_layer = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(self.input_dim * bottleneck_scale, self.input_dim * bottleneck_scale * 5, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_dim * bottleneck_scale * 5, self.output_dim, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, token_embeddings, attention_mask):
        # Get Sentence Embedding
        output_vectors = []
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
        )
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        max_over_time = torch.max(token_embeddings, 1)[0]
        output_vectors.append(max_over_time)

        output_vector = torch.cat(output_vectors, 1)

        x = token_embeddings

        input_shape = x.shape

        # epsilon = torch.empty((x.size()[0],x.size()[1]*self.bottleneck_scale))

        epsilon = torch.empty((x.size()[0], x.size()[1], x.size()[2] * self.bottleneck_scale))
        torch.nn.init.normal(epsilon, mean=0, std=1)  # epsilon is initialized
        epsilon = epsilon.to(x.device)
        # print('epsilon:',epsilon.shape) # bs, 30, 768

        # print(f"[debug] in mid model, x.shape={x.shape}")
        # # x.size() = (batch_size, class_num)
        x_double = self.enlarge_layer(x)
        # print('x_double:',x_double.shape) # bs, 30, 2*768

        mu, std = x_double[:, :, :self.input_dim * self.bottleneck_scale], x_double[:, :,
                                                                           self.input_dim * self.bottleneck_scale:]
        # print(f"mu, std={mu.shape},{std.shape}") # bs, 30, 768*bottleneck_scale   bs, 30, 768*bottleneck_scale

        std = F.softplus(std - self.std_shift)  # ? F.softplus(std-0.5) F.softplus(std-5)
        # print("std:",std.shape)  # bs, 30, 768*bottleneck_scale

        z = mu + std * epsilon
        z = z.to(x.device)

        # print('z:',z.shape) # bs, 30, 768*bottleneck_scale

        z = self.decoder_layer(z)
        # print('decoded z:',z.shape) # bs, 30, 768

        z = z.reshape(input_shape)

        # print('reshape z:',z.shape) # bs, 23040

        mid_loss = self.mid_lambda * torch.mean(
            torch.sum((-0.5) * (1 + 2 * torch.log(std) - mu ** 2 - std ** 2), 1)) / (input_shape[1] * input_shape[2])
        # print('mid_loss:',mid_loss)

        # print('== MID Model Forward Over ==')

        return z, mid_loss


class MID_model(nn.Module):
    def __init__(self, input_dim, output_dim, mid_lambda, bottleneck_scale=1, std_shift=0.5):
        super(MID_model, self).__init__()
        self.bottleneck_scale = bottleneck_scale
        self.input_dim = input_dim
        self.mid_lambda = mid_lambda
        self.std_shift = std_shift
        self.enlarge_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, input_dim * 2 * bottleneck_scale, bias=True),
            nn.ReLU(inplace=True)
        )
        self.decoder_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * bottleneck_scale, input_dim * bottleneck_scale * 5, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim * bottleneck_scale * 5, output_dim, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        epsilon = torch.empty((x.size()[0], x.size()[1] * self.bottleneck_scale))
        torch.nn.init.normal(epsilon, mean=0, std=1)  # epsilon is initialized
        epsilon = epsilon.to(x.device)
        # print(f"[debug] in mid model, x.shape={x.shape}")
        # # x.size() = (batch_size, class_num)
        x_double = self.enlarge_layer(x)  # bs, input_dim*2*bottleneck_scale
        mu, std = x_double[:, :self.input_dim * self.bottleneck_scale], x_double[:,
                                                                        self.input_dim * self.bottleneck_scale:]
        # print(f"mu, std={mu},{std}") # bs, input_dim*bottleneck_scale   bs, input_dim*bottleneck_scale
        std = F.softplus(std - self.std_shift)  # ? F.softplus(std-0.5) F.softplus(std-5)
        z = mu + std * epsilon
        z = z.to(x.device)
        z = self.decoder_layer(z)
        mid_loss = self.mid_lambda * torch.mean(torch.sum((-0.5) * (1 + 2 * torch.log(std) - mu ** 2 - std ** 2), 1))

        return z, mid_loss


class MID_model_small(nn.Module):
    def __init__(self, input_dim, output_dim, mid_lambda, bottleneck_scale=1, std_shift=0.5):
        super(MID_model_small, self).__init__()
        self.bottleneck_scale = bottleneck_scale
        self.input_dim = input_dim
        self.mid_lambda = mid_lambda
        self.std_shift = std_shift
        self.enlarge_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, input_dim * 2 * bottleneck_scale, bias=True),
            nn.ReLU(inplace=True)
        )
        self.decoder_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * bottleneck_scale, input_dim * bottleneck_scale * 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim * bottleneck_scale * 2, output_dim, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        epsilon = torch.empty((x.size()[0], x.size()[1] * self.bottleneck_scale))
        torch.nn.init.normal(epsilon, mean=0, std=1)  # epsilon is initialized
        epsilon = epsilon.to(x.device)
        # # x.size() = (batch_size, class_num)
        x_double = self.enlarge_layer(x)
        mu, std = x_double[:, :self.input_dim * self.bottleneck_scale], x_double[:,
                                                                        self.input_dim * self.bottleneck_scale:]
        # print(f"mu, std={mu},{std}")
        std = F.softplus(std - self.std_shift)  # ? F.softplus(std-0.5) F.softplus(std-5)
        z = mu + std * epsilon
        z = z.to(x.device)
        z = self.decoder_layer(z)
        mid_loss = self.mid_lambda * torch.mean(torch.sum((-0.5) * (1 + 2 * torch.log(std) - mu ** 2 - std ** 2), 1))

        return z, mid_loss


class Passive_local_MID_model(nn.Module):
    def __init__(self, local_model, mid_model):
        super(Passive_local_MID_model, self).__init__()
        assert local_model != None and mid_model != None
        self.local_model = local_model
        self.mid_model = mid_model
        self.mid_loss = None

    def forward(self, x):
        hp = self.local_model(x)  # hp: [bs, seq_length, embed_dim]
        z, self.mid_loss = self.mid_model(hp)
        return z


class Active_global_MID_model(nn.Module):
    def __init__(self, global_model, mid_model_list):
        super(Active_global_MID_model, self).__init__()
        assert global_model != None and len(mid_model_list) > 0
        self.global_model = global_model
        self.mid_model_list = mid_model_list
        self.mid_loss = None

    def forward(self, x):
        assert len(x) - 1 == len(self.mid_model_list)
        z_list = []
        self.mid_loss_list = []
        # give all passive party a mid_model
        for i in range(len(x) - 1):
            _z, mid_loss = self.mid_model_list[i](x[i])
            z_list.append(_z)
            self.mid_loss_list.append(mid_loss)
        # active party does not have mid_model
        z_list.append(x[-1])
        z = self.global_model(z_list)
        # print(f"active party mid global model, before_mid={x}, after_mid={z_list}, final_global_aggregation={z}")
        return z
