import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function


################### Adversarial Model: Privacy Preserving Mapping #####################
class Mapping_MLP2(nn.Module):
    '''
    input --- intermediate : bs, seq_length, 768(embed_dim)
    output --- embedding : bs, seq_length, 768(embed_dim)
    '''

    def __init__(self, seq_length, embed_dim, hidden_size=80):
        super(Mapping_MLP2, self).__init__()
        # print('Adversarial_MLP init:',seq_length, embed_dim)
        self.seq_length = seq_length
        self.embed_dim = embed_dim
       
        self.net1 = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        self.net2 = nn.Sequential(
            nn.Linear(hidden_size, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # origin_shape = x.shape
        # print('x:',x.shape,origin_shape)
        # x = torch.tensor(x,dtype=torch.float32)
        x1 = self.net1(x)
        # print('x1:',x1.shape)

        x2 = self.net2(x1)
        # print('x2:',x2.shape)

        # x2 = x2.reshape(origin_shape)
        return x2


class Mapping_MLP3(nn.Module):
    '''
    input --- intermediate : bs, seq_length, 768(embed_dim)
    output --- embedding : bs, seq_length, 768(embed_dim)
    '''

    def __init__(self, seq_length, embed_dim, hidden_size=80):
        super(Mapping_MLP3, self).__init__()
        # print('Adversarial_MLP init:',seq_length, embed_dim)
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        # self.batch_first = batch_first

        self.net1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_length * embed_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        self.net2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )

        self.net3 = nn.Sequential(
            nn.Linear(hidden_size, seq_length * embed_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print('=== ad model ===')
        origin_shape = x.shape 
        origin_dtype = x.dtype
        # print('x raw:',x.shape,x.dtype)
        # print('origin_shape:',origin_shape)

        # if not self.batch_first:
        if origin_shape[1] != self.seq_length:
            # print('batch_first:',self.batch_first)
            x = x.transpose(0,1) # should be [bs, seq_len, embed_dim]
            # print('x after:',x.shape,x.dtype)
        # print(self.seq_length,'  ',self.embed_dim)
        # print(self.seq_length*self.embed_dim)

        x = torch.tensor(x,dtype=torch.float32)
        x1 = self.net1(x)
        # print('x1:',x1.shape)

        x2 = self.net2(x1)
        # print('x2:',x2.shape)

        x3 = self.net3(x2)
        # print('x3:',x3.shape)

        x3 = x3.reshape(origin_shape)
        # print('x3:',x3.shape,x3.dtype)
        # print('=== ad model ===')

        return x3


class GradientReversal_function(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha * grad_output
        return grad_input, None


revgrad = GradientReversal_function.apply


class GradientReversal(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)
        self.revgrad = GradientReversal_function.apply

    def forward(self, x):
        return self.revgrad(x, self.alpha)


# a model rapper for local model with adversarial component
class Local_Adversarial_combined_model_Bert(nn.Module):
    def __init__(self, local_model, adversarial_model):
        super(Local_Adversarial_combined_model_Bert, self).__init__()
        assert local_model != None and adversarial_model != None
        self.local_model = local_model  # for normal training
        self.adversarial_model = adversarial_model
        self.origin_output = None
        self.adversarial_output = None
        # self.adversarial_loss = None

    def forward(self, **x):
        self.origin_output, self.origin_attention_mask = self.local_model(**x)
        self.adversarial_output = self.adversarial_model(self.origin_output)
        self.embedding_output = self.local_model.embedding_output
        return self.adversarial_output, self.origin_attention_mask


class Local_Adversarial_combined_model_GPT2(nn.Module):
    def __init__(self, local_model, adversarial_model):
        super(Local_Adversarial_combined_model_GPT2, self).__init__()
        assert local_model != None and adversarial_model != None
        self.local_model = local_model  # for normal training
        self.adversarial_model = adversarial_model
        self.origin_output = None
        self.adversarial_output = None
        # self.adversarial_loss = None

    def forward(self, **x):
        self.origin_output, self.origin_sequence_lengths, self.origin_attention_mask = self.local_model(**x)
        self.adversarial_output = self.adversarial_model(self.origin_output)

        self.embedding_output = self.local_model.embedding_output

        return self.adversarial_output, self.origin_sequence_lengths, self.origin_attention_mask


class Local_Adversarial_combined_model_Llama(nn.Module):
    def __init__(self, local_model, adversarial_model):
        super(Local_Adversarial_combined_model_Llama, self).__init__()
        assert local_model != None and adversarial_model != None
        self.local_model = local_model  # for normal training
        self.adversarial_model = adversarial_model
        self.origin_output = None
        self.adversarial_output = None
        # self.adversarial_loss = None

    def forward(self, **x):
        self.origin_output, self.origin_sequence_lengths, self.origin_attention_mask = self.local_model(**x)
        self.adversarial_output = self.adversarial_model(self.origin_output)

        self.embedding_output = self.local_model.embedding_output

        return self.adversarial_output, self.origin_sequence_lengths, self.origin_attention_mask


# a model rapper for local model with adversarial component
class Local_Adversarial_combined_model(nn.Module):
    def __init__(self, local_model, adversarial_model):
        super(Local_Adversarial_combined_model, self).__init__()
        assert local_model != None and adversarial_model != None
        self.local_model = local_model  # for normal training
        self.adversarial_model = adversarial_model
        self.adversarial_output = None
        # self.adversarial_loss = None

    def forward(self, x):
        out = self.local_model(x)
        self.adversarial_output = self.adversarial_model(out)
        return out


# for Attribute Inference adversarial
class Adversarial_MLP3(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(Adversarial_MLP3, self).__init__()
        self.drop_out_rate = 0.1
        self.grad_reverse = nn.Sequential(
            GradientReversal(alpha=1.),
            nn.Dropout(self.drop_out_rate),
        )
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(self.drop_out_rate),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(self.drop_out_rate),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim, bias=False),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm1d(output_dim),
            # nn.Dropout(self.drop_out_rate),
        )

    def forward(self, x):
        x = self.grad_reverse(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
