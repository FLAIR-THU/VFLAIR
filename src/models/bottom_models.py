import os, sys

sys.path.append(os.pardir)

from models.cnn import *
from models.gcn import *
from models.mid_model_rapper import *
from models.mlp import *
from models.regression import *
from models.resnet import *
from models.rnn import *
from models.adversarial_model import *


def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)
