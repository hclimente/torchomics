import random
from functools import reduce

import numpy as np
import scipy
import torch
from torch.nn.init import constant_, normal_, xavier_uniform


def fix_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def pearsonr(x, y):

    x = numpify(x).flatten()
    y = numpify(y).flatten()

    r = scipy.stats.pearsonr(x, y)[0]
    r = -100 if np.isnan(r) else r

    return r


def spearmanr(x, y):

    x = numpify(x)
    y = numpify(y)

    r = scipy.stats.spearmanr(x, y)[0]
    r = -100 if np.isnan(r) else r

    return r


def numpify(x):
    return x.cpu().detach().float().numpy()


class Residual(torch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()

        self.module = module

    def forward(self, x):
        return x + self.module(x)


def count_params(net):
    nb_params = 0
    for param in net.parameters():
        nb_params += reduce(lambda x, y: x * y, param.shape)
    return nb_params


def init_weights(layer, init="glorot"):

    layer_type = layer.__class__.__name__

    if "Conv" in layer_type or "Linear" in layer_type:
        if init == "glorot":
            xavier_uniform(layer.weight.data)
        else:
            normal_(layer.weight.data, 0.0, 0.02)

        if hasattr(layer, "bias") and layer.bias is not None:
            constant_(layer.bias.data, 0.0)

    elif "BatchNorm" in layer_type:
        normal_(layer.weight.data, 1.0, 0.02)
        constant_(layer.bias.data, 0.0)
