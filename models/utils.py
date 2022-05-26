import random
from functools import reduce

import numpy as np
import scipy
import torch


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
