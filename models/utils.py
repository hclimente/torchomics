import random

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

    return scipy.stats.pearsonr(x, y)[0]


def spearmanr(x, y):

    x = numpify(x)
    y = numpify(y)

    return scipy.stats.spearmanr(x, y)[0]


def numpify(x):
    return x.cpu().detach().float().numpy()


class Residual(torch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()

        self.module = module

    def forward(self, x):
        return x + self.module(x)
