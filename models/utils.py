import random

import numpy as np
import scipy
import torch


def fix_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def pearsonr(x, y):

    x = x.cpu().detach().float().numpy().flatten()
    y = y.cpu().detach().float().numpy().flatten()

    return scipy.stats.pearsonr(x, y)[0]


def spearmanr(x, y):

    x = x.cpu().detach().float().numpy()
    y = y.cpu().detach().float().numpy()

    return scipy.stats.spearmanr(x, y)[0]


class Residual(torch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()

        self.module = module

    def forward(self, x):
        return x + self.module(x)
