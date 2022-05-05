import random

import numpy as np
import scipy
import torch


def fix_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def pearsonr(x, y):

    x = x.cpu().detach().numpy().flatten()
    y = y.cpu().detach().numpy().flatten()

    return scipy.stats.pearsonr(x, y)[0]


def spearmanr(x, y):

    x = x.cpu().detach().numpy()
    y = y.cpu().detach().numpy()

    return scipy.stats.spearmanr(x, y)[0]
