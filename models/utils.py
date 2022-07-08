import argparse
import inspect
import random
import typing
from functools import reduce

import numpy as np
import scipy
import torch
import torch.nn as nn
from torch.nn.init import constant_, normal_, xavier_uniform_


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


class Residual(nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()

        self.module = module

    def forward(self, x):
        return x + self.module(x)


def conv_block(
    channels_in,
    channels_out,
    width=16,
    conv=nn.Conv1d,
    nb_repeats=3,
    dilation=1,
    padding="same",
):
    block = []

    for i in range(nb_repeats):
        pad = padding if i == 0 else "same"
        block.append(
            conv(channels_in, channels_out, width, padding=pad, dilation=dilation)
        )
        block.append(nn.BatchNorm1d(channels_out))
        block.append(nn.GELU())

        channels_in = channels_out

    block.append(nn.MaxPool1d(2))

    return nn.Sequential(*block)


def count_params(net):
    nb_params = 0
    for param in net.parameters():
        nb_params += reduce(lambda x, y: x * y, param.shape)
    return nb_params


def init_weights(layer, init="glorot"):

    layer_type = layer.__class__.__name__

    if "Conv" in layer_type or "Linear" in layer_type:
        if init == "glorot":
            xavier_uniform_(layer.weight.data)
        else:
            normal_(layer.weight.data, 0.0, 0.02)

        if hasattr(layer, "bias") and layer.bias is not None:
            constant_(layer.bias.data, 0.0)

    elif "BatchNorm" in layer_type:
        normal_(layer.weight.data, 1.0, 0.02)
        constant_(layer.bias.data, 0.0)


def parser(model):

    p = argparse.ArgumentParser()
    p.add_argument("-seed", default=0, type=int)

    # get other arguments from the signature
    model_args = inspect.getfullargspec(model)

    argnames = model_args.args[1:]
    defaults = model_args.defaults if argnames else []
    types = typing.get_type_hints(model.__init__)

    for arg, val in zip(argnames, defaults):
        p.add_argument(f"-{arg}", default=val, type=types[arg])

    return p
