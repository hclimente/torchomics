from functools import reduce
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import constant_, normal_, xavier_uniform_


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


def pad(seq, expected):

    if expected is None:
        return seq

    if len(seq) == expected:
        return seq
    if len(seq) < expected:
        tail = "".join("N" for _ in range(expected - len(seq)))
        return seq + tail
    if len(seq) > expected:
        return seq[:expected]


def one_hot_encode(
    sequence: str,
    alphabet: str = "ACGT",
    neutral_alphabet: str = "N",
    neutral_value: Any = 0,
    dtype=np.float32,
    padding: float = None,
) -> torch.tensor:
    """One-hot encode sequence."""

    sequence = pad(sequence, padding)

    def to_uint8(string):
        return np.frombuffer(string.encode("ascii"), dtype=np.uint8)

    hash_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
    hash_table[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
    hash_table[to_uint8(neutral_alphabet)] = neutral_value
    hash_table = hash_table.astype(dtype)
    return torch.tensor(hash_table[to_uint8(sequence)]).T
