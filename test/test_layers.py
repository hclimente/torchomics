import random

import numpy as np
import torch

from data import one_hot_encode
from models import SimpleCNN_RC
from models.layers import RevCompConv1D

input_length = 80
layer = RevCompConv1D(4, 20, 15, padding="same")


def test_forward():

    seq = one_hot_encode(
        "".join(random.choice(["A", "C", "G", "T"]) for _ in range(input_length))
    )
    seq = seq[None, :]
    seq3 = torch.vstack((seq, seq, seq))

    assert np.all(list(layer.forward(seq).shape) == [1, 20, 80])
    assert np.all(list(layer.forward(seq3).shape) == [3, 20, 80])


def test_net_forward():

    cnn = SimpleCNN_RC()

    seq = one_hot_encode(
        "".join(random.choice(["A", "C", "G", "T"]) for _ in range(input_length))
    )
    seq = seq[None, :]
    seq3 = torch.vstack((seq, seq, seq))

    assert np.all(list(cnn.forward(seq).shape) == [1, 1])
    assert np.all(list(cnn.forward(seq3).shape) == [3, 1])
