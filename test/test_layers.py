import random

import numpy as np
import torch

from torchomics import one_hot_encode
from torchomics.layers import RevCompConv1D

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
