import random

import numpy as np
import torch

from torchomics import DenseNet, one_hot_encode

input_length = 80
no_bottleneck = DenseNet()
bottleneck = DenseNet(bottleneck=True)


def test_forward():

    seq = one_hot_encode(
        "".join(random.choice(["A", "C", "G", "T"]) for _ in range(input_length))
    )
    seq = seq[None, :]
    seq3 = torch.vstack((seq, seq, seq))

    assert np.all(list(no_bottleneck.forward(seq).shape) == [1, 1])
    assert np.all(list(no_bottleneck.forward(seq3).shape) == [3, 1])

    assert np.all(list(bottleneck.forward(seq).shape) == [1, 1])
    assert np.all(list(bottleneck.forward(seq3).shape) == [3, 1])


def test_str():

    assert no_bottleneck.__class__.__name__ == "DenseNet"
    assert bottleneck.__class__.__name__ == "DenseNet"
