import random

import numpy as np
import torch

from data import one_hot_encode
from models import DenseNet

input_length = 80
net1 = DenseNet()
net2 = DenseNet(bottleneck=True)


def test_forward():

    seq = one_hot_encode(
        "".join(random.choice(["A", "C", "G", "T"]) for _ in range(input_length))
    )
    seq = seq[None, :]
    seq3 = torch.vstack((seq, seq, seq))

    assert np.all(list(net1.forward(seq, seq).shape) == [1, 1])
    assert np.all(list(net1.forward(seq3, seq3).shape) == [3, 1])

    assert np.all(list(net2.forward(seq, seq).shape) == [1, 1])
    assert np.all(list(net2.forward(seq3, seq3).shape) == [3, 1])


def test_str():

    assert net1.__class__.__name__ == "DenseNet"
    assert net2.__class__.__name__ == "DenseNet"
