import random

import numpy as np
import torch

from torchomics import one_hot_encode
from torchomics.models.cnn import SimpleCNN

input_length = 110
cnn = SimpleCNN()


def test_forward():

    seq = one_hot_encode(
        "".join(random.choice(["A", "C", "G", "T"]) for _ in range(input_length))
    )
    seq = seq[None, :]
    seq3 = torch.vstack((seq, seq, seq))

    assert np.all(list(cnn.forward(seq).shape) == [1, 1])
    assert np.all(list(cnn.forward(seq3).shape) == [3, 1])


def test_str():

    assert cnn.__class__.__name__ == "SimpleCNN"
