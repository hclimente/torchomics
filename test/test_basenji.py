import random

import numpy as np
import torch

from torchomics import Basenji, one_hot_encode

input_length = 80
cnn = Basenji()


def test_forward():

    seq = one_hot_encode(
        "".join(random.choice(["A", "C", "G", "T"]) for _ in range(input_length))
    )
    seq = seq[None, :]
    seq3 = torch.vstack((seq, seq, seq))

    assert np.all(list(cnn.forward(seq).shape) == [1, 1])
    assert np.all(list(cnn.forward(seq3).shape) == [3, 1])


def test_str():

    assert cnn.__class__.__name__ == "Basenji"
