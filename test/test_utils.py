import numpy as np

from torchomics import one_hot_encode


def test_one_hot_encode():

    a = [[1, 0, 0, 0]]
    c = [[0, 1, 0, 0]]
    g = [[0, 0, 1, 0]]
    t = [[0, 0, 0, 1]]

    assert np.all(np.array(one_hot_encode("A")).T == a)
    assert np.all(np.array(one_hot_encode("C")).T == c)
    assert np.all(np.array(one_hot_encode("G")).T == g)
    assert np.all(np.array(one_hot_encode("T")).T == t)

    assert np.all(
        np.array(one_hot_encode("AAGCTC")).T == np.concatenate((a, a, g, c, t, c))
    )
