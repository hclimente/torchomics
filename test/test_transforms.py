import random

import torch
from Bio.Seq import Seq

from data import MixUp, Mutate, ReverseComplement, one_hot_encode

seq_len = 100
seq = "".join(random.choice(["A", "C", "G", "T"]) for _ in range(seq_len))
seq_1h = one_hot_encode(seq)
seq_1h = seq_1h[None, :]


def test_mutate():

    i = 0
    for n_mutations in range(1, 5):
        m = Mutate(n_mutations)
        for _ in range(10):
            mut = m(seq_1h.clone())
            diff = torch.sum(mut != seq_1h) / 2

            assert diff <= n_mutations

            i += diff

    assert i > 0


def test_reverseComplement():

    # work on three sequences
    seq3_1h = torch.vstack((seq_1h, seq_1h, seq_1h))

    rc_1h = one_hot_encode(Seq(seq).reverse_complement())
    rc_1h = rc_1h[None, :]
    rc3_1h = torch.vstack((rc_1h, rc_1h, rc_1h))

    r = ReverseComplement(1)

    assert torch.all(rc3_1h == r(seq3_1h))


def test_mixup():

    seq2_1h = one_hot_encode(
        "".join(random.choice(["A", "C", "G", "T"]) for _ in range(seq_len))
    )

    m = MixUp()
    mixup = m(seq_1h, seq2_1h)

    assert seq_1h.shape == mixup.shape
    assert torch.all(seq_1h == torch.logical_or(mixup == 1, mixup == 0.9))
    assert torch.all(seq2_1h == torch.logical_or(mixup == 1, mixup == 0.1))
