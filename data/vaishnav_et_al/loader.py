from random import random

from Bio.Seq import Seq
import torch
from torch.utils.data import Dataset

from models.utils import one_hot_encode


class Vaishnav(Dataset):
    def __init__(self, sequences, expression):

        pos_seqs = [pad(x, 110) for x in sequences]
        neg_seqs = [Seq(x).reverse_complement() for x in pos_seqs]
        self.positive_sequences = torch.stack([one_hot_encode(x) for x in pos_seqs])
        self.negative_sequences = torch.stack([one_hot_encode(x) for x in neg_seqs])
        self.expression = torch.tensor(expression).float()

    def __len__(self):
        return min(6000000, len(self.expression))

    def __getitem__(self, index):
        return (
            self.positive_sequences[index, :]
            if random() < 0.5
            else self.negative_sequences_sequences[index, :],
            self.expression[index, None],
        )


def pad(seq, expected):
    if len(seq) == expected:
        return seq
    if len(seq) < expected:
        tail = "".join("N" for _ in range(expected - len(seq)))
        return seq + tail
    if len(seq) > expected:
        return seq[:expected]
