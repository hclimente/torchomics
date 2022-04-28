from os.path import isfile

import pandas as pd
import torch
from torch.utils.data import Dataset

from models.utils import one_hot_encode


class Vaishnav(Dataset):
    def __init__(self, sequences, expression, transforms=None):

        pos_seqs = [pad(x, 80) for x in sequences]
        self.sequences = torch.stack([one_hot_encode(x) for x in pos_seqs])
        self.expression = torch.tensor(expression).float()
        self.transforms = transforms

    def __len__(self):
        return min(6000000, len(self.expression))

    def __getitem__(self, index):
        sequence = self.sequences[index, :]
        expression = self.expression[index, None]

        if self.transforms:
            sequence = self.transforms(sequence)

        return sequence, expression


def pad(seq, expected):

    # remove primers
    seq = seq.removeprefix("TGCATTTTTTTCACATC")
    seq = seq.removesuffix("GGTTACGGCTGTT")

    if len(seq) == expected:
        return seq
    if len(seq) < expected:
        tail = "".join("N" for _ in range(expected - len(seq)))
        return seq + tail
    if len(seq) > expected:
        return seq[:expected]


def load_vaishnav(table, cached, sep="\t"):

    cached = f"data/vaishnav_et_al/{cached}"

    if isfile(cached):
        ds = torch.load(cached)
    else:
        sequences = pd.read_csv(
            f"data/vaishnav_et_al/{table}",
            # nrows=train_size,
            sep=sep,
            names=["seq", "expr"],
        )
        ds = Vaishnav(sequences.seq, sequences.expr)
        torch.save(ds, cached)

    return ds
