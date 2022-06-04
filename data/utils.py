from os.path import isfile
from typing import Any

import numpy as np
import pandas as pd
import torch


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


def load(table, cached, obj, path="data/dream", sep="\t"):

    cached = f"{path}/{cached}"

    if isfile(cached):
        ds = torch.load(cached)
    else:
        sequences = pd.read_csv(
            f"{path}/{table}",
            sep=sep,
            names=["seq", "expr"],
        )
        ds = obj(sequences.seq, torch.Tensor(sequences.expr))
        torch.save(ds, cached)

    return ds


def one_hot_encode(
    sequence: str,
    alphabet: str = "ACGT",
    neutral_alphabet: str = "N",
    neutral_value: Any = 0,
    dtype=np.float32,
) -> np.ndarray:
    """One-hot encode sequence."""

    def to_uint8(string):
        return np.frombuffer(string.encode("ascii"), dtype=np.uint8)

    hash_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
    hash_table[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
    hash_table[to_uint8(neutral_alphabet)] = neutral_value
    hash_table = hash_table.astype(dtype)
    return torch.tensor(hash_table[to_uint8(sequence)]).T
