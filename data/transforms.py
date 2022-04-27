from random import random

import torch


class Mutate:
    def __init__(self, n=1):
        self.n = n

    def __call__(self, seq):
        length = seq.shape[1]

        pos = torch.randint(high=length, size=(self.n,))
        perm = torch.vstack([torch.randperm(4) for _ in range(self.n)]).T
        seq[:, pos] = seq[perm, pos]

        return seq


class ReverseComplement:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, seq):
        return torch.flip(seq, (0, 1)) if random() < self.p else seq


class MixUp:
    def __init__(self, beta=0.9):
        self.beta = beta

    def __call__(self, x, y):
        return self.beta * x + (1 - self.beta) * y
