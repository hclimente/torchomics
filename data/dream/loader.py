import torch
from torch.utils.data import Dataset


class Dream(Dataset):
    def __init__(
        self, sequences: torch.Tensor, expression: torch.Tensor, transforms=None
    ):

        self.sequences = sequences
        self.expression = expression.float()
        self.transforms = transforms

    def __len__(self):
        return len(self.expression)

    def __getitem__(self, index):
        seq = self.sequences[index, :]
        rc = self.rc_sequences[index, :]
        expression = self.expression[index, None]

        if self.transforms:
            seq = self.transforms(seq)

        return seq, rc, expression

    def cache_rc(self):
        self.rc_sequences = self.sequences.flip(1, 2)
