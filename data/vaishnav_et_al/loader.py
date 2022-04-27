import torch
from torch.utils.data import Dataset
from torchvision import transforms

from models.utils import one_hot_encode
from data import ReverseComplement


class Vaishnav(Dataset):
    def __init__(
        self, sequences, expression, transform=transforms.Compose(ReverseComplement())
    ):

        pos_seqs = [pad(x, 110) for x in sequences]
        self.sequences = torch.stack([one_hot_encode(x) for x in pos_seqs])
        self.expression = torch.tensor(expression).float()
        self.transforms = transform

    def __len__(self):
        return min(6000000, len(self.expression))

    def __getitem__(self, index):
        sequence = self.sequences[index, :]
        expression = self.expression[index, None]

        if self.transforms:
            sequence = self.transforms(sequence)

        return sequence, expression


def pad(seq, expected):
    if len(seq) == expected:
        return seq
    if len(seq) < expected:
        tail = "".join("N" for _ in range(expected - len(seq)))
        return seq + tail
    if len(seq) > expected:
        return seq[:expected]
