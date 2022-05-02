import torch
from torch.utils.data import Dataset

from data.utils import one_hot_encode, pad


class Dream(Dataset):
    def __init__(self, sequences, expression, transforms=None):

        pos_seqs = [pad(x, 80) for x in sequences]
        self.sequences = torch.stack([one_hot_encode(x) for x in pos_seqs])
        self.expression = torch.tensor(expression).float()
        self.transforms = transforms

    def __len__(self):
        return len(self.expression)

    def __getitem__(self, index):
        sequence = self.sequences[index, :]
        expression = self.expression[index, None]

        if self.transforms:
            sequence = self.transforms(sequence)

        return sequence, expression
