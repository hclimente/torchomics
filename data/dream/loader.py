import torch
from torch.utils.data import Dataset

from data.transforms import ReverseComplement
from data.utils import one_hot_encode, pad


class Dream(Dataset):
    def __init__(self, sequences, expression, transforms=None):

        pos_seqs = [pad(x, 80) for x in sequences]
        self.sequences = torch.stack([one_hot_encode(x) for x in pos_seqs])
        self.expression = torch.tensor(expression).float()
        self.transforms = transforms

        rc = ReverseComplement()
        self.rc_sequences = rc(self.sequences)

    def __len__(self):
        return len(self.expression)

    def __getitem__(self, index):
        seq = self.sequences[index, :]
        rc = self.rc_sequences[index, :]
        expression = self.expression[index, None]

        if self.transforms:
            seq = self.transforms(seq)

        return seq, rc, expression
