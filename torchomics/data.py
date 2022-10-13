import random

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from .transforms import Cutmix, Mixup, Mutate, RandomErase
from .utils import one_hot_encode


class OmicsDataset(Dataset):
    def __init__(
        self,
        sequences: torch.Tensor = None,
        labels: torch.Tensor = None,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        erase_alpha: float = 0.0,
        n_mutations: int = 0,
    ):

        self.sequences = sequences
        self.labels = labels

        # TODO accept an arbitrary list of transforms
        self.transforms = [
            Mixup(mixup_alpha),
            Cutmix(cutmix_alpha),
            RandomErase(erase_alpha),
            Mutate(n_mutations),
        ]
        self.transforms = [t for t in self.transforms if t]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx, :], self.labels[idx]

    def collate_fn(self, batch, collate=True):

        # https://github.com/pytorch/vision/blob/main/references/classification/transforms.py
        # https://github.com/pytorch/vision/blob/main/references/classification/train.py

        if collate:
            batch = default_collate(batch)

        if self.transforms:
            t = random.choice(self.transforms)
            batch = t(*batch)

        return batch


def import_genomics_benchmark(obj, *args, **kwargs):
    seqs = [s for s, _ in obj]
    max_length = max([len(s) for s in seqs])
    seqs = torch.stack([one_hot_encode(s, padding=max_length) for s in seqs])

    labels = torch.Tensor([y for _, y in obj])

    return OmicsDataset(seqs, labels, *args, **kwargs)
