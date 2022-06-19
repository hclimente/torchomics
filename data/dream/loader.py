import os
import random
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.distributions.beta import Beta
from torch.utils.data import DataLoader, Dataset

from data.utils import load


class Dream(Dataset):
    def __init__(
        self, sequences: torch.Tensor, expression: torch.Tensor, transforms=[]
    ):

        self.sequences = sequences
        self.expression = expression.float()
        self.transforms = transforms

        # beta distribution to sample mixup probabilities
        alpha = 0.2
        self.beta = Beta(alpha, alpha)

    def __len__(self):
        return len(self.expression)

    def __getitem__(self, idx):
        seq = self.sequences[idx, :]
        rc = self.rc_sequences[idx, :]
        expression = self.expression[idx, None]

        if self.transforms:
            seq, rc, expression = self.apply_transforms(idx, seq, rc, expression)

        return seq, rc, expression

    def cache_rc(self):
        self.rc_sequences = self.sequences.flip(1, 2)

    def apply_transforms(self, idx, seq, rc, expression):

        if "mixup" in self.transforms and idx % 5 == 0:
            # randomly select another sequence
            mixup_idx = random.randint(0, len(self) - 1)
            mixup_seq = self.sequences[mixup_idx]
            mixup_rc = self.rc_sequences[mixup_idx]
            mixup_expression = self.expression[mixup_idx]

            # sample a probability and mixup the sequences accordingly
            p = self.beta.sample()
            seq = p * seq + (1 - p) * mixup_seq
            rc = p * rc + (1 - p) * mixup_rc
            expression = p * expression + (1 - p) * mixup_expression

        return seq, rc, expression


class DreamDM(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "path/to/dir",
        batch_size: int = 32,
        val_size: int = 100,
        accelerator: pl.accelerators = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_size = val_size

        self.dev_machine = True
        if isinstance(accelerator, pl.accelerators.Accelerator):
            if type(accelerator).__name__ != "CPUAccelerator":
                self.dev_machine = False
        elif isinstance(accelerator, str):
            if accelerator != "cpu":
                self.dev_machine = False

        self.params = {
            "batch_size": batch_size,
            # NOTE most multiproc errors happen when num_workers is too large
            "num_workers": int(os.cpu_count() / 4),
            "persistent_workers": True,
        }

    def setup(self, stage: Optional[str] = None):

        tr_cached = "train_dev.pt" if self.dev_machine else "train.pt"
        tr = load("train_sequences.txt", tr_cached, Dream, path=self.data_dir)
        tr = Dream(tr.sequences, tr.expression)

        lengths = [len(tr) - 2 * self.val_size, self.val_size, self.val_size]
        self.train, self.val, self.test = torch.utils.data.random_split(tr, lengths)

        self.pred = torch.load(f"{self.data_dir}/test.pt")
        self.pred.cache_rc()

    def subset_data(self, subset, transforms=None):

        dataset = subset.dataset

        sequences = dataset.sequences[subset.indices]
        expression = dataset.expression[subset.indices]

        ds = Dream(sequences, expression, transforms=transforms)
        ds.cache_rc()

        return ds

    def train_dataloader(self):
        ds = self.subset_data(self.train, ["mixup"])
        return DataLoader(ds, shuffle=True, drop_last=True, **self.params)

    def val_dataloader(self):
        ds = self.subset_data(self.val)
        return DataLoader(ds, **self.params)

    def test_dataloader(self):
        ds = self.subset_data(self.test)
        return DataLoader(ds, **self.params)

    def predict_dataloader(self):
        return DataLoader(self.pred, **self.params)
