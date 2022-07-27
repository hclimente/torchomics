import os
import random
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

from data.transforms import Cutmix, Mixup, Mutate, RandomErase
from data.utils import load, one_hot_encode


class Dream(Dataset):
    def __init__(
        self,
        sequences: torch.Tensor = None,
        expression: torch.Tensor = None,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        erase_alpha: float = 0.0,
        n_mutations: int = 0,
    ):

        self.sequences = sequences
        self.expression = expression.float()

        self.transforms = [
            Mixup(mixup_alpha, self),
            Cutmix(cutmix_alpha, self),
            RandomErase(erase_alpha),
            Mutate(n_mutations),
        ]
        self.transforms = [t for t in self.transforms if t]

    def __len__(self):
        return len(self.expression)

    def __getitem__(self, idx):
        seq = self.sequences[idx, :]
        rc = self.rc_sequences[idx, :]
        expression = self.expression[idx, None]

        return seq, rc, expression

    def cache_rc(self):
        self.rc_sequences = self.sequences.flip(1, 2)

    def transform(self, batch, collate=True):

        # https://github.com/pytorch/vision/blob/main/references/classification/transforms.py
        # https://github.com/pytorch/vision/blob/main/references/classification/train.py

        if collate:
            batch = default_collate(batch)

        if self.transforms:
            t = random.choice(self.transforms)
            batch = t(*batch)

        return batch


class DreamDM(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "path/to/dir",
        batch_size: int = 32,
        val_size: int = 100,
        accelerator: pl.accelerators = None,
        loader_params: dict = dict(),
        model_params: dict = dict(),
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_size = val_size
        self.kernel_size = model_params.get("kernel_size", 0)
        self.loader_params = loader_params

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

        # pad sequences with the real sequence if a kernel_size is provided
        def pad_sequences(d):
            if self.kernel_size != 0:
                head = one_hot_encode("TGCATTTTTTTCACATC")
                head = head[:, -(self.kernel_size - 1) // 2 :]
                head = head.repeat((len(d), 1, 1))

                tail = one_hot_encode("GGTTACGGCTGTT")
                tail = tail[:, 0 : (self.kernel_size - 1) // 2]
                tail = tail.repeat((len(d), 1, 1))

                return torch.cat((head, d.sequences, tail), axis=2)
            else:
                return d.sequences

        tr.sequences = pad_sequences(tr)
        tr.cache_rc()

        lengths = [len(tr) - 2 * self.val_size, self.val_size, self.val_size]
        self.train, self.val, self.test = torch.utils.data.random_split(tr, lengths)
        self.train = self.subset_data(self.train)
        self.val = self.subset_data(self.val, **self.loader_params)
        self.test = self.subset_data(self.test, **self.loader_params)

        pred = torch.load(f"{self.data_dir}/test.pt")
        pred.sequences = pad_sequences(pred)

        self.pred = Dream(pred.sequences, pred.expression, **self.loader_params)
        self.pred.cache_rc()

    def subset_data(self, subset, **kwargs):

        dataset = subset.dataset

        sequences = dataset.sequences[subset.indices]
        expression = dataset.expression[subset.indices]
        ds = Dream(sequences, expression, **kwargs)
        ds.cache_rc()

        return ds

    def train_dataloader(self):
        return DataLoader(
            self.train,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train.transform,
            **self.params,
        )

    def val_dataloader(self):
        return DataLoader(self.val, **self.params)

    def test_dataloader(self):
        return DataLoader(self.test, **self.params)

    def predict_dataloader(self):
        return DataLoader(self.pred, **self.params)
