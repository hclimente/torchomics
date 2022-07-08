import os
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from data.utils import load, one_hot_encode


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


class DreamDM(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "path/to/dir",
        batch_size: int = 32,
        val_size: int = 100,
        accelerator: pl.accelerators = None,
        hparams: dict = dict(),
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_size = val_size
        self.kernel_size = hparams.get("kernel_size", 0)

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
        if self.kernel_size != 0:
            head = one_hot_encode("TGCATTTTTTTCACATC")
            head = head[:, -(self.kernel_size - 1) // 2 :]
            head = head.repeat((len(tr), 1, 1))

            tail = one_hot_encode("GGTTACGGCTGTT")
            tail = tail[:, 0 : (self.kernel_size - 1) // 2]
            tail = tail.repeat((len(tr), 1, 1))

            tr.sequences = torch.cat((head, tr.sequences, tail), axis=2)

        tr.cache_rc()

        lengths = [len(tr) - 2 * self.val_size, self.val_size, self.val_size]
        self.train, self.val, self.test = torch.utils.data.random_split(tr, lengths)

        self.pred = torch.load(f"{self.data_dir}/test.pt")
        self.pred.cache_rc()

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, drop_last=True, **self.params)

    def val_dataloader(self):
        return DataLoader(self.val, **self.params)

    def test_dataloader(self):
        return DataLoader(self.test, **self.params)

    def predict_dataloader(self):
        return DataLoader(self.pred, **self.params)
