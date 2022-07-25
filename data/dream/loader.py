import os
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.distributions.beta import Beta
from torch.utils.data import DataLoader, Dataset

from data.transforms import cutmix, mixup, rand_erase
from data.utils import load, one_hot_encode


class Dream(Dataset):
    def __init__(
        self,
        sequences: torch.Tensor,
        expression: torch.Tensor,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        erase_alpha: float = 0.0,
    ):

        self.sequences = sequences
        self.expression = expression.float()

        self.mixup_dist = Beta(mixup_alpha, mixup_alpha) if mixup_alpha > 0 else None
        self.cutmix_dist = (
            Beta(cutmix_alpha, cutmix_alpha) if cutmix_alpha > 0 else None
        )
        self.rand_erase_dist = (
            Beta(erase_alpha, erase_alpha) if erase_alpha > 0 else None
        )

    def __len__(self):
        return len(self.expression)

    def __getitem__(self, idx):
        seq = self.sequences[idx, :]
        rc = self.rc_sequences[idx, :]
        expression = self.expression[idx, None]

        return self.apply_transforms(idx, seq, rc, expression)

    def cache_rc(self):
        self.rc_sequences = self.sequences.flip(1, 2)

    def apply_transforms(self, idx, seq, rc, expression):

        if self.mixup_dist and idx % 5 == 0:
            seq, rc, expression = mixup(seq, rc, expression, self.mixup_dist, self)

        if self.cutmix_dist:
            seq, rc, expression = cutmix(seq, rc, expression, self.cutmix_dist, self)

        if self.rand_erase_dist:
            seq, rc, expression = rand_erase(seq, rc, expression, self.rand_erase_dist)

        return seq, rc, expression


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

        self.pred = torch.load(f"{self.data_dir}/test.pt")
        self.pred.sequences = pad_sequences(self.pred)
        self.pred.cache_rc()

    def subset_data(self, subset, **kwargs):

        dataset = subset.dataset

        sequences = dataset.sequences[subset.indices]
        expression = dataset.expression[subset.indices]
        ds = Dream(sequences, expression, **kwargs)
        ds.cache_rc()

        return ds

    def train_dataloader(self):
        ds = self.subset_data(self.train, **self.loader_params)
        return DataLoader(ds, shuffle=True, drop_last=True, **self.params)

    def val_dataloader(self):
        ds = self.subset_data(self.val)
        return DataLoader(ds, **self.params)

    def test_dataloader(self):
        ds = self.subset_data(self.test)
        return DataLoader(ds, **self.params)

    def predict_dataloader(self):
        return DataLoader(self.pred, **self.params)
