# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import os

# + tags=[]
from importlib import import_module
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from pyprojroot import here
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchmetrics.functional import pearson_corrcoef, spearman_corrcoef

from data import Dream, load

pl.seed_everything(0)

# + tags=[]
# hyperparameters
model_name = "SimpleCNN"
arch = getattr(import_module("models"), model_name)
batch_size = 1024
val_size = 10000

# setup
logs_path = here(f"results/models/{model_name}")
Path(logs_path).mkdir(exist_ok=True)
logger = TensorBoardLogger(save_dir=here("results/models/"), name=model_name)
trainer = pl.Trainer(max_epochs=20, callbacks=RichProgressBar(), logger=logger)
num_workers = int(os.cpu_count() / 2)


# + tags=[]
class Model(arch):
    def __init__(self):
        super(Model, self).__init__()
        self.loss = torch.nn.MSELoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer

    def training_step(self, batch, batch_idx):

        seq, rc, y = batch
        y_pred = self(seq, rc)
        loss = self.loss(y_pred, y)

        self.log("tr_r", pearson_corrcoef(y, y_pred))
        self.log("tr_rho", spearman_corrcoef(y, y_pred))

        return loss

    def validation_step(self, batch, batch_idx):

        seq, rc, y = batch
        y_pred = self(seq, rc)
        loss = self.loss(y_pred, y)

        self.log("val_r", pearson_corrcoef(y, y_pred))
        self.log("val_rho", spearman_corrcoef(y, y_pred))

        return loss

    def validation_epoch_end(self, val_step_outputs):
        avg_loss = torch.Tensor(val_step_outputs).mean()

        return {"val_loss": avg_loss}

    def setup(self, stage=None):
        tr_cached = "train_dev.pt"  # if device.type == "cpu" else "train.pt"
        tr = load("train_sequences.txt", tr_cached, Dream, path=here("data/dream"))
        tr.cache_rc()
        self.tr, self.val = torch.utils.data.random_split(
            tr, [len(tr) - val_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.tr,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
        )

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=val_size, num_workers=num_workers)


# + tags=[]
# training
model = Model()
trainer.fit(model)

# +
# save predictions on the test set
test = torch.load(here("data/dream/test_one_hot.pt"))
te_pred = model(test)

sequences = pd.read_csv(
    here("data/dream/test_sequences.txt"),
    sep="\t",
    names=["seq", "expr"],
)

sequences["expr"] = te_pred.detach().numpy()
sequences.to_csv(
    f"{logger.log_dir}/predictions.csv",
    header=False,
    index=False,
    sep="\t",
)

# + tags=[]
# examine model
logs = here("results/models/")
# %reload_ext tensorboard
# %tensorboard --logdir=$tb_logs
