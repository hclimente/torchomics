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

# + tags=[]
from importlib import import_module
from pathlib import Path

import pytorch_lightning as pl
import torch
from pyprojroot import here
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.functional import pearson_corrcoef, spearman_corrcoef

from data import DreamDM, save_preds

# + tags=[]
# hyperparameters
model_name = "SimpleCNN"
ARCH = getattr(import_module("models"), model_name)
BATCH_SIZE = 1024
VAL_SIZE = 10000
N_EPOCHS = 2

# setup
all_logs = here("results/models/")
logs_path = f"{all_logs}/{model_name}"
Path(logs_path).mkdir(exist_ok=True)


# + tags=[]
class Model(ARCH):
    def __init__(self):
        super(Model, self).__init__()
        self.loss = torch.nn.MSELoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "Train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "Validation")

    def testing_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "Test")

    def step(self, batch, batch_idx, label):

        seq, rc, y = batch
        y_pred = self(seq, rc)
        loss = self.loss(y_pred, y)

        self.log(f"{label} loss", loss)
        self.log(f"{label} Pearson", pearson_corrcoef(y, y_pred.float()))
        self.log(f"{label} Spearman", spearman_corrcoef(y, y_pred.float()))

        return loss

    def validation_epoch_end(self, val_step_outputs):
        avg_loss = torch.Tensor(val_step_outputs).mean()

        return {"val_loss": avg_loss}


# + tags=[]
if __name__ == "__main__":
    # setup
    pl.seed_everything(0, workers=True)
    logger = TensorBoardLogger(save_dir=here("results/models/"), name=model_name)
    trainer = pl.Trainer(
        max_epochs=N_EPOCHS,
        callbacks=RichProgressBar(),
        logger=logger,
        gpus=-1,
        precision=16,
        deterministic=True,
    )
    dm = DreamDM(here("data/dream/"), BATCH_SIZE, VAL_SIZE, trainer.accelerator)

    # training
    model = Model()
    trainer.fit(model, dm)

    # predictions
    preds = trainer.predict(ckpt_path="best", datamodule=dm)
    preds = torch.cat(preds)
    save_preds(preds, here("data/dream/test_sequences.txt"), logger.log_dir)
# -

# examine model
# %reload_ext tensorboard
# %tensorboard --logdir=$all_logs
