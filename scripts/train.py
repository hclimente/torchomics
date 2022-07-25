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

import sys

# + tags=[]
from importlib import import_module
from pathlib import Path

import pytorch_lightning as pl
import torch
from git import Repo
from pyprojroot import here
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.functional import pearson_corrcoef, spearman_corrcoef

from data import Dream, DreamDM, save_preds
from models.utils import parser, parser_from_object

# + tags=[]
# hyperparameters
model_name = "SimpleCNN"
ARCH = getattr(import_module("models"), model_name)
BATCH_SIZE = 1024
VAL_SIZE = 10000
N_EPOCHS = 12

# setup
# create fake arguments if in interactive mode
sys.argv = ["train.py"] if hasattr(sys, "ps1") else sys.argv
opt_params, rest = parser().parse_known_args(sys.argv[1:])
opt_params = vars(opt_params)
model_params, rest = parser_from_object(ARCH).parse_known_args(rest)
model_params = vars(model_params)
loader_params = vars(parser_from_object(Dream).parse_args(rest))

# prepare logs path
logs_path = f"{here('results/models/')}/{model_name}/"

seed = opt_params.pop("seed")
sha = Repo(search_parent_directories=True).head.object.hexsha
version = sha[:5]
for k, v in (opt_params | model_params | loader_params).items():
    if type(v) is list:
        v = f"[{','.join(str(x) for x in v)}]" if v else "none"
    elif (k in loader_params.keys() or k == "weight_decay") and v == 0.0:
        continue

    version += f"-{k}={v}"

Path(f"{logs_path}/{version}/").mkdir(parents=True, exist_ok=True)


# + tags=[]
class Model(ARCH):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)

        if opt_params["loss"] == "mse":
            self.loss = torch.nn.MSELoss()
        elif opt_params["loss"] == "huber":
            self.loss = torch.nn.HuberLoss()

        kernel_size = kwargs.get("kernel_size", 0)
        self.example_input_array = torch.rand((1, 4, 80 + 2 * (kernel_size // 2)))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=0)
        return optimizer

    def on_train_start(self):
        # store hyperparameters
        hparams = {
            "batch_size": BATCH_SIZE,
            "model": model_name,
            "sha": sha,
            "seed": seed,
            **loader_params,
            **opt_params,
            **model_params,
        }

        self.logger.log_hyperparams(hparams, {"test/pearson": 0, "test/spearman": 0})

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        seq, rc, _ = batch
        return self(seq, rc)

    def step(self, batch, batch_idx, label):

        seq, rc, y = batch
        y_pred = self(seq, rc)
        loss = self.loss(y, y_pred)

        self.log(f"{label}/loss", loss)
        self.log(f"{label}/pearson", pearson_corrcoef(y, y_pred.float()))
        self.log(f"{label}/spearman", spearman_corrcoef(y, y_pred.float()))

        return loss

    def validation_epoch_end(self, val_step_outputs):
        avg_loss = torch.Tensor(val_step_outputs).mean()

        return {"val/loss": avg_loss}


# + tags=[]
if __name__ == "__main__":

    # setup
    pl.seed_everything(seed, workers=True)
    logger = TensorBoardLogger(
        save_dir=logs_path,
        name=version,
        version=seed,
        default_hp_metric=False,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    trainer = pl.Trainer(
        max_epochs=N_EPOCHS,
        callbacks=[checkpoint_callback, RichProgressBar()],
        logger=logger,
        # NOTE comment out next two lines in dev machines
        gpus=-1,
        strategy="ddp_find_unused_parameters_false",
        # resume_from_checkpoint=f"{logs_path}/version_X/checkpoints/last.ckpt",
        precision=16,
        deterministic=True,
    )
    dm = DreamDM(
        here("data/dream/"),
        BATCH_SIZE,
        VAL_SIZE,
        trainer.accelerator,
        loader_params=loader_params,
        model_params=model_params,
    )

    # training
    model = Model(**model_params)
    trainer.fit(model, dm)

    # predictions
    # On a single GPU: https://github.com/Lightning-AI/lightning/issues/8375
    torch.distributed.destroy_process_group()
    if trainer.global_rank == 0:
        trainer = pl.Trainer(
            callbacks=[checkpoint_callback, RichProgressBar()],
            logger=logger,
            accelerator="gpu",
            devices=1,
            max_epochs=1,
        )
        model = Model.load_from_checkpoint(
            checkpoint_callback.best_model_path, **model_params
        )
        trainer.test(model, datamodule=dm)
        preds = trainer.predict(model, datamodule=dm)
        save_preds(
            torch.vstack(preds),
            logger.log_dir,
            here("data/dream/sample_submission.json"),
        )
# -
