# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
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
import os
import warnings
from random import random

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from git import Repo
from pyprojroot import here
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms

from data import Dream, load
from models import ResNet, fix_seeds
from models.utils import init_weights, numpify, pearsonr, spearmanr

if "GCP_PROJECT" in os.environ:
    from contextlib import nullcontext

    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    autocast = nullcontext()
else:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    autocast = torch.autocast(device_type=device.type)

if device == torch.device("cpu"):
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

# save the current commit hash iff the repo has no un-committed changes
repo = Repo(search_parent_directories=True)
sha = None if repo.is_dirty() else repo.head.object.hexsha

if not sha:
    warnings.warn("Uncommitted changes. The model parameters won't be saved.")
# -

# # Model and hyperparameters

# + tags=["parameters"]
# model hyperparameters
model_obj = ResNet
n_epochs = 10
batch_size = 1024
seed = 0

# data transforms
rc_transform = False

fix_seeds(seed)

# + tags=[]
# model specification
model = model_obj().to(device)
model_name = f"{model.__class__.__name__}_epochs={n_epochs}_batch={batch_size}"

print(summary(model))

# initialise model weights
model.apply(init_weights)

# initialize last layer's bias to the average
with torch.no_grad():
    model.fc[-1].bias[0] = torch.tensor(11.147).to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
# -

# # Data loading

# + tags=[]
# train and validation
val_size = 10000

tr_cached = "train_dev.pt" if device.type == "cpu" else "train.pt"
tr = load("train_sequences.txt", tr_cached, Dream, path=here("data/dream"))
tr.cache_rc()
tr, val = torch.utils.data.random_split(tr, [len(tr) - val_size, val_size])

val_loader = DataLoader(val, batch_size=val_size)
val_seqs, val_rc, val_expression = next(iter(val_loader))

# test (unlabelled)
te = load("test_sequences.txt", "test.pt", Dream, path=here("data/dream"))
te.cache_rc()

# + tags=[]
# data transformations
tf = []

if rc_transform:
    model_name += "_t=rc"

tr.transforms = transforms.Compose(tf)
tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=True)

# + tags=[]
# # create a smaller training set for development
# tr = load("train_sequences.txt", "train.pt", Dream, path=here("data/dream"))
# tr.cache_rc()
# tr, val = torch.utils.data.random_split(tr, [len(tr) - val_size, val_size])

# d = Dream(torch.Tensor(), torch.Tensor())
# d.sequences, d.rc_sequences, d.expression = next(
#     iter(DataLoader(tr, batch_size=100000 + val_size, shuffle=True))
# )
# d.expression = d.expression.flatten()

# torch.save(d, here("data/dream/train_dev.pt"))
# -

# # Model training

# + tags=[]
tr_losses = []
tr_pearson_list = []
tr_spearman_list = []
val_losses = []
val_pearson_list = []
val_spearman_list = []
tr_pearson = 0
tr_spearman = 0
best_performance = -float("inf")

# scaler for mixed precision training
scaler = torch.cuda.amp.GradScaler()

for epoch in range(n_epochs):
    with tqdm(tr_loader) as tepoch:
        for seq, rc, y in tepoch:

            # forward
            model.train()

            seq, rc, y = seq.to(device), rc.to(device), y.to(device)

            if rc_transform and random() < 0.5:
                seq, rc = rc, seq

            with autocast:  # mixed precision
                y_pred = model(seq, rc)
                tr_loss = criterion(y_pred, y)

            # backward (with mixed precision)
            optimizer.zero_grad()
            scaler.scale(tr_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # evaluation
            model.eval()

            tr_losses.append(tr_loss.item())
            tr_pearson = 0.9 * tr_pearson + 0.1 * pearsonr(y, y_pred)
            tr_spearman = 0.9 * tr_spearman + 0.1 * spearmanr(y, y_pred)

            tepoch.set_postfix(
                tr_loss=tr_loss.item(),
                r=tr_pearson,
                rho=tr_spearman,
            )

        with torch.no_grad():
            tr_pearson_list.append(tr_pearson)
            tr_spearman_list.append(tr_spearman)
            val_pred = model(val_seqs.to(device), val_rc.to(device)).cpu()
            val_loss = criterion(val_pred, val_expression)
            val_losses.append(val_loss.item())
            val_pearson = pearsonr(val_expression, val_pred)
            val_pearson_list.append(val_pearson)
            val_spearman = spearmanr(val_expression, val_pred)
            val_spearman_list.append(val_spearman)

        # store model iff on a cuda environment and if the repo is clean
        performance = val_pearson + val_spearman

        if device.type == "cuda" and sha and performance > best_performance:

            best_performance = performance
            torch.save(model.state_dict(), here(f"results/models/{model_name}.pt"))
            torch.save(
                {
                    "commit": sha,
                    "train_loss": tr_losses,
                    "train_pearson": tr_pearson,
                    "train_spearman": tr_spearman,
                    "val_loss": val_losses,
                    "val_pearson": val_pearson,
                    "val_spearman": val_spearman,
                },
                here(f"results/models/{model_name}_stats.pt"),
            )

# + tags=[]
if device.type == "cuda" and sha:
    # Best mean of pearson and spearman
    best_epoch = torch.argmax(
        torch.Tensor(val_pearson_list) + torch.Tensor(val_spearman_list)
    )

    tr_pearson = tr_pearson_list[best_epoch]
    tr_spearman = tr_spearman_list[best_epoch]
    val_pearson = val_pearson_list[best_epoch]
    val_spearman = val_spearman_list[best_epoch]

    with (open(here("results/models/summary.tsv"), "a")) as S:
        S.write(f"{model_name}\t{tr_pearson:.3f}\t{tr_spearman:.3f}\t")
        S.write(f"{val_pearson:.3f}\t{val_spearman:.3f}\t{sha}\n")

# + tags=[]
fig, ax = plt.subplots(1, 1, figsize=(15, 6))
sns.lineplot(x=list(range(len(tr_losses))), y=tr_losses)
sns.lineplot(
    x=[(i + 1) * len(tr_loader) for i in range(len(val_losses))],
    y=val_losses,
    color="orange",
)
ax.set(xlabel="Minibatch", ylabel="Loss")
# -

# # Model analysis

# + tags=[]
del model

model = model_obj().to(device)
model.load_state_dict(
    torch.load(here(f"results/models/{model_name}.pt"), map_location=device)
)
model.eval()

with torch.no_grad():
    # training predictions
    tr_seqs, tr_rc, tr_expression = next(iter(tr_loader))
    tr_pred = model(tr_seqs.to(device), tr_rc.to(device)).cpu()

    # validation predictions
    val_pred = model(val_seqs.to(device), val_rc.to(device)).cpu()


# + tags=[]
def plt_predictions(y, y_pred, split):

    y = numpify(y).flatten()
    y_pred = numpify(y_pred).flatten()
    axis = 0 if split == "Train" else 1

    g = sns.scatterplot(x=y, y=y_pred, ax=ax[axis])
    g.set(xlabel="y", ylabel="y_pred")
    g.set_title(split)
    ax[axis].axline([0, 0], [17, 17], color="red")


fig, ax = plt.subplots(1, 2)
plt_predictions(tr_expression, tr_pred, "Train")
plt_predictions(val_expression, val_pred, "Validation")
