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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from pyprojroot import here
from sklearn.decomposition import PCA

from data import Dream

# from data.utils import one_hot_encode, pad
from models.utils import numpify

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# -

# load data
train_df = pd.read_csv(
    f"{here('data/dream')}/train_sequences.txt", sep="\t", names=["seq", "expr"]
)
test_df = pd.read_csv(
    f"{here('data/dream')}/test_sequences.txt", sep="\t", names=["seq", "expr"]
)

# all sequences START with the same subsequence
primer5 = "TGCATTTTTTTCACATC"
print("Train:", sum([not x.startswith(primer5) for x in train_df.seq]))
print("Test:", sum([not x.startswith(primer5) for x in test_df.seq]))
train_df.seq = [x.removeprefix(primer5) for x in train_df.seq]
test_df.seq = [x.removeprefix(primer5) for x in test_df.seq]

# all sequences END with the same subsequence
primer3 = "GGTTACGGCTGTT"
print("Train:", sum([not x.endswith(primer3) for x in train_df.seq]))
print("Test:", sum([not x.endswith(primer3) for x in test_df.seq]))
train_df.seq = [x.removesuffix(primer3) for x in train_df.seq]
test_df.seq = [x.removesuffix(primer3) for x in test_df.seq]

# +
# length of the sequences
tr_seq_len = [len(x) for x in train_df.seq]
te_seq_len = [len(x) for x in test_df.seq]

g = sns.histplot(x=tr_seq_len)
g.set(yscale="log")
g.set(xlabel="Sequence length")

print(f"Train: max length={max(tr_seq_len)}; min length={min(tr_seq_len)}")
print(f"Test: max length={max(te_seq_len)}; min length={min(te_seq_len)}")

# +
# convert into one-hot
# train_one_hot = torch.stack(
#     [one_hot_encode(pad(x, max(tr_seq_len))) for x in train_df.seq]
# )
# test_one_hot = torch.stack(
#     [one_hot_encode(pad(x, max(te_seq_len))) for x in test_df.seq]
# )

# torch.save(train_one_hot, f"{here('data/dream')}/train_one_hot.pt")
# torch.save(test_one_hot, f"{here('data/dream')}/test_one_hot.pt")

train_one_hot = torch.load(f"{here('data/dream')}/train_one_hot.pt")
test_one_hot = torch.load(f"{here('data/dream')}/test_one_hot.pt")


# + tags=[]
# study nucleotide frequencies
def plot_nt_freq(seqs, **plt_kwargs):

    df = pd.DataFrame(
        (seqs.sum(axis=0) / seqs.sum(axis=(0, 1))).T, columns=["A", "C", "T", "G"]
    )
    df["Position"] = list(range(seqs.shape[2]))
    df = df.melt("Position", var_name="Base", value_name="Frequency")

    g = sns.lineplot(x="Position", y="Frequency", hue="Base", data=df, **plt_kwargs)
    g.axis(ymin=0.14, ymax=0.4, xmin=0, xmax=seqs.shape[2])
    g.set_title("Test" if seqs is test_one_hot else "Train")

    return g


fig, ax = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Nucleotide frequencies", fontsize=16)
plot_nt_freq(train_one_hot[:, :, :100], ax=ax[0])
plot_nt_freq(test_one_hot, ax=ax[1])


# + tags=[]
# study missing values
def plot_na(seqs, **plt_kwargs):
    missing = seqs.shape[0] - seqs.sum(axis=(0, 1))

    g = sns.lineplot(x=list(range(80)), y=missing + 0.99, **plt_kwargs)
    g.set(yscale="log")
    g.set_title("Test" if seqs is test_one_hot else "Train")
    g.axis(ymin=0.9, ymax=2e6)
    g.set(xlabel="Position", ylabel="Number of samples")

    return g


plt.rcParams["figure.figsize"] = 15, 6
fig, ax = plt.subplots(1, 2)
fig.suptitle("Number of missing values", fontsize=16)
plot_na(train_one_hot[:, :, :80], ax=ax[0])
plot_na(test_one_hot, ax=ax[1])

# + tags=[]
# number of missing values in the first 80 positions
g = sns.histplot(x=numpify(80 - train_one_hot[:, :, :80].sum(axis=(1, 2))))
g.set(yscale="log")
g.set(xlabel="# missing values")
# -

sns.scatterplot(x=np.array(tr_seq_len), y=train_df.expr)
g.set(xlabel="Sequence length", ylabel="Expression")

# + tags=[]
# study expression
g = sns.histplot(x=train_df.expr)
g.set(yscale="log")
g.set(xlabel="Expression")
g.axis(xmin=-0.1, xmax=17.1)

print("Min expression:", train_df.expr.min())
print("Mean expression:", train_df.expr.mean())
print("Median expression:", train_df.expr.median())
print("Max expression:", train_df.expr.max())

# + tags=[]
# check outliers: 0 expression
outliers = train_one_hot[train_df.expr == 0, :, :80]

df = pd.DataFrame(
    (outliers.sum(axis=0) / outliers.sum(axis=(0, 1))).T,
    columns=["A", "C", "T", "G"],
)

df["Position"] = list(range(80))
df = df.melt("Position", var_name="Base", value_name="Frequency")

g = sns.lineplot(x="Position", y="Frequency", hue="Base", data=df)
g.axis(xmin=0, xmax=79)
g.set_title(
    f"Base frequencies in {sum(train_df.expr == 0)} 0-expression sequences", fontsize=16
)

# + tags=[]
decimals, _ = np.modf(train_df.expr)
g = sns.histplot(x=decimals)
g.set(yscale="log")
g.set(xlabel="Decimals")
g.axis(xmin=-0.01, xmax=1.01)

# +
# mantissa distribution
u, counts = np.unique(decimals[decimals != 0], return_counts=True)
counts_sort_ind = np.argsort(-counts)

u[counts_sort_ind][:10]

# +
tr_categorical = (
    torch.argmax(train_one_hot, axis=1).float() - 1.5
)  # numerise and zero-center
te_categorical = (
    torch.argmax(test_one_hot, axis=1).float() - 1.5
)  # numerise and zero-center

idx = range(0, 80, 4)

vars_tr, vars_te = (
    PCA().fit(tr_categorical).explained_variance_ratio_.cumsum()[idx],
    PCA().fit(te_categorical).explained_variance_ratio_.cumsum()[idx],
)

g = sns.FacetGrid(
    pd.concat(
        map(
            pd.DataFrame,
            [
                {
                    "explained variance": vars_tr,
                    "no. components": idx,
                    "Data": len(idx) * ["Train"],
                },
                {
                    "explained variance": vars_te,
                    "no. components": idx,
                    "Data": len(idx) * ["Test"],
                },
            ],
        )
    ),
    col="Data",
    size=5,
)
g.map_dataframe(sns.barplot, x="no. components", y="explained variance")
# -

# # Prepare training set

# +
# remove sequences too long (>80) or too short (<78)
tr_seq_len = torch.Tensor(tr_seq_len)
ok_length = torch.logical_and(tr_seq_len > 77, tr_seq_len <= 80)

# remove sequences with more than 2 missing values in the first 80 positions
ok_missing = 80 - train_one_hot[:, :, :80].sum(axis=(1, 2)) < 3
# -

# total sequences removed
ok_sequences = torch.logical_and(ok_length, ok_missing)
torch.logical_not(ok_sequences).sum()

# +
train = Dream(
    train_one_hot[ok_sequences, :, 0:80], torch.Tensor(train_df.expr)[ok_sequences]
)
train.rc_sequences = None  # save space

torch.save(train, f"{here('data/dream')}/train.pt")

# +
test = Dream(test_one_hot, torch.Tensor(test_df.expr))
test.rc_sequences = None  # save space

torch.save(test, f"{here('data/dream')}/test.pt")
