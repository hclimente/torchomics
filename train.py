import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from data import Vaishnav
from models import OneStrandCNN

epochs = 5
batch_size = 1024
# train_size = 1024
device = "cuda"
tr_losses = []
te_losses = []


def get_loader(path, save_path):

    sequences = pd.read_csv(
        "data/vaishnav_et_al/" + path,
        # nrows=train_size,
        sep="\t",
        names=["seq", "expr"],
    )
    ds = Vaishnav(sequences.seq, sequences.expr)
    torch.save(ds, "data/vaishnav_et_al/" + save_path)

    return DataLoader(ds, batch_size=batch_size, shuffle=True)


# tr_loader = get_loader("complex_media_training_data_Glu.txt",
#                       "complex_train.torch")
# tr_loader = get_loader("Random_testdata_complex_media.txt",
#                       "complex_test.torch")

tr = torch.load("data/vaishnav_et_al/complex_train.torch")
tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
te = torch.load("data/vaishnav_et_al/complex_test.torch")
te_loader = DataLoader(te, batch_size=len(te), shuffle=True)

net = OneStrandCNN().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0005, weight_decay=1e-4)

with trange(epochs) as epochs:
    for epoch in epochs:
        test_data = iter(te_loader)
        with tqdm(tr_loader, total=int(len(tr_loader) / batch_size)) as tepoch:
            for seq, expr in tepoch:
                # train
                net.train()

                seq, expr = seq.to(device), expr.to(device)

                optimizer.zero_grad()
                out = net(seq)
                tr_loss = criterion(out, expr)
                tr_loss.backward()
                optimizer.step()

                tr_losses.append(tr_loss.item())
                tepoch.set_postfix(tr_loss=tr_loss.item())

            # test
            net.eval()

            seq_test, expr_test = next(test_data)
            seq_test, expr_test = seq_test.to(device), expr_test.to(device)
            te_pred = net(seq_test)
            te_loss = criterion(te_pred, expr_test)
            te_losses.append(te_loss.item())

            torch.save(net.state_dict(), "model.torch")
            torch.save((tr_losses, te_losses, te_pred), "losses.torch")
