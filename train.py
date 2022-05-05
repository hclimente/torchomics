import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import trange, tqdm

from data import load_vaishnav, ReverseComplement
from models import OneStrandCNN
from models.utils import fix_seeds

epochs = 5
batch_size = 1024
# train_size = 1024
device = "cuda"
tr_losses = []
te_losses = []
seed = 0

fix_seeds(seed)

tr = load_vaishnav("defined_media_training_data_SC_Ura.txt", "defined_train.pt")
tr.transform = transforms.Compose(ReverseComplement())
tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)

te = load_vaishnav("Random_testdata_defined_media.txt", "defined_test.pt")
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

            rc = ReverseComplement(1)

            seq_test, expr_test = next(test_data)
            seq_test, expr_test = seq_test.to(device), expr_test.to(device)
            # average prediction for sequence and its rc
            te_pred = (net(seq_test) + net(rc(seq_test))) / 2
            te_loss = criterion(te_pred, expr_test)
            te_losses.append(te_loss.item())

            net_name = net.__class__.__name__
            torch.save(net.state_dict(), f"results/models/{net_name}.torch")
            torch.save(
                (tr_losses, te_losses, te_pred), f"results/losses/{net_name}.torch"
            )
