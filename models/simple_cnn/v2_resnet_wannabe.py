import pytorch_lightning as pl
import torch.nn as nn

from models.utils import conv_block


class Wannabe(pl.LightningModule):
    def __init__(self):
        super(Wannabe, self).__init__()
        kernel_size = 7

        self.conv = nn.Sequential(
            conv_block(4, 256, kernel_size),
            ResidualWannabe(conv_block(256, 256, kernel_size)),
            ResidualWannabe(conv_block(256, 256, kernel_size)),
            ResidualWannabe(conv_block(256, 256, kernel_size)),
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 5, 128),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, rc=None):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


class ResidualWannabe(pl.LightningModule):
    def __init__(self, module):
        super(ResidualWannabe, self).__init__()

        self.module = module

    def forward(self, x):
        return nn.functional.avg_pool1d(x, kernel_size=2, padding=1)[
            :, :, :-1
        ] + self.module(x)
