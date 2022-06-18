import pytorch_lightning as pl
import torch.nn as nn

from models.utils import conv_block


class ResNetWannabe(pl.LightningModule):
    def __init__(self, nb_outputs=1):
        super(ResNetWannabe, self).__init__()

        self.conv = nn.Sequential(
            conv_block(4, 256, 7),
            ResidualWannabe(conv_block(256, 256, 7)),
            ResidualWannabe(conv_block(256, 256, 7)),
            ResidualWannabe(conv_block(256, 256, 7)),
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, nb_outputs),
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
