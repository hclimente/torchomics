import pytorch_lightning as pl
import torch.nn as nn

from models.layers import RevCompConv1D
from models.utils import conv_block


class ResNet(pl.LightningModule):
    def __init__(self, nb_outputs=1):
        super(ResNet, self).__init__()

        self.conv = nn.Sequential(
            conv_block(4, 256, conv=RevCompConv1D),
            Residual(conv_block(256, 256, 8)),
            Residual(conv_block(256, 256, 8)),
            Residual(conv_block(256, 256, 8)),
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


class MyopicResNet(pl.LightningModule):
    def __init__(self):
        super(MyopicResNet, self).__init__()

        self.conv = nn.Sequential(
            conv_block(4, 256, conv=RevCompConv1D),
            Residual(conv_block(256, 256)),
            Residual(conv_block(256, 256)),
            Residual(conv_block(256, 256)),
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, rc=None):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


class Residual(pl.LightningModule):
    def __init__(self, module):
        super(Residual, self).__init__()

        self.module = module

    def forward(self, x):
        return nn.functional.avg_pool1d(x, kernel_size=2, padding=1)[
            :, :, :-1
        ] + self.module(x)
