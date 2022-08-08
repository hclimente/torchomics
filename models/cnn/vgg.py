import pytorch_lightning as pl
import torch.nn as nn


class SimpleCNN(pl.LightningModule):
    def __init__(self, p_dropout: float = 0.0, kernel_size: int = 5):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(640, 256),
            nn.Dropout(p_dropout),
            nn.ReLU(),
            nn.Linear(256, 96),
            nn.Dropout(p_dropout),
            nn.ReLU(),
            nn.Linear(96, 1),
        )

    def forward(self, x, rc=None):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


class VGG(pl.LightningModule):
    def __init__(self, kernel_size: int = 15):
        super(VGG, self).__init__()

        def conv_block(channels_in, channels_out, width, padding="same"):
            return nn.Sequential(
                nn.Conv1d(channels_in, channels_out, width, padding=padding),
                nn.BatchNorm1d(channels_out),
                nn.GELU(),
                nn.MaxPool1d(2),
            )

        self.conv = nn.Sequential(
            conv_block(4, 250, kernel_size, "valid"),
            conv_block(250, 360, kernel_size),
            conv_block(360, 432, kernel_size),
            conv_block(432, 520, kernel_size),
            conv_block(520, 624, kernel_size),
        )

        self.fc = nn.Sequential(
            nn.Linear(1248, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x, rc=None):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x
