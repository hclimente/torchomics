import pytorch_lightning as pl
import torch.nn as nn

from models.layers import RevCompConv1D


class SimpleCNN(pl.LightningModule):
    def __init__(self, p_dropout: float = 0.0, kernel_size: int = 5):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size, padding="same"),
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


class SimpleCNN_BN(pl.LightningModule):
    def __init__(self):
        super(SimpleCNN_BN, self).__init__()

        def conv_block(channels_in, channels_out, width=5):
            return nn.Sequential(
                nn.Conv1d(channels_in, channels_out, width, padding="same", bias=False),
                nn.BatchNorm1d(channels_out),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )

        self.conv = nn.Sequential(
            conv_block(4, 16),
            conv_block(16, 32),
        )

        self.fc = nn.Sequential(
            nn.Linear(640, 256),
            nn.ReLU(),
            nn.Linear(256, 96),
            nn.ReLU(),
            nn.Linear(96, 1),
        )

    def forward(self, x, rc=None):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


class SimpleCNN_GELU(pl.LightningModule):
    def __init__(self):
        super(SimpleCNN_GELU, self).__init__()

        def conv_block(channels_in, channels_out, width=5):
            return nn.Sequential(
                nn.Conv1d(channels_in, channels_out, width, padding="same"),
                nn.GELU(),
                nn.MaxPool1d(2),
            )

        self.conv = nn.Sequential(
            conv_block(4, 16),
            conv_block(16, 32),
        )

        self.fc = nn.Sequential(
            nn.Linear(640, 256),
            nn.ReLU(),
            nn.Linear(256, 96),
            nn.ReLU(),
            nn.Linear(96, 1),
        )

    def forward(self, x, rc=None):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


class DeepCNN(pl.LightningModule):
    def __init__(self):
        super(DeepCNN, self).__init__()

        def conv_block(channels_in, channels_out, width=15):
            return nn.Sequential(
                nn.Conv1d(channels_in, channels_out, width, padding="same"),
                nn.BatchNorm1d(channels_out),
                nn.GELU(),
                nn.MaxPool1d(2),
            )

        self.conv = nn.Sequential(
            conv_block(4, 250),
            conv_block(250, 360),
            conv_block(360, 432),
            conv_block(432, 520),
            conv_block(520, 624),
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


class SimpleCNN_RC(pl.LightningModule):
    def __init__(self):
        super(SimpleCNN_RC, self).__init__()
        self.conv = nn.Sequential(
            RevCompConv1D(4, 16, 5, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(640, 256),
            nn.ReLU(),
            nn.Linear(256, 96),
            nn.ReLU(),
            nn.Linear(96, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x
