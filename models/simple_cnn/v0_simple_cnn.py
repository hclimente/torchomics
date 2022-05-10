import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4, 16, 5, padding="same"),
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

    def forward(self, x, rc=None):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


class SimpleCNN_BN(nn.Module):
    def __init__(self):
        super(SimpleCNN_BN, self).__init__()

        def conv_block(channels_in, channels_out, width=5):
            return nn.Sequential(
                nn.Conv1d(channels_in, channels_out, width, padding="same"),
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


class SimpleCNN_GELU(nn.Module):
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
