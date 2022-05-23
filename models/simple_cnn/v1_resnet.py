import torch.nn as nn

from models.layers import RevCompConv1D


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        def conv_block(channels_in, channels_out, width=16, conv=nn.Conv1d):
            return nn.Sequential(
                conv(channels_in, channels_out, width, padding="same"),
                nn.BatchNorm1d(channels_out),
                nn.GELU(),
                nn.MaxPool1d(2),
            )

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
            nn.Linear(64, 1),
        )

    def forward(self, x, rc=None):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


class Residual(nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()

        self.module = module

    def forward(self, x):
        return nn.functional.avg_pool1d(x, kernel_size=2, padding=1)[
            :, :, :-1
        ] + self.module(x)