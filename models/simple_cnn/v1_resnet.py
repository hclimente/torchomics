import pytorch_lightning as pl
import torch.nn as nn

from models.layers import RevCompConv1D


class Bottleneck(nn.Module):
    def __init__(
        self,
        channels_in,
        channels_bn,
        channels_out,
        stride,
        downsample=None,
    ):
        super(Bottleneck, self).__init__()

        self.downsample = downsample

        self.bottleneck = nn.Sequential(
            nn.Conv1d(channels_in, channels_bn, 1, bias=False),
            nn.BatchNorm1d(channels_bn),
            nn.ReLU(),
            nn.Conv1d(
                channels_bn, channels_bn, 3, padding=1, bias=False, stride=stride
            ),
            nn.BatchNorm1d(channels_bn),
            nn.ReLU(),
            nn.Conv1d(channels_bn, channels_out, 1, bias=False),
            nn.BatchNorm1d(channels_out),
        )

    def forward(self, x):

        identity = x if self.downsample is None else self.downsample(x)
        x = identity + self.bottleneck(x)
        x = nn.functional.relu(x)

        return x


class ResNet(pl.LightningModule):
    def __init__(self, layers):
        super(ResNet, self).__init__()

        self.channels_in = 64

        self.input = nn.Sequential(
            RevCompConv1D(4, self.channels_in, 15, padding="same", bias=False),
            nn.BatchNorm1d(self.channels_in),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.layer1 = self._make_layer(layers[0], 64, 64, stride=1)
        self.layer2 = self._make_layer(layers[1], 128, 256, stride=2)
        self.layer3 = self._make_layer(layers[2], 256, 1024, stride=2)
        self.layer4 = self._make_layer(layers[3], 512, 2048, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool1d(2048)

        self.fc = nn.Linear(2048, 1)

    def forward(self, x, rc=None):
        x = self.input(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _make_layer(self, nb_repeats, channels_bn, channels_out, stride=1):

        downsample = None
        layers = []

        if stride != 1 or self.channels_in != channels_out:
            downsample = nn.Sequential(
                nn.Conv1d(self.channels_in, channels_out, 1, stride=stride, bias=False),
                nn.BatchNorm1d(channels_out),
            )

        for _ in range(nb_repeats):
            layers.append(
                Bottleneck(
                    self.channels_in, channels_bn, channels_out, stride, downsample
                )
            )

        self.channels_in = channels_out

        return nn.Sequential(*layers)


def ResNet50():
    return ResNet([3, 4, 6, 3])
