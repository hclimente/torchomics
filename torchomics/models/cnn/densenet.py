import torch
import torch.nn as nn
import torch.nn.functional as F

from torchomics.layers import RevCompConv1D


class DenseNet(nn.Module):
    def __init__(
        self,
        layers: list = [6, 12, 24, 16],
        growth_rate: int = 32,
        base_width: int = 16,
        kernel_size: int = 3,
        p_dropout: float = 0.0,
        bottleneck: bool = False,
        reduction: float = 0.9,
    ):
        super(DenseNet, self).__init__()

        def dense_block(channels_in, growth_rate, nb_repeats, bottleneck):
            layers = []
            layer = DenseLayer if not bottleneck else BottleneckLayer

            for _ in range(nb_repeats):
                layers.append(layer(channels_in, growth_rate))
                channels_in += growth_rate

            return nn.Sequential(*layers)

        self.input = nn.Sequential(
            RevCompConv1D(4, base_width, kernel_size),
            nn.MaxPool1d(2),
        )

        conv_layers = []
        channels_in = base_width
        for n in layers:
            conv_layers.append(dense_block(channels_in, growth_rate, n, bottleneck))
            channels_in += n * growth_rate
            channels_out = int(channels_in * 0.9)
            conv_layers.append(Transition(channels_in, channels_out))
            channels_in = channels_out

        self.conv = nn.Sequential(*conv_layers)

        self.fc = nn.Sequential(
            nn.Linear(2 * channels_out, 256),
            nn.Dropout(p_dropout),
            nn.ReLU(),
            nn.Linear(256, 96),
            nn.Dropout(p_dropout),
            nn.ReLU(),
            nn.Linear(96, 1),
        )

    def forward(self, x, rc=None):
        x = self.input(x)
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


class DenseLayer(nn.Module):
    def __init__(self, channels_in, growth_rate):
        super(DenseLayer, self).__init__()

        self.bn = nn.BatchNorm1d(channels_in)
        self.conv = nn.Conv1d(
            channels_in, growth_rate, kernel_size=3, bias=False, padding="same"
        )

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = torch.cat((x, out), 1)

        return out


class BottleneckLayer(nn.Module):
    def __init__(self, channels_in, growth_rate):
        super(BottleneckLayer, self).__init__()
        channels_bottleneck = 4 * growth_rate
        self.bn1 = nn.BatchNorm1d(channels_in)
        self.conv1 = nn.Conv1d(
            channels_in, channels_bottleneck, kernel_size=1, bias=False, padding="same"
        )
        self.bn2 = nn.BatchNorm1d(channels_bottleneck)
        self.conv2 = nn.Conv1d(
            channels_bottleneck, growth_rate, kernel_size=3, bias=False, padding="same"
        )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(Transition, self).__init__()

        self.bn = nn.BatchNorm1d(channels_in)
        self.conv = nn.Conv1d(
            channels_in,
            channels_out,
            kernel_size=1,
            bias=False,
            padding="same",
        )

    def forward(self, x):
        x = self.conv(F.relu(self.bn(x)))
        x = F.avg_pool1d(x, 2)
        return x
