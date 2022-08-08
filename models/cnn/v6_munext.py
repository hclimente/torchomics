import torch.nn as nn

from models.cnn.resnet import ResNet
from models.cnn.v5_attn_resnet import MultiHeadAttention as MHA
from models.layers import RevCompConv1D


class MuNext(ResNet):
    def __init__(self, layers: list = [3, 9, 9, 3], base_width: int = 96):
        super(MuNext, self).__init__(layers, MuBottleneck, base_width=base_width)

        self.input = nn.Sequential(
            RevCompConv1D(4, base_width, 15, padding="same", bias=False),
            nn.BatchNorm1d(base_width),
            nn.GELU(),
        )


class MuBottleneck(nn.Module):

    expansion: int = 1

    def __init__(
        self,
        channels_in,
        width,
        kernel_size,
        stride,
        groups=None,
        downsample=None,
    ):
        super(MuBottleneck, self).__init__()

        self.downsample = downsample

        next_width = int(self.expansion * width)

        self.bottleneck = nn.Sequential(
            MHA(channels_in),
            nn.GroupNorm(num_groups=1, num_channels=channels_in),
            nn.Conv1d(
                channels_in,
                4 * next_width,
                kernel_size=1,
                bias=False,
                stride=stride,
            ),
            nn.GELU(),
            nn.Conv1d(4 * next_width, next_width, 1, bias=False),
        )

    def forward(self, x):

        identity = x if self.downsample is None else self.downsample(x)
        x = identity + self.bottleneck(x)

        return x
