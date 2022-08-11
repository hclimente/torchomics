import torch.nn as nn

from torchomics.layers import RevCompConv1D


class BasicBlock(nn.Module):

    expansion: int = 1

    def __init__(
        self,
        channels_in,
        width,
        kernel_size,
        stride,
        groups,
        downsample=None,
    ):
        super(BasicBlock, self).__init__()

        self.downsample = downsample

        self.bottleneck = nn.Sequential(
            nn.Conv1d(
                channels_in,
                width,
                kernel_size,
                padding=1,
                bias=False,
                stride=stride,
                groups=groups,
            ),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Conv1d(width, width, kernel_size, padding=1, bias=False, groups=groups),
            nn.BatchNorm1d(width),
        )

    def forward(self, x):

        identity = x if self.downsample is None else self.downsample(x)
        x = identity + self.bottleneck(x)
        x = nn.functional.relu(x)

        return x


class Bottleneck(nn.Module):

    expansion: int = 4

    def __init__(
        self,
        channels_in,
        width,
        kernel_size,
        stride,
        groups,
        downsample=None,
    ):
        super(Bottleneck, self).__init__()

        self.downsample = downsample

        self.bottleneck = nn.Sequential(
            nn.Conv1d(channels_in, width, 1, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Conv1d(
                width,
                width,
                kernel_size,
                padding=1,
                bias=False,
                stride=stride,
                groups=groups,
            ),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Conv1d(width, width * self.expansion, 1, bias=False),
            nn.BatchNorm1d(width * self.expansion),
        )

    def forward(self, x):

        identity = x if self.downsample is None else self.downsample(x)
        x = identity + self.bottleneck(x)
        x = nn.functional.relu(x)

        return x


class ConvNeXtBottleneck(nn.Module):

    """
    Note that GroupNorm with num_groups=1 is equivalent to LayerNorm:
    https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
    (see under "Examples")


    Note also that groups is a dummy parameter, as groups=channels_in
    (i.e. depth-wise convolution)
    """

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
        super(ConvNeXtBottleneck, self).__init__()

        self.downsample = downsample

        next_width = int(self.expansion * width)

        self.bottleneck = nn.Sequential(
            nn.Conv1d(
                channels_in,
                next_width,
                kernel_size=kernel_size,
                padding="same",
                bias=False,
                groups=channels_in,
            ),
            nn.GroupNorm(num_groups=1, num_channels=next_width),
            nn.Conv1d(
                next_width,
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


class GenericResNet(nn.Module):
    def __init__(
        self, layers, block, kernel_size=7, base_width=64, groups=1, p_dropout=0.0
    ):
        super(GenericResNet, self).__init__()

        self.channels_in = base_width

        if type(kernel_size) is not list:
            kernel_size = [kernel_size] * (len(layers) + 1)
        elif len(kernel_size) == 1:
            kernel_size = kernel_size * (len(layers) + 1)

        self.input = nn.Sequential(
            RevCompConv1D(4, self.channels_in, kernel_size[0]),
            nn.BatchNorm1d(self.channels_in),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.layer1 = self._make_layer(
            block, layers[0], base_width, kernel_size[1], stride=1, groups=groups
        )
        self.layer2 = self._make_layer(
            block, layers[1], 2 * base_width, kernel_size[2], stride=2, groups=groups
        )
        self.layer3 = self._make_layer(
            block, layers[2], 4 * base_width, kernel_size[3], stride=2, groups=groups
        )
        self.layer4 = self._make_layer(
            block, layers[3], 8 * base_width, kernel_size[4], stride=2, groups=groups
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = self.fc = nn.Sequential(
            nn.Linear(8 * base_width * block.expansion, 256),
            nn.Dropout(p_dropout),
            nn.ReLU(),
            nn.Linear(256, 96),
            nn.Dropout(p_dropout),
            nn.ReLU(),
            nn.Linear(96, 1),
        )

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

    def _make_layer(self, block, nb_repeats, width, kernel_size, stride=1, groups=1):

        downsample = None
        layers = []

        expanded_width = width * block.expansion

        if stride != 1 or self.channels_in != expanded_width:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.channels_in, expanded_width, 1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(expanded_width),
            )

        for _ in range(nb_repeats):
            layers.append(
                block(
                    self.channels_in,
                    width,
                    kernel_size,
                    stride,
                    groups,
                    downsample,
                )
            )
            downsample = None  # downsampling only occurs in the first block
            stride = 1
            self.channels_in = expanded_width

        return nn.Sequential(*layers)


class ResNet(GenericResNet):
    # ResNet18: layers = [2, 2, 2, 2]
    # ResNet50: layers = [3, 4, 6, 3]
    def __init__(
        self,
        layers: list = [3, 4, 6, 3],
        base_width: int = 64,
        kernel_size: list = [3],
        p_dropout: float = 0.0,
    ):
        super().__init__(
            layers,
            BasicBlock,
            kernel_size=kernel_size,
            base_width=base_width,
        )


class ResNeXt(GenericResNet):
    # ResNeXt18: layers = [2, 2, 2, 2]
    # ResNeXt50: layers = [3, 4, 6, 3]
    def __init__(
        self,
        layers: list = [3, 4, 6, 3],
        base_width: int = 64,
        groups: int = 32,
        kernel_size: list = [3],
        p_dropout: float = 0.0,
    ):
        super().__init__(
            layers,
            BasicBlock,
            kernel_size=kernel_size,
            base_width=base_width,
            groups=groups,
        )


class ConvNeXt(GenericResNet):
    def __init__(
        self,
        layers: list = [3, 3, 9, 3],
        base_width: int = 96,
        kernel_size: list = [7],
        p_dropout: float = 0.0,
    ):
        super().__init__(
            layers,
            ConvNeXtBottleneck,
            kernel_size=kernel_size,
            base_width=base_width,
        )
