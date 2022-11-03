import torch.nn as nn

from torchomics.layers import RevCompConv1D


class SimpleCNN(nn.Module):
    def __init__(self, p_dropout: float = 0.0, kernel_size: int = 5):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size, padding="same"),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(20),
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

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


class VGG(nn.Module):
    def __init__(self, kernel_size: int = 3, layers: list = [2, 2, 3, 3, 3]):
        super(VGG, self).__init__()

        def conv_block(channels_in, channels_out, width, nb_repeats):
            block = []

            for _ in range(nb_repeats):
                block.append(
                    nn.Conv1d(channels_in, channels_out, width, padding="same")
                )
                block.append(nn.ReLU())
                channels_in = channels_out

            block.append(nn.MaxPool1d(2, stride=2))

            return block

        self.input = nn.Sequential(
            RevCompConv1D(4, 64, kernel_size, padding="valid"),
            nn.ReLU(),
        )

        layers[0] -= 1

        blocks = []
        channels_in = 64
        channels_out = 64

        for n in layers:
            blocks.extend(conv_block(channels_in, channels_out, kernel_size, n))
            channels_in = channels_out
            channels_out = min(512, 2 * channels_out)

        self.conv = nn.Sequential(*blocks)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels_out, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.input(x)
        x = self.conv(x)
        x = self.avg_pool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x
