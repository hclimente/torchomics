import torch.nn as nn


class SimpleCNN(nn.Module):
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


class VGG(nn.Module):
    def __init__(self, kernel_size: int = 3, layers: list = [2, 2, 3, 3, 3]):
        super(VGG, self).__init__()

        def conv_block(channels_in, channels_out, width, nb_repeats):
            block = []

            padding = "valid" if channels_in == 4 else "same"

            for _ in range(nb_repeats):
                block.append(
                    nn.Conv1d(channels_in, channels_out, width, padding=padding)
                )
                block.append(nn.ReLU())
                channels_in = channels_out

            block.append(nn.MaxPool1d(2, stride=2))

            return block

        blocks = []
        channels_in = 4
        channels_out = 64

        for n in layers:
            blocks.extend(conv_block(channels_in, channels_out, kernel_size, n))
            channels_in = channels_out
            channels_out = min(512, 2 * channels_out)

        self.conv = nn.Sequential(*blocks)
        self.fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, rc=None):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x
