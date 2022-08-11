import torch.nn as nn

from models.utils import conv_block


class Wannabe(nn.Module):
    def __init__(
        self, p_dropout: float = 0.1, kernel_size: int = 7, nb_repeats: int = 3
    ):
        super(Wannabe, self).__init__()

        self.conv = nn.Sequential(
            conv_block(4, 256, kernel_size, padding=0),
            ResidualWannabe(conv_block(256, 256, kernel_size, nb_repeats=nb_repeats)),
            ResidualWannabe(conv_block(256, 256, kernel_size, nb_repeats=nb_repeats)),
            ResidualWannabe(conv_block(256, 256, kernel_size, nb_repeats=nb_repeats)),
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 5, 128),
            nn.Dropout(p_dropout),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(p_dropout),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, rc=None):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


class ResidualWannabe(nn.Module):
    def __init__(self, module):
        super(ResidualWannabe, self).__init__()

        self.module = module

    def forward(self, x):
        return nn.functional.avg_pool1d(x, kernel_size=2, padding=1)[
            :, :, :-1
        ] + self.module(x)
