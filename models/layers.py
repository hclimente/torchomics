import torch
import torch.nn as nn


class RevCompConv1D(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, *args, **kwargs):

        # each string will produce half of the output channels
        if (out_channels % 2) != 0:
            raise ValueError("out_channels must be an even number")

        out_channels = int(out_channels / 2)

        super(RevCompConv1D, self).__init__(
            in_channels, out_channels, kernel_size, *args, **kwargs
        )

    def forward(self, x):
        pos = super().forward(x)
        neg = super().forward(x.flip(1, 2))
        neg = neg.flip(2)

        return torch.hstack((pos, neg))
