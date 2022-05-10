import torch.nn as nn

from models.utils import Residual


class Basenji(nn.Module):
    def __init__(self):
        super(Basenji, self).__init__()

        def conv_block(channels_in, channels_out, width):
            return nn.Sequential(
                nn.Conv1d(channels_in, channels_out, width, padding="same"),
                nn.BatchNorm1d(channels_out),
                nn.GELU(),
                nn.MaxPool1d(2),
            )

        self.convs = nn.Sequential(
            conv_block(4, 288, 15),
            conv_block(288, 339, 5),
            conv_block(339, 400, 5),
            # conv_block(399, 469, 5),
            # conv_block(469, 550, 5),
            # conv_block(552, 650, 5),
            # conv_block(650, 768, 5),
        )

        def dilated_conv_block(dilation):

            return nn.Sequential(
                nn.GELU(),
                nn.Conv1d(400, 200, 3, dilation=dilation, padding="same"),
                nn.BatchNorm1d(200),
                nn.GELU(),
                nn.Conv1d(200, 400, 1),
                nn.Dropout(0.3),
            )

        self.dilated_convs = nn.Sequential(
            Residual(dilated_conv_block(int(3 * (1.5**0)))),
            Residual(dilated_conv_block(int(3 * (1.5**1)))),
            Residual(dilated_conv_block(int(3 * (1.5**2)))),
            Residual(dilated_conv_block(int(3 * (1.5**3)))),
            Residual(dilated_conv_block(int(3 * (1.5**4)))),
            Residual(dilated_conv_block(int(3 * (1.5**5)))),
            Residual(dilated_conv_block(int(3 * (1.5**6)))),
            Residual(dilated_conv_block(int(3 * (1.5**7)))),
            Residual(dilated_conv_block(int(3 * (1.5**8)))),
            Residual(dilated_conv_block(int(3 * (1.5**9)))),
            Residual(dilated_conv_block(int(3 * (1.5**10)))),
        )

        self.fc = nn.Sequential(
            nn.Linear(4000, 1536),
            nn.Dropout(0.05),
            nn.ReLU(),
            nn.Linear(1536, 1),
        )

    def forward(self, x, rc=None):
        x = self.convs(x)
        x = self.dilated_convs(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x
