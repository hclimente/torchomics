import pytorch_lightning as pl
import torch.nn as nn


class RNN(pl.LightningModule):
    def __init__(self):
        super(RNN, self).__init__()

        def conv_block(channels_in, channels_out, width):
            return nn.Sequential(
                nn.Conv1d(channels_in, channels_out, width, padding="same"),
                nn.BatchNorm1d(channels_out),
                nn.GELU(),
            )

        self.convs = nn.Sequential(
            conv_block(4, 256, 15),
        )

        self.gru = nn.GRU(input_size=256, hidden_size=256, num_layers=1)

        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.Dropout(0.05),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, rc=None):

        nb_batch, _, seq_length = x.shape

        x = self.convs(x).view(seq_length, nb_batch, -1)
        x, _ = self.gru(x)
        x = self.fc(x[-1])

        return x
