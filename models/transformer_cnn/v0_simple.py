import pytorch_lightning as pl
import torch
import torch.nn as nn


class RNN(pl.LightningModule):
    def __init__(self):

        super(RNN, self).__init__()

        self.embedding = nn.Embedding(4, 16)
        self.lstm = nn.LSTM(16, 256, batch_first=True)
        self.fc = nn.Linear(256, 1)

    def forward(self, x, rc=None):

        x = self.embedding(torch.argmax(x, axis=1))  # tokenise
        hts, (final_ht, final_ct) = self.lstm(x)
        x = self.fc(final_ht[-1])

        return x
