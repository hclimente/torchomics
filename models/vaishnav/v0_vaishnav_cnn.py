import pytorch_lightning as pl
import torch
import torch.nn as nn


class VaishnavCNN(pl.LightningModule):
    def __init__(self, p_dropout=0.2, seq_length=80):
        super(VaishnavCNN, self).__init__()

        self.p_dropout = p_dropout

        self.strand_conv = nn.Sequential(
            nn.Conv1d(4, 256, 30, padding="same"),
            nn.ReLU(),
            nn.Conv1d(256, 256, 30, padding="same"),
            nn.ReLU(),
        )

        self.joint_conv = nn.Sequential(
            nn.Conv1d(512, 256, 30, padding="same"),
            nn.ReLU(),
            nn.Conv1d(256, 256, 30, padding="same"),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Dropout(self.p_dropout),
            nn.Linear(256 * seq_length, 256),
            nn.ReLU(),
            nn.Dropout(self.p_dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, seq, rc):

        x1 = self.strand_conv(seq)
        x2 = self.strand_conv(rc)

        x = torch.hstack((x1, x2))
        x = self.joint_conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x
