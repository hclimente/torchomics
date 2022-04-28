import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    def __init__(self, p_dropout=0.2):
        super(BaselineCNN, self).__init__()

        self.p_dropout = p_dropout

        self.strand_conv = nn.Sequential(
            nn.Conv1d(4, 256, 30, padding="same"),
            nn.Conv1d(256, 256, 30, padding="same"),
        )

        self.joint_conv = nn.Sequential(
            nn.Conv1d(512, 256, 30, padding="same"),
            nn.ReLU(),
            nn.Conv1d(256, 256, 30, padding="same"),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Dropout(self.p_dropout),
            nn.Linear(256 * 110, 256),
            nn.ReLU(),
            nn.Dropout(self.p_dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, plus, minus):

        x1 = self.strand_conv(plus)
        x2 = self.strand_conv(minus)

        x = torch.hstack((x1, x2))
        x = self.joint_conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x

    def __name__(self):
        return "v0_baseline"
