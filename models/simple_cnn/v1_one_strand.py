import torch.nn as nn


class OneStrandCNN(nn.Module):
    def __init__(self, p_dropout=0.2):
        super(OneStrandCNN, self).__init__()

        self.p_dropout = p_dropout

        self.conv = nn.Sequential(
            nn.Conv1d(4, 256, 29, padding="same"),
            nn.Conv1d(256, 256, 29, padding="same"),
            nn.Conv1d(256, 128, 29, padding="same"),
            nn.ReLU(),
            nn.Conv1d(128, 128, 29, padding="same"),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Dropout(self.p_dropout),
            nn.Linear(128 * 110, 256),
            nn.ReLU(),
            nn.Dropout(self.p_dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):

        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x

    def __name__(self):
        return "v1_one_strand"
