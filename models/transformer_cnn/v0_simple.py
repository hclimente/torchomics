import torch.nn as nn


class TransformerCNN(nn.Module):
    def __init__(self):
        super(TransformerCNN, self).__init__()

        def conv_block(channels_in, channels_out, width):
            return nn.Sequential(
                nn.Conv1d(channels_in, channels_out, width, padding="same"),
                nn.BatchNorm1d(channels_out),
                nn.GELU(),
            )

        self.convs = nn.Sequential(
            conv_block(4, 256, 15),
        )

        self.gru = nn.GRU(input_size=80, hidden_size=80, num_layers=1)

        self.fc = nn.Sequential(
            nn.Linear(20480, 1024),
            nn.Dropout(0.05),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )

    def forward(self, x, rc=None):
        x = self.convs(x)
        _, x = self.gru(x, x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x
