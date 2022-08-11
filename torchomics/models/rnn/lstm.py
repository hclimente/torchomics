import torch
import torch.nn as nn


class SimpleLSTM(nn.Module):
    def __init__(self, embedding_size: int = 16, width: int = 256):

        super(SimpleLSTM, self).__init__()

        self.embedding = nn.Embedding(4, embedding_size)
        self.lstm = nn.LSTM(embedding_size, width, batch_first=True)
        self.fc = nn.Linear(width, 1)

    def forward(self, x, rc=None):

        x = self.embedding(torch.argmax(x, axis=1))  # tokenise
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1])

        return x


class DeepLSTM(nn.Module):
    def __init__(self, embedding_size: int = 16, width: int = 256):

        super(DeepLSTM, self).__init__()

        self.embedding = nn.Embedding(4, embedding_size)

        self.lstm1 = nn.LSTM(
            embedding_size, width // 2, batch_first=True, bidirectional=True
        )
        self.lstm2 = nn.LSTM(width, width, batch_first=True)
        self.lstm3 = nn.LSTM(width, width, batch_first=True)

        self.fc = nn.Sequential(nn.Linear(width, 1))

    def forward(self, x, rc=None):

        x = self.embedding(torch.argmax(x, axis=1))  # tokenise

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)

        x = self.fc(x[:, -1])

        return x
