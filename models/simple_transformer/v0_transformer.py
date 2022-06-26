import math

import torch
from torch import nn


class Transformer(nn.Module):
    def __init__(self, n_layers=2, d_model=64, n_head=8, d_hidden=512, dropout=0.5):

        """
        n_layers    - number of transformer modules
        d_model     - dimension of token embeddings
        d_head      - number of self-attention heads in each layer
        d_hidden    - hidden dimension in self-attention heads (K/Q/V dimension)
        """

        super(Transformer, self).__init__()

        self.encoder = nn.Embedding(4, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model, n_head, d_hidden, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)

        self.d_model = d_model
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x, rc=None):

        x = self.encoder(torch.argmax(x, axis=1)) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x[:, 0])

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):

        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):

        x = x + self.pe[:, : x.shape[1]]
        return self.dropout(x)
