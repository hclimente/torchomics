import math

import matplotlib.pyplot as plt
import torch
from torch import nn


class Transformer(nn.Module):

    REG_TOKEN: int = 4

    def __init__(self, n_layers=2, d_model=64, n_head=8, d_hidden=512, dropout=0.1):

        """
        n_layers    - number of transformer modules
        d_model     - dimension of token embeddings
        d_head      - number of self-attention heads in each layer
        d_hidden    - hidden dimension in self-attention heads (K/Q/V dimension)
        """

        super(Transformer, self).__init__()

        self.encoder = nn.Embedding(self.REG_TOKEN + 1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model, n_head, d_hidden, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)

        self.d_model = d_model
        self.decoder = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
        )

        self.register_buffer("reg_token", torch.Tensor([[self.REG_TOKEN]]).int())

    def forward(self, x, rc=None):

        # prepend regression token
        batch_reg_token = self.reg_token.repeat(x.shape[0], 1)
        x = torch.cat([batch_reg_token, torch.argmax(x, axis=1)], axis=1)

        x = self.encoder(x)
        x = self.pos_encoder(x)

        x = self.transformer_encoder(x)
        x = self.decoder(x[:, 0])  # decode regression token

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=81):

        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(81.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):

        x = x + self.pe[:, : x.shape[1]]
        return self.dropout(x)

    def visualise(self):
        fig, ax = plt.subplots()
        ax.imshow(self.pe[0])
        plt.show()
