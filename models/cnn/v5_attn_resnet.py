import math

import torch
import torch.nn as nn

from models.cnn.resnet import BasicBlock, Bottleneck, ResNet
from models.layers import RevCompConv1D


class SelfAttention(nn.Module):

    """Self attention Layer"""

    def __init__(
        self,
        in_dim,
    ):

        super(SelfAttention, self).__init__()

        self.proj_dim = in_dim // 8

        self.query_conv = nn.Conv1d(
            in_channels=in_dim, out_channels=self.proj_dim, kernel_size=1
        )
        self.key_conv = nn.Conv1d(
            in_channels=in_dim, out_channels=self.proj_dim, kernel_size=1
        )
        self.value_conv = nn.Conv1d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        proj_query = self.query_conv(x).permute(0, 2, 1)  # transpose
        proj_key = self.key_conv(x)

        energy = torch.bmm(proj_query, proj_key) / math.sqrt(self.proj_dim)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = self.gamma * out + x

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):

        super(MultiHeadAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):

        x = x.permute((0, 2, 1))
        x = self.attn(x, x, x, need_weights=False)[0]
        x = x.permute((0, 2, 1))

        return x


class AttentionResNet(ResNet):
    def __init__(self, layers, block, attention_block, base_width=64, groups=1):
        super(AttentionResNet, self).__init__(layers, block, base_width=64, groups=1)

        self.input = nn.Sequential(
            RevCompConv1D(4, base_width, 15, padding="same", bias=False),
            nn.BatchNorm1d(base_width),
            nn.ReLU(),
        )

        self.maxpool = nn.MaxPool1d(2)
        self.attn1 = attention_block(base_width)
        self.attn2 = attention_block(base_width * block.expansion)
        self.attn3 = attention_block(2 * base_width * block.expansion)
        self.attn4 = attention_block(4 * base_width * block.expansion)

    def forward(self, x, rc=None):

        x = self.input(x)
        x = self.maxpool(x)

        x = self.attn1(x)
        x = self.layer1(x)

        x = self.attn2(x)
        x = self.layer2(x)

        x = self.attn3(x)
        x = self.layer3(x)

        x = self.attn4(x)
        x = self.layer4(x)

        x = self.avg_pool(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


class AttentionResNet18(AttentionResNet):
    def __init__(self):
        super().__init__([2, 2, 2, 2], BasicBlock, SelfAttention)


class AttentionResNet50(AttentionResNet):
    def __init__(self):
        super().__init__([3, 4, 6, 3], Bottleneck, SelfAttention)


class MultiHeadAttentionResNet18(AttentionResNet):
    def __init__(self):
        super().__init__([2, 2, 2, 2], BasicBlock, MultiHeadAttention)


class MultiHeadAttentionResNet50(AttentionResNet):
    def __init__(self):
        super().__init__([3, 4, 6, 3], Bottleneck, MultiHeadAttention)
