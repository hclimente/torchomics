import pytorch_lightning as pl
import torch.nn as nn


class StackedNet(pl.LightningModule):
    def __init__(self):
        super(StackedNet, self).__init__()

        def conv_block(channels_in, channels_out):

            return nn.Sequential(nn.Conv1d(channels_in, channels_out, dilation=1))
