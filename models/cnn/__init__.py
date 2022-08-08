from .basenji import Basenji
from .densenet import DenseNet
from .resnet import ConvNeXt, ResNet18, ResNet50, ResNeXt18, ResNeXt50
from .v2_resnet_wannabe import Wannabe
from .v5_attn_resnet import (
    AttentionResNet18,
    AttentionResNet50,
    MultiHeadAttentionResNet18,
    MultiHeadAttentionResNet50,
)
from .v6_munext import MuNext
from .vgg import VGG, SimpleCNN

__all__ = [
    "SimpleCNN",
    "VGG",
    "Basenji",
    "Wannabe",
    "ResNet18",
    "ResNet50",
    "ResNeXt18",
    "ResNeXt50",
    "ConvNeXt",
    "DenseNet",
    "AttentionResNet18",
    "AttentionResNet50",
    "MultiHeadAttentionResNet18",
    "MultiHeadAttentionResNet50",
    "MuNext",
]
