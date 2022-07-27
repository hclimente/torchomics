from .v0_simple_cnn import DeepCNN, SimpleCNN
from .v1_resnet import ConvNeXt, ResNet18, ResNet50, ResNeXt18, ResNeXt50
from .v2_resnet_wannabe import Wannabe
from .v4_densenet import DenseNet
from .v5_attn_resnet import (
    AttentionResNet18,
    AttentionResNet50,
    MultiHeadAttentionResNet18,
    MultiHeadAttentionResNet50,
)
from .v6_munext import MuNext

__all__ = [
    "SimpleCNN",
    "DeepCNN",
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
