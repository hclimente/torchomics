from .v0_simple_cnn import (
    DeepCNN,
    SimpleCNN,
    SimpleCNN_BN,
    SimpleCNN_GELU,
    SimpleCNN_RC,
)
from .v1_resnet import ConvNeXt26, ConvNeXt50, ResNet18, ResNet50, ResNeXt18, ResNeXt50
from .v2_resnet_wannabe import Wannabe
from .v4_densenet import DenseNet
from .v5_attn_resnet import (
    AttentionResNet18,
    AttentionResNet50,
    MultiHeadAttentionResNet18,
    MultiHeadAttentionResNet50,
)

__all__ = [
    "SimpleCNN",
    "SimpleCNN_BN",
    "SimpleCNN_GELU",
    "DeepCNN",
    "SimpleCNN_RC",
    "Wannabe",
    "ResNet18",
    "ResNet50",
    "ResNeXt18",
    "ResNeXt50",
    "ConvNeXt26",
    "ConvNeXt50",
    "DenseNet",
    "AttentionResNet18",
    "AttentionResNet50",
    "MultiHeadAttentionResNet18",
    "MultiHeadAttentionResNet50",
]
