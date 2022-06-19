from .v0_simple_cnn import (
    DeepCNN,
    SimpleCNN,
    SimpleCNN_BN,
    SimpleCNN_GELU,
    SimpleCNN_RC,
)
from .v1_resnet import ConvNeXt50, ResNet18, ResNet50, ResNeXt18, ResNeXt50
from .v2_resnet_wannabe import Wannabe

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
    "ConvNeXt50",
]
