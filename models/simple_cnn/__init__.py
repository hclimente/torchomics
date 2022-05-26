from .v0_simple_cnn import (
    DeepCNN,
    SimpleCNN,
    SimpleCNN_BN,
    SimpleCNN_GELU,
    SimpleCNN_RC,
)
from .v1_resnet import MyopicResNet, ResNet

__all__ = [
    "SimpleCNN",
    "SimpleCNN_BN",
    "SimpleCNN_GELU",
    "DeepCNN",
    "SimpleCNN_RC",
    "ResNet",
    "MyopicResNet",
]
