from ._version import __version__
from .basenji import Basenji
from .simple_cnn import (
    AttentionResNet18,
    AttentionResNet50,
    ConvNeXt,
    DeepCNN,
    DenseNet,
    MultiHeadAttentionResNet18,
    MultiHeadAttentionResNet50,
    ResNet18,
    ResNet50,
    ResNeXt18,
    ResNeXt50,
    SimpleCNN,
    SimpleCNN_BN,
    SimpleCNN_GELU,
    SimpleCNN_RC,
    Wannabe,
)
from .simple_rnn import DeepLSTM, SimpleLSTM
from .simple_transformer import Transformer
from .utils import fix_seeds
from .vaishnav import OneStrandCNN, VaishnavCNN

__all__ = [
    "VaishnavCNN",
    "OneStrandCNN",
    "SimpleCNN",
    "SimpleCNN_BN",
    "SimpleCNN_GELU",
    "DeepCNN",
    "SimpleCNN_RC",
    "ResNet18",
    "ResNet50",
    "ResNeXt18",
    "ResNeXt50",
    "Wannabe",
    "SimpleLSTM",
    "DeepLSTM",
    "Basenji",
    "Transformer",
    "fix_seeds",
    "AttentionResNet18",
    "AttentionResNet50",
    "MultiHeadAttentionResNet18",
    "MultiHeadAttentionResNet50",
]
