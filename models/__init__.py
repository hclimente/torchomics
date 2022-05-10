from ._version import __version__
from .basenji import Basenji
from .simple_cnn import DeepCNN, SimpleCNN, SimpleCNN_BN, SimpleCNN_GELU
from .utils import fix_seeds
from .vaishnav import OneStrandCNN, VaishnavCNN

__all__ = [
    "VaishnavCNN",
    "OneStrandCNN",
    "SimpleCNN",
    "SimpleCNN_BN",
    "SimpleCNN_GELU",
    "DeepCNN",
    "Basenji",
    "fix_seeds",
]
