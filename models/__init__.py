from ._version import __version__

from .simple_cnn.v0_baseline import BaselineCNN
from .simple_cnn.v1_one_strand import OneStrandCNN

__all__ = ["BaselineCNN", "OneStrandCNN"]
