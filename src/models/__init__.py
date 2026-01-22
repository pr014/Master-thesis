"""ECG classification models module."""

from .base_model import BaseECGModel
from .cnn_scratch import CNNScratch
from .pretrained_CNN import ResNet1D14

__all__ = ["BaseECGModel", "CNNScratch", "ResNet1D14"]
