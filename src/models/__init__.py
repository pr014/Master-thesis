"""ECG classification models module."""

from .base_model import BaseECGModel
from .cnn_scratch import CNNScratch

__all__ = ["BaseECGModel", "CNNScratch"]
