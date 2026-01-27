"""Pretrained CNN models for ECG classification."""

from .resnet1d_14 import ResNet1D14
from .xresnet1d_101 import XResNet1D101

__all__ = ["ResNet1D14", "XResNet1D101"]

