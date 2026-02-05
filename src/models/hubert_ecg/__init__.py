"""HuBERT-ECG model for multi-task learning."""

from .encoder import HuBERTEncoder
from .model import HuBERT_ECG

__all__ = ["HuBERTEncoder", "HuBERT_ECG"]

