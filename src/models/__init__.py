"""ECG classification models module."""

from .core import BaseECGModel, MultiTaskECGModel
from .cnn_scratch import CNNScratch
from .pretrained_CNN import ResNet1D14, XResNet1D101
from .efficientnet1d import EfficientNet1D_B1
from .lstm import LSTM1D
from .hybrid_cnn_lstm import HybridCNNLSTM
from .deepecg_sl import DeepECG_SL
from .hubert_ecg import HuBERT_ECG

__all__ = [
    "BaseECGModel",
    "MultiTaskECGModel",
    "CNNScratch",
    "ResNet1D14",
    "XResNet1D101",
    "EfficientNet1D_B1",
    "LSTM1D",
    "HybridCNNLSTM",
    "DeepECG_SL",
    "HuBERT_ECG",
]
