"""ECG classification models module."""

from .core import BaseECGModel, MultiTaskECGModel
from .cnn_scratch import CNNScratch
from .lstm import LSTM1D
from .hybrid_cnn_lstm import HybridCNNLSTM
from .deepecg_sl import DeepECG_SL
from .hubert_ecg import HuBERT_ECG

__all__ = [
    "BaseECGModel",
    "MultiTaskECGModel",
    "CNNScratch",
    "LSTM1D",
    "HybridCNNLSTM",
    "DeepECG_SL",
    "HuBERT_ECG",
]
