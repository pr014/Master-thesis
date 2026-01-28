"""LSTM models for ECG classification."""

from .unidirectional import LSTM1D_Unidirectional
from .bidirectional import LSTM1D_Bidirectional

# Alias for backward compatibility
LSTM1D = LSTM1D_Unidirectional

__all__ = ["LSTM1D_Unidirectional", "LSTM1D", "LSTM1D_Bidirectional"]
