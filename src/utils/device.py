"""GPU/CPU device handling utilities for PyTorch."""

import torch
from typing import Optional


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the appropriate device (GPU/CPU) for PyTorch.
    
    Args:
        device: Optional device string ('cuda', 'cpu', 'cuda:0', etc.)
                If None, automatically selects best available device.
    
    Returns:
        torch.device: The selected device.
    
    Examples:
        >>> device = get_device()  # Auto-select
        >>> device = get_device('cuda:0')  # Specific GPU
        >>> device = get_device('cpu')  # Force CPU
    """
    if device is None:
        # Auto-select: Use CUDA if available, otherwise CPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return torch.device(device)


def get_device_count() -> int:
    """Get the number of available GPUs.
    
    Returns:
        int: Number of CUDA devices available (0 if no GPU).
    """
    return torch.cuda.device_count()


def is_cuda_available() -> bool:
    """Check if CUDA is available.
    
    Returns:
        bool: True if CUDA is available, False otherwise.
    """
    return torch.cuda.is_available()


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Sets seeds for:
    - PyTorch (CPU and CUDA)
    - NumPy (if available)
    - Python random module
    
    Args:
        seed: Random seed value (default: 42).
    """
    import random
    import numpy as np
    
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # Python random
    random.seed(seed)
    
    # PyTorch deterministic operations (may slow down training)
    # Uncomment if you need full reproducibility:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def move_to_device(tensor_or_model, device: torch.device):
    """Move tensor or model to specified device.
    
    Args:
        tensor_or_model: PyTorch tensor or nn.Module.
        device: Target device.
    
    Returns:
        Tensor or model on the specified device.
    """
    return tensor_or_model.to(device)
