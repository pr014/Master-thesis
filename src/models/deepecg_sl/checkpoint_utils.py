"""Checkpoint utilities wrapper for fairseq-signals."""

import os
import sys
from pathlib import Path
from typing import Optional
import importlib.util


def get_fairseq_signals_checkpoint_utils():
    """Get checkpoint_utils from fairseq-signals.
    
    Returns:
        checkpoint_utils module from fairseq-signals
        
    Raises:
        ImportError: If fairseq-signals is not installed
    """
    try:
        from fairseq_signals.utils import checkpoint_utils
        return checkpoint_utils
    except ImportError:
        # Try alternative import path
        try:
            import fairseq_signals.utils.checkpoint_utils as checkpoint_utils
            return checkpoint_utils
        except ImportError:
            raise ImportError(
                "fairseq-signals not installed. "
                "Install with: pip install git+https://github.com/HeartWise-AI/fairseq-signals.git"
            )


def load_wcr_checkpoint(
    checkpoint_path: str,
    base_ssl_path: Optional[str] = None,
    map_location: Optional[str] = None
):
    """Load WCR checkpoint using fairseq-signals checkpoint_utils.
    
    Args:
        checkpoint_path: Path to checkpoint file (.pt)
        base_ssl_path: Path to base_ssl.pt file (optional, will be inferred)
        map_location: Device to load model on (e.g., "cuda:0", "cpu")
        
    Returns:
        Tuple of (model, args, task) from fairseq-signals
    """
    checkpoint_utils = get_fairseq_signals_checkpoint_utils()
    
    # If base_ssl_path is not provided, try to find it in the same directory
    if base_ssl_path is None:
        checkpoint_dir = Path(checkpoint_path).parent
        base_ssl_path = checkpoint_dir / "base_ssl.pt"
        if not base_ssl_path.exists():
            raise FileNotFoundError(
                f"base_ssl.pt not found in {checkpoint_dir}. "
                "Please specify base_ssl_path explicitly."
            )
        base_ssl_path = str(base_ssl_path)
    
    # Prepare overrides
    overrides = {"model_path": base_ssl_path}
    
    # Load model
    model, args, task = checkpoint_utils.load_model_and_task(
        checkpoint_path,
        arg_overrides=overrides,
        suffix=""
    )
    
    # Move to specified device if provided
    if map_location is not None:
        model = model.to(map_location)
    
    return model, args, task

