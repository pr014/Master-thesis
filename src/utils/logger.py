"""Logging utility for training and evaluation."""

from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime


def setup_logger(
    name: str,
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    log_to_file: bool = True,
) -> logging.Logger:
    """Set up a logger with file and console handlers.
    
    Args:
        name: Logger name.
        log_dir: Directory for log files. If None, logs only to console.
        level: Logging level (default: INFO).
        log_to_file: Whether to log to file (default: True).
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Format for log messages
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (always)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if requested)
    if log_to_file and log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class TensorBoardLogger:
    """Wrapper for TensorBoard logging during training."""
    
    def __init__(self, log_dir: Path):
        """Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs.
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(log_dir))
            self.enabled = True
        except ImportError:
            print("Warning: TensorBoard not available. Install with: pip install tensorboard")
            self.writer = None
            self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value.
        
        Args:
            tag: Tag name for the scalar.
            value: Scalar value to log.
            step: Step/epoch number.
        """
        if self.enabled and self.writer is not None:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int) -> None:
        """Log multiple scalars.
        
        Args:
            main_tag: Main tag name.
            tag_scalar_dict: Dictionary of tag-value pairs.
            step: Step/epoch number.
        """
        if self.enabled and self.writer is not None:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        """Log a histogram.
        
        Args:
            tag: Tag name for the histogram.
            values: Values to create histogram from.
            step: Step/epoch number.
        """
        if self.enabled and self.writer is not None:
            self.writer.add_histogram(tag, values, step)
    
    def close(self) -> None:
        """Close the TensorBoard writer."""
        if self.enabled and self.writer is not None:
            self.writer.close()
