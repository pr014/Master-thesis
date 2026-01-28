"""Core model components: base classes and wrappers."""

from .base_model import BaseECGModel
from .multi_task_model import MultiTaskECGModel

__all__ = ["BaseECGModel", "MultiTaskECGModel"]

