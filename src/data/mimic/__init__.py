"""MIMIC-IV clinical data pipeline.

This package provides utilities for loading and processing MIMIC-IV clinical data:
- ICU Stays
- Chart Events (Vitals, Monitoring)
- Lab Events
- Input Events (Medications)
- Output Events

Not ECG-specific - for general clinical data processing.
"""

from .pipeline import MimicIVPipeline, create_pipeline

__all__ = [
    "MimicIVPipeline",
    "create_pipeline",
]

