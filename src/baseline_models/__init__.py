"""
Clinical Baseline Models für MA Thesis

Dieses Modul enthält Implementierungen etablierter klinischer Scores
zur Verwendung als Baselines für ML-Modell-Vergleiche:

- SOFA (Sequential Organ Failure Assessment)
- APACHE II/III/IV (zukünftig)
- SAPS II/III (zukünftig)
- qSOFA (zukünftig)

Usage:
    from src.baseline_models.sofa import calculate_sofa_from_dict
    from src.baseline_models import load_mimic_data
"""

__version__ = "1.0.0"

# Gemeinsame Imports für alle Scores
from .config import MIMIC_IV_BASE_PATH, OUTPUT_BASE_PATH, OUTPUT_PATHS
from .utils import validate_data, save_results

__all__ = [
    'MIMIC_IV_BASE_PATH',
    'OUTPUT_BASE_PATH',
    'OUTPUT_PATHS',
    'validate_data',
    'save_results',
]

