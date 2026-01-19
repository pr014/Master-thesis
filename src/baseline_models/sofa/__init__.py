"""
SOFA Score (Sequential Organ Failure Assessment)

Implementierung des SOFA Scores nach klinischem Standard.

Usage:
    from src.baseline_models.sofa import calculate_sofa_from_dict
    
    patient_data = {
        'pao2_fio2_ratio': 250,
        'is_ventilated': True,
        'platelets': 120,
        # ... weitere Parameter
    }
    
    scores = calculate_sofa_from_dict(patient_data)
    print(scores['sofa_total'])

Modules:
    - calculator: Kern-Logik für SOFA-Berechnung
    - data_loader: Lädt Daten aus MIMIC-IV
    - itemid_mappings: MIMIC-IV itemid Zuordnungen

References:
    - Vincent et al. (1996): Original SOFA Paper
    - Singer et al. (2016): Sepsis-3 Consensus
"""

from .calculator import (
    calculate_sofa_from_dict,
    calculate_sofa_batch,
    calculate_respiration_score,
    calculate_coagulation_score,
    calculate_liver_score,
    calculate_cardiovascular_score,
    calculate_cns_score,
    calculate_renal_score,
)

from .data_loader import load_sofa_data

__all__ = [
    'calculate_sofa_from_dict',
    'calculate_sofa_batch',
    'load_sofa_data',
]

