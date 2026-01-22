"""
SOFA Score Berechnung - Kern-Logik

Implementiert die offizielle SOFA (Sequential Organ Failure Assessment) Score Berechnung
nach den klinischen Standards.

Referenzen:
- Vincent et al. (1996): Original SOFA Paper
- Singer et al. (2016): Sepsis-3 Consensus

Jedes Organsystem wird mit 0-4 Punkten bewertet.
Gesamt-SOFA: 0-24 Punkte (höher = schlechter)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple

# =============================================================================
# 1. RESPIRATION - PaO2/FiO2 Ratio
# =============================================================================

def calculate_respiration_score(
    pao2_fio2_ratio: Optional[float],
    is_mechanically_ventilated: bool = False
) -> int:
    """
    Berechnet SOFA Respiration Score basierend auf PaO2/FiO2 Ratio.
    
    Punktevergabe:
    - ≥400: 0 Punkte
    - <400: 1 Punkt
    - <300: 2 Punkte
    - <200 (mit Beatmung): 3 Punkte
    - <100 (mit Beatmung): 4 Punkte
    
    Args:
        pao2_fio2_ratio: PaO2/FiO2 Verhältnis (mmHg)
        is_mechanically_ventilated: Ob Patient mechanisch beatmet wird
        
    Returns:
        Score (0-4)
    """
    if pao2_fio2_ratio is None or pd.isna(pao2_fio2_ratio):
        return 0  # Missing = 0 (konservativ)
    
    if pao2_fio2_ratio >= 400:
        return 0
    elif pao2_fio2_ratio >= 300:
        return 1
    elif pao2_fio2_ratio >= 200:
        return 2
    elif pao2_fio2_ratio >= 100:
        # Score 3 nur wenn mechanisch beatmet
        return 3 if is_mechanically_ventilated else 2
    else:  # <100
        # Score 4 nur wenn mechanisch beatmet
        return 4 if is_mechanically_ventilated else 2

# =============================================================================
# 2. KOAGULATION - Thrombozytenzahl
# =============================================================================

def calculate_coagulation_score(platelets: Optional[float]) -> int:
    """
    Berechnet SOFA Koagulation Score basierend auf Thrombozytenzahl.
    
    Punktevergabe (×10³/μl):
    - ≥150: 0 Punkte
    - <150: 1 Punkt
    - <100: 2 Punkte
    - <50: 3 Punkte
    - <20: 4 Punkte
    
    Args:
        platelets: Thrombozytenzahl in ×10³/μl (K/uL)
        
    Returns:
        Score (0-4)
    """
    if platelets is None or pd.isna(platelets):
        return 0  # Missing = 0 (konservativ)
    
    if platelets >= 150:
        return 0
    elif platelets >= 100:
        return 1
    elif platelets >= 50:
        return 2
    elif platelets >= 20:
        return 3
    else:  # <20
        return 4

# =============================================================================
# 3. LEBER - Bilirubin
# =============================================================================

def calculate_liver_score(bilirubin: Optional[float]) -> int:
    """
    Berechnet SOFA Leber Score basierend auf Bilirubin.
    
    Punktevergabe (mg/dl):
    - <1.2: 0 Punkte
    - 1.2-1.9: 1 Punkt
    - 2.0-5.9: 2 Punkte
    - 6.0-11.9: 3 Punkte
    - ≥12.0: 4 Punkte
    
    Args:
        bilirubin: Bilirubin Total in mg/dl
        
    Returns:
        Score (0-4)
    """
    if bilirubin is None or pd.isna(bilirubin):
        return 0  # Missing = 0 (konservativ)
    
    if bilirubin < 1.2:
        return 0
    elif bilirubin < 2.0:
        return 1
    elif bilirubin < 6.0:
        return 2
    elif bilirubin < 12.0:
        return 3
    else:  # ≥12.0
        return 4

# =============================================================================
# 4. KARDIOVASKULÄR - MAP & Vasopressoren
# =============================================================================

def calculate_cardiovascular_score(
    map_value: Optional[float],
    dopamine_dose: float = 0.0,
    dobutamine_dose: float = 0.0,
    epinephrine_dose: float = 0.0,
    norepinephrine_dose: float = 0.0
) -> int:
    """
    Berechnet SOFA Kardiovaskulär Score basierend auf MAP und Vasopressoren.
    
    Punktevergabe:
    - MAP ≥70: 0 Punkte
    - MAP <70: 1 Punkt
    - Dopamin ≤5 ODER Dobutamin (beliebig): 2 Punkte
    - Dopamin >5 ODER Adrenalin ≤0.1 ODER Noradrenalin ≤0.1: 3 Punkte
    - Dopamin >15 ODER Adrenalin >0.1 ODER Noradrenalin >0.1: 4 Punkte
    
    Dosierungen in μg/kg/min
    
    Args:
        map_value: Mean Arterial Pressure in mmHg
        dopamine_dose: Dopamin Dosis in μg/kg/min
        dobutamine_dose: Dobutamin Dosis in μg/kg/min
        epinephrine_dose: Adrenalin Dosis in μg/kg/min
        norepinephrine_dose: Noradrenalin Dosis in μg/kg/min
        
    Returns:
        Score (0-4)
    """
    # Score 4: Höchste Vasopressor-Dosen
    if dopamine_dose > 15 or epinephrine_dose > 0.1 or norepinephrine_dose > 0.1:
        return 4
    
    # Score 3: Mittlere Vasopressor-Dosen
    if (dopamine_dose > 5 or 
        (epinephrine_dose > 0 and epinephrine_dose <= 0.1) or
        (norepinephrine_dose > 0 and norepinephrine_dose <= 0.1)):
        return 3
    
    # Score 2: Niedrige Vasopressor-Dosen
    if (dopamine_dose > 0 and dopamine_dose <= 5) or dobutamine_dose > 0:
        return 2
    
    # Score 1: Keine Vasopressoren, aber MAP <70
    if map_value is not None and not pd.isna(map_value):
        if map_value < 70:
            return 1
        else:
            return 0
    
    # Kein MAP Wert vorhanden, aber auch keine Vasopressoren
    return 0

# =============================================================================
# 5. ZENTRALES NERVENSYSTEM - Glasgow Coma Scale (GCS)
# =============================================================================

def calculate_cns_score(gcs: Optional[float]) -> int:
    """
    Berechnet SOFA ZNS Score basierend auf Glasgow Coma Scale.
    
    Punktevergabe:
    - GCS 15: 0 Punkte
    - GCS 13-14: 1 Punkt
    - GCS 10-12: 2 Punkte
    - GCS 6-9: 3 Punkte
    - GCS <6: 4 Punkte
    
    Args:
        gcs: Glasgow Coma Scale (3-15)
        
    Returns:
        Score (0-4)
    """
    if gcs is None or pd.isna(gcs):
        return 0  # Missing = 0 (konservativ)
    
    # GCS sollte zwischen 3 und 15 liegen
    gcs = max(3, min(15, gcs))
    
    if gcs == 15:
        return 0
    elif gcs >= 13:
        return 1
    elif gcs >= 10:
        return 2
    elif gcs >= 6:
        return 3
    else:  # <6
        return 4

# =============================================================================
# 6. NIERE - Kreatinin & Urinausscheidung
# =============================================================================

def calculate_renal_score(
    creatinine: Optional[float],
    urine_output_24h: Optional[float] = None
) -> int:
    """
    Berechnet SOFA Niere Score basierend auf Kreatinin und Urinausscheidung.
    
    Punktevergabe:
    - Kreatinin <1.2 mg/dl: 0 Punkte
    - Kreatinin 1.2-1.9 mg/dl: 1 Punkt
    - Kreatinin 2.0-3.4 mg/dl: 2 Punkte
    - Kreatinin 3.5-4.9 mg/dl ODER Urin <500 ml/24h: 3 Punkte
    - Kreatinin ≥5.0 mg/dl ODER Urin <200 ml/24h: 4 Punkte
    
    Args:
        creatinine: Kreatinin in mg/dl
        urine_output_24h: Urinausscheidung in ml/24h
        
    Returns:
        Score (0-4)
    """
    # Bestimme Score basierend auf Kreatinin
    creat_score = 0
    if creatinine is not None and not pd.isna(creatinine):
        if creatinine >= 5.0:
            creat_score = 4
        elif creatinine >= 3.5:
            creat_score = 3
        elif creatinine >= 2.0:
            creat_score = 2
        elif creatinine >= 1.2:
            creat_score = 1
        else:
            creat_score = 0
    
    # Bestimme Score basierend auf Urinausscheidung
    urine_score = 0
    if urine_output_24h is not None and not pd.isna(urine_output_24h):
        if urine_output_24h < 200:
            urine_score = 4
        elif urine_output_24h < 500:
            urine_score = 3
    
    # Nimm das Maximum von beiden (schlechterer Wert)
    return max(creat_score, urine_score)

# =============================================================================
# GESAMT-SOFA SCORE
# =============================================================================

def calculate_total_sofa_score(
    respiration_score: int,
    coagulation_score: int,
    liver_score: int,
    cardiovascular_score: int,
    cns_score: int,
    renal_score: int
) -> int:
    """
    Berechnet den Gesamt-SOFA Score als Summe aller Komponenten.
    
    Args:
        respiration_score: Respiration Score (0-4)
        coagulation_score: Koagulation Score (0-4)
        liver_score: Leber Score (0-4)
        cardiovascular_score: Kardiovaskulär Score (0-4)
        cns_score: ZNS Score (0-4)
        renal_score: Niere Score (0-4)
        
    Returns:
        Gesamt-SOFA Score (0-24)
    """
    total = (
        respiration_score +
        coagulation_score +
        liver_score +
        cardiovascular_score +
        cns_score +
        renal_score
    )
    
    # Stelle sicher, dass Score im gültigen Bereich liegt
    return max(0, min(24, total))

# =============================================================================
# CONVENIENCE FUNKTION
# =============================================================================

def calculate_sofa_from_dict(patient_data: Dict) -> Dict:
    """
    Berechnet SOFA Score aus einem Dictionary mit Patient-Daten.
    
    Args:
        patient_data: Dictionary mit klinischen Parametern
            Keys: pao2_fio2_ratio, is_ventilated, platelets, bilirubin,
                  map, dopamine, dobutamine, epinephrine, norepinephrine,
                  gcs, creatinine, urine_output_24h
    
    Returns:
        Dictionary mit allen Scores (Komponenten + Total)
    """
    # Berechne Komponenten-Scores
    resp_score = calculate_respiration_score(
        patient_data.get('pao2_fio2_ratio'),
        patient_data.get('is_ventilated', False)
    )
    
    coag_score = calculate_coagulation_score(
        patient_data.get('platelets')
    )
    
    liver_score = calculate_liver_score(
        patient_data.get('bilirubin')
    )
    
    cardio_score = calculate_cardiovascular_score(
        patient_data.get('map'),
        patient_data.get('dopamine', 0.0),
        patient_data.get('dobutamine', 0.0),
        patient_data.get('epinephrine', 0.0),
        patient_data.get('norepinephrine', 0.0)
    )
    
    cns_score = calculate_cns_score(
        patient_data.get('gcs')
    )
    
    renal_score = calculate_renal_score(
        patient_data.get('creatinine'),
        patient_data.get('urine_output_24h')
    )
    
    # Berechne Gesamt-Score
    total_score = calculate_total_sofa_score(
        resp_score, coag_score, liver_score,
        cardio_score, cns_score, renal_score
    )
    
    return {
        'sofa_respiration': resp_score,
        'sofa_coagulation': coag_score,
        'sofa_liver': liver_score,
        'sofa_cardiovascular': cardio_score,
        'sofa_cns': cns_score,
        'sofa_renal': renal_score,
        'sofa_total': total_score,
    }

# =============================================================================
# BATCH-BERECHNUNG FÜR DATAFRAME
# =============================================================================

def calculate_sofa_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet SOFA Scores für einen kompletten DataFrame.
    
    Args:
        df: DataFrame mit Spalten für alle benötigten Parameter
        
    Returns:
        DataFrame mit zusätzlichen SOFA Score Spalten
    """
    # Erstelle Kopie um Original nicht zu verändern
    result_df = df.copy()
    
    # Berechne jeden Komponenten-Score
    result_df['sofa_respiration'] = result_df.apply(
        lambda row: calculate_respiration_score(
            row.get('pao2_fio2_ratio'),
            row.get('is_ventilated', False)
        ), axis=1
    )
    
    # Platelets können fehlen
    if 'platelets' in result_df.columns:
        result_df['sofa_coagulation'] = result_df['platelets'].apply(calculate_coagulation_score)
    else:
        result_df['sofa_coagulation'] = 0
        print("  ⚠️  Thrombozyten-Daten fehlen - Koagulation Score = 0")
    
    # Bilirubin kann fehlen
    if 'bilirubin' in result_df.columns:
        result_df['sofa_liver'] = result_df['bilirubin'].apply(calculate_liver_score)
    else:
        result_df['sofa_liver'] = 0
        print("  ⚠️  Bilirubin-Daten fehlen - Leber Score = 0")
    
    result_df['sofa_cardiovascular'] = result_df.apply(
        lambda row: calculate_cardiovascular_score(
            row.get('map'),
            row.get('dopamine', 0.0),
            row.get('dobutamine', 0.0),
            row.get('epinephrine', 0.0),
            row.get('norepinephrine', 0.0)
        ), axis=1
    )
    
    # GCS kann fehlen - dann 0 Punkte (konservativ)
    if 'gcs' in result_df.columns:
        result_df['sofa_cns'] = result_df['gcs'].apply(calculate_cns_score)
    else:
        result_df['sofa_cns'] = 0
        print("  ⚠️  GCS-Daten fehlen - CNS Score = 0 für alle")
    
    result_df['sofa_renal'] = result_df.apply(
        lambda row: calculate_renal_score(
            row.get('creatinine'),
            row.get('urine_output_24h')
        ), axis=1
    )
    
    # Berechne Gesamt-Score
    result_df['sofa_total'] = result_df.apply(
        lambda row: calculate_total_sofa_score(
            row['sofa_respiration'],
            row['sofa_coagulation'],
            row['sofa_liver'],
            row['sofa_cardiovascular'],
            row['sofa_cns'],
            row['sofa_renal']
        ), axis=1
    )
    
    return result_df

# =============================================================================
# VALIDIERUNG & TESTS
# =============================================================================

def validate_sofa_calculation():
    """
    Führt Unit-Tests für SOFA Score Berechnung durch.
    Testet gegen bekannte Beispiele aus der Literatur.
    """
    print("🧪 Validiere SOFA Score Berechnung...")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Gesunder Patient (alle Werte normal)
    tests_total += 1
    healthy = {
        'pao2_fio2_ratio': 450,
        'is_ventilated': False,
        'platelets': 200,
        'bilirubin': 0.8,
        'map': 85,
        'dopamine': 0,
        'dobutamine': 0,
        'epinephrine': 0,
        'norepinephrine': 0,
        'gcs': 15,
        'creatinine': 0.9,
        'urine_output_24h': 2000
    }
    result = calculate_sofa_from_dict(healthy)
    if result['sofa_total'] == 0:
        print("  ✓ Test 1: Gesunder Patient (Score = 0)")
        tests_passed += 1
    else:
        print(f"  ✗ Test 1 FAILED: Erwartet 0, bekommen {result['sofa_total']}")
    
    # Test 2: Schwerkranker Patient
    tests_total += 1
    critical = {
        'pao2_fio2_ratio': 80,
        'is_ventilated': True,
        'platelets': 15,
        'bilirubin': 15.0,
        'map': 50,
        'dopamine': 20,
        'dobutamine': 0,
        'epinephrine': 0.2,
        'norepinephrine': 0.15,
        'gcs': 4,
        'creatinine': 6.0,
        'urine_output_24h': 100
    }
    result = calculate_sofa_from_dict(critical)
    # Sollte nahe am Maximum sein (20-24)
    if result['sofa_total'] >= 20:
        print(f"  ✓ Test 2: Schwerkranker Patient (Score = {result['sofa_total']})")
        tests_passed += 1
    else:
        print(f"  ✗ Test 2 FAILED: Erwartet ≥20, bekommen {result['sofa_total']}")
    
    # Test 3: Einzelne Komponenten
    tests_total += 1
    if calculate_respiration_score(150, True) == 3:
        print("  ✓ Test 3: Respiration Score korrekt")
        tests_passed += 1
    else:
        print("  ✗ Test 3 FAILED: Respiration Score")
    
    tests_total += 1
    if calculate_coagulation_score(45) == 3:
        print("  ✓ Test 4: Koagulation Score korrekt")
        tests_passed += 1
    else:
        print("  ✗ Test 4 FAILED: Koagulation Score")
    
    tests_total += 1
    if calculate_liver_score(8.5) == 3:
        print("  ✓ Test 5: Leber Score korrekt")
        tests_passed += 1
    else:
        print("  ✗ Test 5 FAILED: Leber Score")
    
    print(f"\n✅ {tests_passed}/{tests_total} Tests bestanden")
    
    return tests_passed == tests_total

# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'calculate_respiration_score',
    'calculate_coagulation_score',
    'calculate_liver_score',
    'calculate_cardiovascular_score',
    'calculate_cns_score',
    'calculate_renal_score',
    'calculate_total_sofa_score',
    'calculate_sofa_from_dict',
    'calculate_sofa_batch',
    'validate_sofa_calculation',
]

