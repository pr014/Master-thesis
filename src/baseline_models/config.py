"""
Gemeinsame Konfiguration für alle Clinical Baseline Models

Hier werden Pfade und Parameter für alle Scores (SOFA, APACHE, SAPS, etc.) konfiguriert.
"""

import os
from pathlib import Path

# =============================================================================
# PFADE ZU MIMIC-IV DATEN (HIER ANPASSEN!)
# =============================================================================

# =============================================================================
# DATEN-PFADE (auf externer SSD)
# =============================================================================

# Basis-Pfad zu allen PhysioNet Files
PHYSIONET_BASE_PATH = r"D:\MA\physionet.org\files"

# MIMIC-IV Clinical Data (Labels, ICU Stays, Vitals, Labs, etc.)
MIMIC_IV_BASE_PATH = r"D:\MA\physionet.org\files\mimic-iv\3.1"

# MIMIC-IV-ECG (ECG Waveforms)
MIMIC_IV_ECG_PATH = r"D:\MA\physionet.org\files\mimic-iv-ecg\1.0"

# Spezifische Tabellen-Pfade (alle als .csv.gz komprimiert!)
MIMIC_IV_PATHS = {
    # ICU Daten
    "icustays": os.path.join(MIMIC_IV_BASE_PATH, "icu", "icustays.csv.gz"),
    "chartevents": os.path.join(MIMIC_IV_BASE_PATH, "icu", "chartevents.csv.gz"),      # 3.3 GB!
    "inputevents": os.path.join(MIMIC_IV_BASE_PATH, "icu", "inputevents.csv.gz"),
    "outputevents": os.path.join(MIMIC_IV_BASE_PATH, "icu", "outputevents.csv.gz"),
    "d_items": os.path.join(MIMIC_IV_BASE_PATH, "icu", "d_items.csv.gz"),
    
    # Hospital Daten (Labs)
    "labevents": os.path.join(MIMIC_IV_BASE_PATH, "hosp", "labevents.csv.gz"),         # 2.5 GB!
    "d_labitems": os.path.join(MIMIC_IV_BASE_PATH, "hosp", "d_labitems.csv.gz"),
    
    # Core Daten
    "patients": os.path.join(MIMIC_IV_BASE_PATH, "hosp", "patients.csv.gz"),
    "admissions": os.path.join(MIMIC_IV_BASE_PATH, "hosp", "admissions.csv.gz"),
}

# =============================================================================
# OUTPUT PFADE
# =============================================================================

# Basis Output-Pfad (für alle Baseline Models)
# Erstellt automatisch Unterordner für jeden Score: sofa/, apache/, saps/, etc.
OUTPUT_BASE_PATH = os.path.join("outputs", "baseline_models")

# Spezifische Output-Pfade pro Score
OUTPUT_PATHS = {
    "sofa": os.path.join(OUTPUT_BASE_PATH, "sofa"),
    "apache": os.path.join(OUTPUT_BASE_PATH, "apache"),
    "saps": os.path.join(OUTPUT_BASE_PATH, "saps"),
    "qsofa": os.path.join(OUTPUT_BASE_PATH, "qsofa"),
}

# =============================================================================
# GEMEINSAME PARAMETER
# =============================================================================

# Zeitfenster für Score-Berechnung (erste X Stunden nach ICU-Admission)
TIME_WINDOW_HOURS = 24

# Aggregationsmethode: "worst" (schlechtester Wert) oder "mean" (Durchschnitt)
AGGREGATION_METHOD = "worst"

# Missing Value Handling
MISSING_VALUE_SCORE = 0  # Falls Werte fehlen: Mit 0 Punkten bewerten (konservativ)

# Chunk Size für große Dateien (chartevents, labevents)
CHUNK_SIZE = 100000

# Minimale Anzahl an Komponenten für validen Score
MIN_COMPONENTS_FOR_VALID_SCORE = 4

# =============================================================================
# LOGGING & DEBUG
# =============================================================================

LOG_LEVEL = "INFO"
SAVE_INTERMEDIATE_RESULTS = True
VERBOSE = True

# =============================================================================
# KLINISCHE DEFINITIONEN (für alle Scores)
# =============================================================================

# Mechanische Beatmung
MECHANICAL_VENTILATION_ITEMIDS = [720, 223848, 223849, 224701]

# Gewicht (für Dosierungs-Berechnungen)
DEFAULT_PATIENT_WEIGHT_KG = 80
WEIGHT_ITEMIDS = [226512, 224639, 763, 762]

# Unit Conversions
BILIRUBIN_UMOL_TO_MG_FACTOR = 0.058  # μmol/l → mg/dl
CREATININ_UMOL_TO_MG_FACTOR = 0.0113  # μmol/l → mg/dl

# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

def create_output_dirs(score_name="sofa"):
    """
    Erstellt Output-Verzeichnisse für einen spezifischen Score.
    
    Args:
        score_name: Name des Scores (sofa, apache, saps, etc.)
    """
    output_path = OUTPUT_PATHS.get(score_name, os.path.join(OUTPUT_BASE_PATH, score_name))
    
    # Erstelle Haupt-Ordner
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "intermediate"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "logs"), exist_ok=True)
    
    print(f"✓ Output-Verzeichnisse erstellt: {output_path}")
    return output_path

def validate_paths():
    """Validiert, ob alle benötigten MIMIC-IV Pfade existieren."""
    if MIMIC_IV_BASE_PATH == "PFAD/ZU/MIMIC-IV":
        print("\n⚠️  WARNUNG: MIMIC_IV_BASE_PATH noch nicht gesetzt!")
        print("   → Bitte setze den Pfad in src/baseline_models/config.py\n")
        return False
    
    missing = []
    for name, path in MIMIC_IV_PATHS.items():
        if not os.path.exists(path):
            missing.append(f"  - {name}: {path}")
    
    if missing:
        print("❌ FEHLER: Folgende MIMIC-IV Dateien wurden nicht gefunden:")
        print("\n".join(missing))
        print("\n⚠️  Bitte passe die Pfade in config.py an!")
        return False
    else:
        print("✓ Alle MIMIC-IV Dateien gefunden!")
        return True

def get_project_root():
    """Gibt das Projekt-Root-Verzeichnis zurück."""
    # Von src/baseline_models/ zwei Ebenen hoch
    return Path(__file__).parent.parent.parent

# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'MIMIC_IV_BASE_PATH',
    'MIMIC_IV_PATHS',
    'OUTPUT_BASE_PATH',
    'OUTPUT_PATHS',
    'TIME_WINDOW_HOURS',
    'AGGREGATION_METHOD',
    'MISSING_VALUE_SCORE',
    'CHUNK_SIZE',
    'MIN_COMPONENTS_FOR_VALID_SCORE',
    'create_output_dirs',
    'validate_paths',
    'get_project_root',
]

