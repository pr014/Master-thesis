"""
Gemeinsame Konfiguration für alle Clinical Baseline Models

Hier werden Pfade und Parameter für alle Scores (SOFA, APACHE, SAPS, etc.) konfiguriert.
"""

import os
from pathlib import Path

# =============================================================================
# PFADE ZU MIMIC-IV DATEN
# =============================================================================
# Projekt-Root → data/labeling/:
#   - labels_csv/: icustays, patients, admissions (wie im Training)
#   - mimic-iv/<tabelle>.csv/<tabelle>.csv: große ICU/Hosp-Tabellen (unkomprimiert)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LABELING_DIR = _PROJECT_ROOT / "data" / "labeling"
_LABELS_CSV = _LABELING_DIR / "labels_csv"
_MIMIC_IV_DIR = _LABELING_DIR / "mimic-iv"


def _mimic_iv_table_csv(name: str) -> str:
    """Pfad zu mimic-iv/<name>.csv/<name>.csv (PhysioNet-Entpack-Layout)."""
    return str(_MIMIC_IV_DIR / f"{name}.csv" / f"{name}.csv")


# Basis für Dokumentation / Overrides (SOFA nutzt primär MIMIC_IV_PATHS)
PHYSIONET_BASE_PATH = str(_LABELING_DIR)

# Logischer MIMIC-IV-Ordner (große Tabellen)
MIMIC_IV_BASE_PATH = str(_MIMIC_IV_DIR)

# ECG-Waveforms: falls woanders, hier manuell setzen (wird von SOFA-Loader nicht genutzt)
MIMIC_IV_ECG_PATH = str(_LABELING_DIR)

# Konkrete Tabellen (unkomprimierte .csv wie im Repo unter data/labeling/)
MIMIC_IV_PATHS = {
    "icustays": str(_LABELS_CSV / "icustays.csv"),
    "chartevents": _mimic_iv_table_csv("chartevents"),
    "inputevents": _mimic_iv_table_csv("inputevents"),
    "outputevents": _mimic_iv_table_csv("outputevents"),
    "d_items": _mimic_iv_table_csv("d_items"),
    "labevents": _mimic_iv_table_csv("labevents"),
    "d_labitems": _mimic_iv_table_csv("d_labitems"),
    "patients": str(_LABELS_CSV / "patients.csv"),
    "admissions": str(_LABELS_CSV / "admissions.csv"),
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
        print("   → Bitte setze die Pfade in src/scoring_models/config.py\n")
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
    return _PROJECT_ROOT

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

