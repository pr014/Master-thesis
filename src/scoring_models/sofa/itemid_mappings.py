"""
MIMIC-IV itemid Mappings für SOFA Score Berechnung

Diese Datei enthält die Zuordnung von klinischen Variablen zu MIMIC-IV itemids.

Basiert auf:
- MIMIC-IV d_items und d_labitems Tabellen
- MIT-LCP mimic-code Repository
- Offizielle MIMIC-IV Dokumentation

WICHTIG: Diese itemids sind für MIMIC-IV validiert.
Für MIMIC-III müssen andere itemids verwendet werden!
"""

# =============================================================================
# RESPIRATION - PaO2 / FiO2
# =============================================================================

# PaO2 (Arterieller Sauerstoffpartialdruck) - mmHg
PAO2_ITEMIDS = [
    50821,  # PaO2 (labevents)
    50816,  # PO2 (labevents)
]

# FiO2 (Inspiratorische Sauerstofffraktion) - Prozent oder Dezimal
FIO2_ITEMIDS = [
    223835,  # Inspired O2 Fraction (FiO2) - Metavision
    3420,    # FiO2 - CareVue
    3422,    # FiO2 [Measured] - CareVue
    190,     # FiO2 set - CareVue
]

# Mechanische Beatmung (für SOFA Respiration 3-4 Punkte)
MECHANICAL_VENTILATION_ITEMIDS = [
    720,      # ventilator mode - CareVue
    223848,   # ventilator mode (Mechanical Ventilation) - Metavision
    223849,   # ventilator mode (iMDSoft)
    445,      # ventilator mode
    448,      # ventilator mode
    224701,   # ventilator mode (Hamilton)
]

# =============================================================================
# KOAGULATION - Thrombozyten
# =============================================================================

# Thrombozytenzahl - ×10³/μl (K/uL)
PLATELETS_ITEMIDS = [
    51265,  # Platelet Count (labevents)
]

# =============================================================================
# LEBER - Bilirubin
# =============================================================================

# Bilirubin (Total) - mg/dl
BILIRUBIN_ITEMIDS = [
    50885,  # Bilirubin, Total (labevents)
]

# =============================================================================
# KARDIOVASKULÄR - MAP & Vasopressoren
# =============================================================================

# Mean Arterial Pressure (MAP) - mmHg
MAP_ITEMIDS = [
    220052,  # Arterial Blood Pressure mean - Metavision
    220181,  # Non Invasive Blood Pressure mean - Metavision
    225312,  # ART BP mean - Metavision
    52,      # Mean Arterial Pressure - CareVue
    443,     # Manual Blood Pressure Mean(calc) - CareVue
    456,     # NBP Mean - CareVue
    6702,    # Arterial BP Mean #2 - CareVue
    3312,    # Manual BP Mean(calc) - CareVue
]

# Systolischer Blutdruck (für MAP Berechnung falls nicht direkt verfügbar)
SBP_ITEMIDS = [
    220050,  # Arterial Blood Pressure systolic - Metavision
    220179,  # Non Invasive Blood Pressure systolic - Metavision
    51,      # Arterial BP [Systolic] - CareVue
    442,     # Manual Blood Pressure Systolic Left - CareVue
    455,     # NBP [Systolic] - CareVue
]

# Diastolischer Blutdruck (für MAP Berechnung)
DBP_ITEMIDS = [
    220051,  # Arterial Blood Pressure diastolic - Metavision
    220180,  # Non Invasive Blood Pressure diastolic - Metavision
    8368,    # Arterial BP [Diastolic] - CareVue
    8440,    # Manual Blood Pressure Diastolic Left - CareVue
    8441,    # NBP [Diastolic] - CareVue
]

# Vasopressoren (inputevents) - Dosierung in μg/kg/min oder μg/min

# Dopamin
DOPAMINE_ITEMIDS = [
    221662,  # Dopamine (Metavision) - inputevents
    30043,   # Dopamine (CareVue) - inputevents
]

# Noradrenalin (Norepinephrine)
NOREPINEPHRINE_ITEMIDS = [
    221906,  # Norepinephrine (Metavision) - inputevents
    30047,   # Norepinephrine (CareVue) - inputevents
    30120,   # Norepinephrine (CareVue) - inputevents
]

# Adrenalin (Epinephrine)
EPINEPHRINE_ITEMIDS = [
    221289,  # Epinephrine (Metavision) - inputevents
    30044,   # Epinephrine (CareVue) - inputevents
    30119,   # Epinephrine (CareVue) - inputevents
]

# Dobutamin (für SOFA Cardio Score 2)
DOBUTAMINE_ITEMIDS = [
    221653,  # Dobutamine (Metavision) - inputevents
    30042,   # Dobutamine (CareVue) - inputevents
]

# =============================================================================
# ZENTRALES NERVENSYSTEM - Glasgow Coma Scale (GCS)
# =============================================================================

# GCS Total Score (3-15)
GCS_TOTAL_ITEMIDS = [
    198,     # GCS Total - CareVue
    226755,  # GCS Total - Metavision
]

# GCS Komponenten (falls Total nicht verfügbar)
GCS_EYE_ITEMIDS = [
    220739,  # GCS - Eye Opening - Metavision
    184,     # GCS - Eye Opening - CareVue
]

GCS_VERBAL_ITEMIDS = [
    223900,  # GCS - Verbal Response - Metavision
    223,     # GCS - Verbal Response - CareVue
]

GCS_MOTOR_ITEMIDS = [
    223901,  # GCS - Motor Response - Metavision
    454,     # GCS - Motor Response - CareVue
]

# =============================================================================
# NIERE - Kreatinin & Urinausscheidung
# =============================================================================

# Kreatinin (Serum) - mg/dl
CREATININE_ITEMIDS = [
    50912,  # Creatinine (labevents)
]

# Urinausscheidung - ml (outputevents)
URINE_OUTPUT_ITEMIDS = [
    40055,   # Urine Out Foley - CareVue
    43175,   # Urine - CareVue
    40069,   # Urine Out Void - CareVue
    40094,   # Urine Out Condom Cath - CareVue
    40715,   # Urine Out Suprapubic - CareVue
    40473,   # Urine Out IleoConduit - CareVue
    40085,   # Urine Out Incontinent - CareVue
    40057,   # Urine Out Rt Nephrostomy - CareVue
    40056,   # Urine Out Lt Nephrostomy - CareVue
    40405,   # Urine Out Other - CareVue
    40428,   # Urine Out Straight Cath - CareVue
    40086,   # Urine Out Incontinent - CareVue
    40096,   # Urine Out Ureteral Stent #1 - CareVue
    40651,   # Urine Out Ureteral Stent #2 - CareVue
    226559,  # Foley - Metavision
    226560,  # Void - Metavision
    226561,  # Condom Cath - Metavision
    226584,  # Ileoconduit - Metavision
    226563,  # Suprapubic - Metavision
    226564,  # R Nephrostomy - Metavision
    226565,  # L Nephrostomy - Metavision
    226567,  # Straight Cath - Metavision
    226557,  # R Ureteral Stent - Metavision
    226558,  # L Ureteral Stent - Metavision
]

# =============================================================================
# ZUSÄTZLICHE VARIABLEN
# =============================================================================

# Patientengewicht (für Vasopressor-Dosierung)
WEIGHT_ITEMIDS = [
    226512,  # Admit Wt - Metavision
    224639,  # Daily Weight - Metavision
    763,     # Weight (Admit) - CareVue
    762,     # Weight (Daily) - CareVue
]

# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

def get_all_chartevents_itemids():
    """Gibt alle itemids zurück, die aus chartevents geladen werden müssen."""
    return (
        FIO2_ITEMIDS +
        MECHANICAL_VENTILATION_ITEMIDS +
        MAP_ITEMIDS +
        SBP_ITEMIDS +
        DBP_ITEMIDS +
        GCS_TOTAL_ITEMIDS +
        GCS_EYE_ITEMIDS +
        GCS_VERBAL_ITEMIDS +
        GCS_MOTOR_ITEMIDS +
        WEIGHT_ITEMIDS
    )

def get_all_labevents_itemids():
    """Gibt alle itemids zurück, die aus labevents geladen werden müssen."""
    return (
        PAO2_ITEMIDS +
        PLATELETS_ITEMIDS +
        BILIRUBIN_ITEMIDS +
        CREATININE_ITEMIDS
    )

def get_all_inputevents_itemids():
    """Gibt alle itemids zurück, die aus inputevents geladen werden müssen."""
    return (
        DOPAMINE_ITEMIDS +
        NOREPINEPHRINE_ITEMIDS +
        EPINEPHRINE_ITEMIDS +
        DOBUTAMINE_ITEMIDS
    )

def get_all_outputevents_itemids():
    """Gibt alle itemids zurück, die aus outputevents geladen werden müssen."""
    return URINE_OUTPUT_ITEMIDS

# Vasopressor Mapping (für Kategorisierung)
VASOPRESSOR_MAPPING = {
    'dopamine': DOPAMINE_ITEMIDS,
    'norepinephrine': NOREPINEPHRINE_ITEMIDS,
    'epinephrine': EPINEPHRINE_ITEMIDS,
    'dobutamine': DOBUTAMINE_ITEMIDS,
}

# =============================================================================
# VALIDIERUNG
# =============================================================================

def validate_itemids():
    """
    Überprüft, ob alle itemid-Listen eindeutige Werte enthalten.
    """
    all_chart = get_all_chartevents_itemids()
    all_lab = get_all_labevents_itemids()
    all_input = get_all_inputevents_itemids()
    all_output = get_all_outputevents_itemids()
    
    # Prüfe auf Duplikate innerhalb jeder Liste
    for name, itemids in [
        ("chartevents", all_chart),
        ("labevents", all_lab),
        ("inputevents", all_input),
        ("outputevents", all_output)
    ]:
        if len(itemids) != len(set(itemids)):
            duplicates = [x for x in itemids if itemids.count(x) > 1]
            print(f"⚠️  Warnung: Duplikate in {name}: {set(duplicates)}")
    
    print(f"✓ itemid Validation abgeschlossen")
    print(f"  - chartevents: {len(set(all_chart))} unique itemids")
    print(f"  - labevents: {len(set(all_lab))} unique itemids")
    print(f"  - inputevents: {len(set(all_input))} unique itemids")
    print(f"  - outputevents: {len(set(all_output))} unique itemids")

# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    # Respiration
    'PAO2_ITEMIDS',
    'FIO2_ITEMIDS',
    'MECHANICAL_VENTILATION_ITEMIDS',
    
    # Koagulation
    'PLATELETS_ITEMIDS',
    
    # Leber
    'BILIRUBIN_ITEMIDS',
    
    # Kardiovaskulär
    'MAP_ITEMIDS',
    'SBP_ITEMIDS',
    'DBP_ITEMIDS',
    'DOPAMINE_ITEMIDS',
    'NOREPINEPHRINE_ITEMIDS',
    'EPINEPHRINE_ITEMIDS',
    'DOBUTAMINE_ITEMIDS',
    'VASOPRESSOR_MAPPING',
    
    # ZNS
    'GCS_TOTAL_ITEMIDS',
    'GCS_EYE_ITEMIDS',
    'GCS_VERBAL_ITEMIDS',
    'GCS_MOTOR_ITEMIDS',
    
    # Niere
    'CREATININE_ITEMIDS',
    'URINE_OUTPUT_ITEMIDS',
    
    # Zusatz
    'WEIGHT_ITEMIDS',
    
    # Hilfsfunktionen
    'get_all_chartevents_itemids',
    'get_all_labevents_itemids',
    'get_all_inputevents_itemids',
    'get_all_outputevents_itemids',
    'validate_itemids',
]

