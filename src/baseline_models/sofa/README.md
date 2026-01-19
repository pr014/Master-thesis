# SOFA Score Berechnung für MIMIC-IV

## Übersicht

Dieses Modul berechnet den **SOFA Score (Sequential Organ Failure Assessment)** aus MIMIC-IV Clinical Daten.

Der SOFA Score ist ein etablierter klinischer Severity-Score zur Bewertung von Organversagen auf der Intensivstation und wird als Baseline für ML-Modell-Vergleiche verwendet.

## SOFA Score Definition

**Score Range:** 0-24 Punkte (6 Organsysteme × 0-4 Punkte)

### Die 6 Organsysteme:

1. **Respiration** (PaO₂/FiO₂ Ratio)
2. **Koagulation** (Thrombozytenzahl)
3. **Leber** (Bilirubin)
4. **Kardiovaskulär** (MAP + Vasopressoren)
5. **Zentrales Nervensystem** (Glasgow Coma Scale)
6. **Niere** (Kreatinin + Urinausscheidung)

Jedes System wird mit 0-4 Punkten bewertet, wobei höhere Werte schlechtere Organfunktion bedeuten.

## Ordnerstruktur

**📁 Detaillierte Übersicht:** Siehe [`STRUCTURE.md`](STRUCTURE.md)

```
sofa_score/
├── README.md                  # Diese Datei (Vollständige Doku)
├── QUICKSTART.md             # 5-Minuten Schnellstart
├── STRUCTURE.md               # ⭐ Ordnerstruktur & Übersicht
├── requirements.txt           # Python Dependencies
├── config.py                  # ⭐ KONFIGURATION (HIER PFADE SETZEN!)
├── itemid_mappings.py         # MIMIC-IV itemid Mappings
├── data_loader.py             # Lädt und extrahiert MIMIC-IV Daten
├── sofa_calculator.py         # ⭐ KERN: SOFA Score Berechnung
├── utils.py                   # Hilfsfunktionen
├── run_sofa_calculation.py    # ⭐ MAIN SCRIPT (Hier starten!)
└── example_usage.py           # Beispiele & Tests
```

## Installation

```bash
cd sofa_score
pip install -r requirements.txt
```

## Verwendung

### 1. Konfiguration anpassen

Öffne `config.py` und setze deine Pfade:

```python
MIMIC_IV_PATH = "PFAD/ZU/DEINER/MIMIC-IV/DATEN"
OUTPUT_PATH = "PFAD/FÜR/OUTPUT"
```

### 2. SOFA Score berechnen

```bash
python run_sofa_calculation.py
```

### 3. Output

Das Script erstellt:
- `sofa_scores.csv` - SOFA Scores für alle Patienten
- `sofa_components.csv` - Einzelne Komponenten-Scores (zur Analyse)
- `sofa_statistics.txt` - Zusammenfassung und Statistiken

## SOFA Score Berechnungslogik

### Zeitfenster
- **Standard:** Erste 24 Stunden nach ICU-Admission
- **Methode:** Worst case (schlechtester Wert im Zeitfenster)

### Datenquellen (MIMIC-IV)
- `icu/chartevents.csv` - Vitals, GCS, PaO₂, FiO₂
- `hosp/labevents.csv` - Laborwerte (Bilirubin, Kreatinin, Thrombozyten)
- `icu/inputevents.csv` - Vasopressoren-Dosierungen
- `icu/outputevents.csv` - Urinausscheidung
- `icu/icustays.csv` - ICU Admission Zeitstempel

### Punktevergabe

#### 1. Respiration (PaO₂/FiO₂)
- **≥400:** 0 Punkte
- **<400:** 1 Punkt
- **<300:** 2 Punkte
- **<200:** 3 Punkte (mit mechanischer Beatmung)
- **<100:** 4 Punkte (mit mechanischer Beatmung)

#### 2. Koagulation (Thrombozyten ×10³/μl)
- **≥150:** 0 Punkte
- **<150:** 1 Punkt
- **<100:** 2 Punkte
- **<50:** 3 Punkte
- **<20:** 4 Punkte

#### 3. Leber (Bilirubin mg/dl)
- **<1.2:** 0 Punkte
- **1.2-1.9:** 1 Punkt
- **2.0-5.9:** 2 Punkte
- **6.0-11.9:** 3 Punkte
- **≥12.0:** 4 Punkte

#### 4. Kardiovaskulär (MAP mmHg / Vasopressoren)
- **MAP ≥70:** 0 Punkte
- **MAP <70:** 1 Punkt
- **Dopamin ≤5 oder Dobutamin (beliebig):** 2 Punkte
- **Dopamin >5 ODER Adrenalin ≤0.1 ODER Noradrenalin ≤0.1:** 3 Punkte
- **Dopamin >15 ODER Adrenalin >0.1 ODER Noradrenalin >0.1:** 4 Punkte

*(Dosierungen in μg/kg/min)*

#### 5. Zentrales Nervensystem (Glasgow Coma Scale)
- **GCS 15:** 0 Punkte
- **GCS 13-14:** 1 Punkt
- **GCS 10-12:** 2 Punkte
- **GCS 6-9:** 3 Punkte
- **GCS <6:** 4 Punkte

#### 6. Niere (Kreatinin mg/dl / Urinausscheidung ml/Tag)
- **<1.2:** 0 Punkte
- **1.2-1.9:** 1 Punkt
- **2.0-3.4:** 2 Punkte
- **3.5-4.9 ODER Urin <500 ml/Tag:** 3 Punkte
- **≥5.0 ODER Urin <200 ml/Tag:** 4 Punkte

## Validierung

Der Code wurde gegen die offizielle SOFA Score Definition validiert:
- Vincent et al. (1996): "The SOFA (Sepsis-related Organ Failure Assessment) score"
- Singer et al. (2016): "The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3)"

## Referenzen

1. **Original SOFA Paper:**
   Vincent JL, et al. (1996). "The SOFA (Sepsis-related Organ Failure Assessment) score to describe organ dysfunction/failure." Intensive Care Medicine.

2. **Sepsis-3 Consensus:**
   Singer M, et al. (2016). "The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3)." JAMA.

3. **MIMIC-IV Documentation:**
   https://mimic.mit.edu/docs/iv/

4. **MIT-LCP mimic-code Repository:**
   https://github.com/MIT-LCP/mimic-code

## Hinweise

- **Missing Data:** Falls Werte fehlen, wird die Komponente mit 0 Punkten bewertet (konservativ)
- **Beatmung:** Für Respiration-Score 3-4 wird mechanische Beatmung benötigt (aus `ventdurations` oder `chartevents`)
- **Vasopressoren:** Dosierungen werden gewichtsbasiert berechnet (μg/kg/min)

## Lizenz

Dieses Code-Modul ist Teil der Masterarbeit und basiert auf öffentlich verfügbaren SOFA Score Definitionen.

