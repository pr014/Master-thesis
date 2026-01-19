# Clinical Baseline Models

Implementierungen etablierter klinischer Severity Scores für Baseline-Vergleiche mit ML-Modellen.

## Struktur

```
baseline_models/
│
├── README.md                 # Diese Datei
├── config.py                 # ⭐ Gemeinsame Konfiguration (PFADE HIER!)
├── utils.py                  # Gemeinsame Hilfsfunktionen
│
├── sofa/                     # ✅ SOFA Score (implementiert)
│   ├── README.md
│   ├── calculator.py
│   ├── data_loader.py
│   └── itemid_mappings.py
│
├── apache/                   # 🔜 APACHE Score (zukünftig)
│   └── ...
│
├── saps/                     # 🔜 SAPS Score (zukünftig)
│   └── ...
│
└── qsofa/                    # 🔜 qSOFA Score (zukünftig)
    └── ...
```

## Scripts zum Ausführen

Die Main-Scripts liegen in `scripts/baseline_models/`:

```bash
# SOFA Score berechnen
python scripts/baseline_models/calculate_sofa.py

# Zukünftig: Alle Scores vergleichen
python scripts/baseline_models/compare_all_scores.py
```

## Verwendung in Thesis

### 1. Scores berechnen
```bash
cd scripts/baseline_models
python calculate_sofa.py
```

### 2. Ergebnisse finden
```
outputs/baseline_models/
├── sofa/
│   ├── sofa_scores.csv
│   └── sofa_statistics.txt
├── apache/
└── saps/
```

### 3. In ML-Pipeline verwenden
```python
from src.baseline_models.sofa import calculate_sofa_from_dict

# Berechne SOFA für Patient
scores = calculate_sofa_from_dict(patient_data)
```

## Konfiguration

**Pfade setzen in:** `src/baseline_models/config.py`

```python
MIMIC_IV_BASE_PATH = "E:/MIMIC-IV"  # DEIN PFAD!
OUTPUT_PATH = "outputs/baseline_models"
```

## Implementierte Scores

### ✅ SOFA (Sequential Organ Failure Assessment)
- **Status:** Vollständig implementiert
- **Dokumentation:** `sofa/README.md`
- **Script:** `scripts/baseline_models/calculate_sofa.py`
- **Usage:** `from src.baseline_models.sofa import calculate_sofa_from_dict`

### 🔜 APACHE II
- **Status:** Geplant
- **Zweck:** Alternativer Severity Score
- **Implementierung:** Nach SOFA

### 🔜 SAPS II
- **Status:** Geplant
- **Zweck:** Weiterer etablierter Score
- **Implementierung:** Nach APACHE

### 🔜 qSOFA
- **Status:** Geplant
- **Zweck:** Vereinfachter Score (nur 3 Variablen)
- **Implementierung:** Nach SAPS

## Vorteile dieser Struktur

✅ **Gemeinsame Konfiguration** - Ein config.py für alle Scores
✅ **Wiederverwendbare Utils** - Gemeinsame Hilfsfunktionen
✅ **Skalierbar** - Einfach weitere Scores hinzufügen
✅ **Integriert** - Teil der src/ Struktur
✅ **Organisiert** - Scripts getrennt von Modulen

## Thesis Integration

**Methods Section:**
> "We compared our ML models against established clinical severity scores, including the Sequential Organ Failure Assessment (SOFA) score [Vincent et al., 1996], APACHE II [Knaus et al., 1985], and SAPS II [Le Gall et al., 1993]. These scores serve as clinically validated baselines for ICU mortality and length of stay prediction."

**Results Section:**
- Tabelle: Vergleich ML-Modelle vs. Clinical Scores
- Metriken: AUROC, AUPRC, Sensitivität, Spezifität

## Schnellstart

1. Pfade setzen → `config.py`
2. SOFA berechnen → `python scripts/baseline_models/calculate_sofa.py`
3. Ergebnisse nutzen → `outputs/baseline_models/sofa/sofa_scores.csv`

