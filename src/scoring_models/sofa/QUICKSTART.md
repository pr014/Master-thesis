# SOFA Score - Quick Start Guide

## 🚀 Schnellstart in 5 Minuten

### 1. Installation

```bash
cd sofa_score
pip install -r requirements.txt
```

### 2. Konfiguration anpassen

Öffne `config.py` und setze deine Pfade:

```python
# Zeile 14-15
MIMIC_IV_BASE_PATH = "E:/MIMIC-IV"  # DEIN PFAD!
OUTPUT_PATH = "../outputs/sofa_scores"  # DEIN OUTPUT-PFAD!
```

### 3. SOFA Scores berechnen

```bash
python run_sofa_calculation.py
```

**Das war's!** 🎉

---

## 📋 Was passiert im Hintergrund?

Der Script:
1. ✅ Validiert deine MIMIC-IV Pfade
2. 📊 Lädt Daten aus 5 Tabellen (chartevents, labevents, inputevents, outputevents, icustays)
3. 🧮 Berechnet SOFA Scores (6 Komponenten + Total)
4. 🔍 Validiert Ergebnisse (Plausibilitäts-Checks)
5. 💾 Speichert 4 Output-Dateien

---

## 📁 Output-Dateien

Nach dem Lauf findest du:

```
outputs/sofa_scores/
├── sofa_scores.csv           # Nur Total-Scores (für ML-Modelle)
├── sofa_components.csv       # Alle 6 Komponenten einzeln
├── sofa_complete_data.csv    # Vollständig (mit allen Features)
└── sofa_statistics.txt       # Statistiken & Analyse
```

---

## 🎯 Verwendung für Thesis

### In deinen ML-Pipelines:

```python
import pandas as pd

# Lade SOFA Scores
sofa_df = pd.read_csv('outputs/sofa_scores/sofa_scores.csv')

# Merge mit deinen ECG-Daten
ecg_data = pd.read_csv('your_ecg_features.csv')
merged = ecg_data.merge(sofa_df, on=['subject_id', 'hadm_id'])

# Nutze SOFA als Baseline für Mortality-Prediction
from sklearn.metrics import roc_auc_score

# SOFA Baseline Performance
baseline_auroc = roc_auc_score(y_true, merged['sofa_total'])
print(f"SOFA Baseline AUROC: {baseline_auroc:.3f}")

# Dein ML-Modell
ml_auroc = roc_auc_score(y_true, your_model_predictions)
print(f"ML Model AUROC: {ml_auroc:.3f}")
print(f"Improvement: +{(ml_auroc - baseline_auroc)*100:.1f}%")
```

---

## 🧪 Testen mit Beispiel-Daten

Falls du erstmal testen willst:

```bash
# Beispiele ansehen
python example_usage.py
```

Oder in `run_sofa_calculation.py` (Zeile 54):
```python
# Teste mit nur 3 Patienten (schneller)
subject_ids = [10000019, 10000032, 10000033]
```

---

## ⚙️ Anpassungen

### Zeitfenster ändern

In `config.py`:
```python
SOFA_TIME_WINDOW_HOURS = 24  # Erste 24h (Standard)
# Ändern auf z.B.:
SOFA_TIME_WINDOW_HOURS = 48  # Erste 48h
```

### Nur bestimmte Patienten

In `run_sofa_calculation.py`:
```python
# Nur Patienten mit ICU-Aufenthalt >24h
icustays = icustays[icustays['los'] > 1.0]
```

---

## 🐛 Problemlösung

### "Dateien nicht gefunden"
→ Pfade in `config.py` prüfen!

### "Keine Daten geladen"
→ Prüfe ob MIMIC-IV Struktur korrekt ist:
```
MIMIC-IV/
├── hosp/
│   ├── patients.csv
│   ├── admissions.csv
│   └── labevents.csv
└── icu/
    ├── icustays.csv
    ├── chartevents.csv
    ├── inputevents.csv
    └── outputevents.csv
```

### "Langsam / Out of Memory"
→ In `config.py`:
```python
CHUNK_SIZE = 50000  # Reduzieren (Standard: 100000)
```

Oder: Nur Subset der Patienten laden

---

## 📚 Weiterführend

- **Vollständige Doku**: Siehe `README.md`
- **Beispiele**: Siehe `example_usage.py`
- **Code-Kommentare**: Alle Funktionen gut dokumentiert

---

## ✅ Checkliste für Thesis

- [ ] SOFA Scores berechnet
- [ ] Output-Dateien geprüft
- [ ] Mit ECG-Daten gemerged
- [ ] Baseline-Performance evaluiert (AUROC, AUPRC)
- [ ] In Methods-Section dokumentiert:
  - "SOFA scores were calculated using the worst values within the first 24 hours of ICU admission"
  - "Based on Vincent et al. (1996) definition"
- [ ] In Results: Vergleich SOFA vs. ML-Modelle
- [ ] In Discussion: Klinische Interpretation

---

**Viel Erfolg mit deiner Masterarbeit! 🚀**

