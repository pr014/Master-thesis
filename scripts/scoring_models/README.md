# Baseline Models - Scripts

Scripts zur Berechnung klinischer Baseline-Scores.

## Verwendung

### SOFA Score berechnen

```bash
python scripts/baseline_models/calculate_sofa.py
```

**Output:**
- `outputs/baseline_models/sofa/sofa_scores.csv`
- `outputs/baseline_models/sofa/sofa_statistics.txt`

## Konfiguration

**Pfade setzen in:** `src/baseline_models/config.py`

```python
MIMIC_IV_BASE_PATH = "E:/MIMIC-IV"  # DEIN PFAD!
```

## Zukünftige Scripts

- `calculate_apache.py` - APACHE II Score
- `calculate_saps.py` - SAPS II Score
- `calculate_qsofa.py` - qSOFA Score
- `compare_all_scores.py` - Vergleich aller Scores

