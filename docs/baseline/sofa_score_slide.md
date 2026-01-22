# SOFA Score Calculation

## What is the SOFA Score?

**SOFA** (Sequential Organ Failure Assessment) is a clinical score for assessing organ function in critically ill patients.

- **Range**: 0-24 points (higher = worse prognosis)
- **6 Organ Systems**: Each scored 0-4 points
- **Reference**: Vincent et al. (1996), Sepsis-3 Consensus (2016)

---

## SOFA Score Components

| Organ System | Parameter | Scoring (0-4 points) |
|-------------|-----------|---------------------|
| **Respiration** | PaO₂/FiO₂ Ratio | ≥400: 0, <400: 1, <300: 2, <200 (ventilated): 3, <100 (ventilated): 4 |
| **Coagulation** | Platelets (×10³/μl) | ≥150: 0, <150: 1, <100: 2, <50: 3, <20: 4 |
| **Liver** | Bilirubin (mg/dl) | <1.2: 0, 1.2-1.9: 1, 2.0-5.9: 2, 6.0-11.9: 3, ≥12.0: 4 |
| **Cardiovascular** | MAP + Vasopressors | MAP ≥70: 0, <70: 1, Vasopressors: 2-4 (depending on dose) |
| **CNS** | Glasgow Coma Scale | 15: 0, 13-14: 1, 10-12: 2, 6-9: 3, <6: 4 |
| **Renal** | Creatinine (mg/dl) or Urine (ml/24h) | <1.2: 0, 1.2-1.9: 1, 2.0-3.4: 2, 3.5-4.9 or Urine <500: 3, ≥5.0 or Urine <200: 4 |

**Total SOFA** = Sum of all 6 components

---

## Implementation for MIMIC-IV

### Data Sources

| MIMIC-IV Table | Loaded Parameters |
|---------------|------------------|
| `icustays` | ICU stays (basis for time window) |
| `labevents` | PaO₂, Bilirubin, Creatinine, Platelets |
| `chartevents` | MAP, GCS, FiO₂, Mechanical ventilation, Weight |
| `inputevents` | Vasopressors (Dopamine, Norepinephrine, Epinephrine, Dobutamine) |
| `outputevents` | Urine output (24h sum) |

### Methodology

**Time Window**: First 24 hours after ICU admission

**Aggregation**: Worst-case values per patient
- **Minimum** for: PaO₂, Platelets, MAP, GCS, Urine output
- **Maximum** for: Bilirubin, Creatinine, Vasopressor doses, FiO₂

**Missing Values**: Conservative = 0 points (conservative approach)

**Vasopressor Dosing**: Normalized to μg/kg/min
- If weight missing: Default = 80 kg

**PaO₂/FiO₂ Ratio**: 
- Calculated from PaO₂ (labevents) and FiO₂ (chartevents)
- Capped at 600 (for unrealistic values)

### Technical Details

- **Chunk-based Loading**: Large files (chartevents: 3.3 GB, labevents: 2.5 GB) processed in chunks of 100,000 rows
- **Item ID Mappings**: Supports Metavision and CareVue systems (different itemids)
- **Validation**: Unit tests against known clinical examples

---

## Results (Example)

**Statistics** (23 patients):
- **Mean**: 5.52 ± 3.79
- **Median**: 7.0
- **Range**: 1 - 11

**Component Distribution**:
- Respiration: 1.61 ± 1.80
- Cardiovascular: 1.87 ± 1.06
- Renal: 1.52 ± 1.62
- Coagulation: 0.43 ± 0.66
- Liver: 0.09 ± 0.42
- CNS: 0.00 ± 0.00

---

## Constraints & Limitations

✅ **Strengths**:
- Complete implementation of all 6 SOFA components
- Validated itemid mappings for MIMIC-IV
- Robust handling of missing values
- Efficient processing of large datasets

⚠️ **Limitations**:
- Missing values are conservatively scored as 0 (may underestimate score)
- Weight for vasopressor dosing: Default 80 kg if not available
- Time window fixed at 24h (no dynamic adjustment)
- PaO₂/FiO₂ ratio requires synchronization of labevents and chartevents

---

## Code Structure

```
src/baseline_models/sofa/
├── calculator.py        # SOFA calculation logic (6 components)
├── data_loader.py       # MIMIC-IV data loading & aggregation
└── itemid_mappings.py   # MIMIC-IV itemid mappings

scripts/baseline_models/
└── calculate_sofa.py   # Main script (orchestration)
```

