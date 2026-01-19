# MIMIC-IV Dataset - Scan Ergebnisse

Automatischer Scan der MIMIC-IV Datenstruktur auf externer SSD.

## 📍 Pfad

```
D:\MA\physionet.org\files\mimic-iv\3.1\
```

## 📊 Verfügbare Daten

### ICU Ordner (9 Dateien, ~4.2 GB komprimiert)

| Datei | Größe | Beschreibung |
|-------|-------|--------------|
| `chartevents.csv.gz` | **3.3 GB** | ⭐ Vitals, Monitoring (HR, BP, SpO2, GCS, FiO2, etc.) |
| `inputevents.csv.gz` | 382 MB | Medikamente, Infusionen, Vasopressoren |
| `ingredientevents.csv.gz` | 297 MB | Medikamenten-Inhaltsstoffe |
| `datetimeevents.csv.gz` | 60 MB | Zeitstempel-Events |
| `outputevents.csv.gz` | 47 MB | Ausscheidungen (Urin, Drainage) |
| `procedureevents.csv.gz` | 22 MB | Prozeduren auf ICU |
| `icustays.csv.gz` | 3.2 MB | **ICU Aufenthalte** (Zeitstempel, LOS) |
| `d_items.csv.gz` | 0.06 MB | Item-Definitionen (Labels) |
| `caregiver.csv.gz` | 0.04 MB | Pflegepersonal |

### HOSP Ordner (22 Dateien, ~5.9 GB komprimiert)

| Datei | Größe | Beschreibung |
|-------|-------|--------------|
| `labevents.csv.gz` | **2.5 GB** | ⭐ Laborwerte (Bilirubin, Kreatinin, Blutbild, etc.) |
| `emar.csv.gz` | 773 MB | Electronic Medication Administration Record |
| `emar_detail.csv.gz` | 713 MB | EMAR Details |
| `poe.csv.gz` | 635 MB | Provider Order Entry |
| `prescriptions.csv.gz` | 578 MB | Verschreibungen |
| `pharmacy.csv.gz` | 501 MB | Pharmazie-Daten |
| `microbiologyevents.csv.gz` | 112 MB | Mikrobiologie (Kulturen, Antibiotika) |
| `omr.csv.gz` | 42 MB | Outpatient Medical Records |
| `diagnoses_icd.csv.gz` | 32 MB | ICD-Diagnosen |
| `admissions.csv.gz` | 19 MB | **Hospital Admissions** |
| `drgcodes.csv.gz` | 9.3 MB | DRG Codes |
| `services.csv.gz` | 8.2 MB | Klinische Services |
| `procedures_icd.csv.gz` | 7.4 MB | ICD-Prozeduren |
| `transfers.csv.gz` | 44 MB | Verlegungen |
| `patients.csv.gz` | 2.7 MB | **Patienten-Stammdaten** |
| `hcpcsevents.csv.gz` | 2.1 MB | HCPCS Events |
| `d_icd_diagnoses.csv.gz` | 0.84 MB | ICD-Diagnose-Definitionen |
| `d_icd_procedures.csv.gz` | 0.56 MB | ICD-Prozedur-Definitionen |
| `d_hcpcs.csv.gz` | 0.41 MB | HCPCS-Definitionen |
| `provider.csv.gz` | 0.12 MB | Provider-Daten |
| `d_labitems.csv.gz` | 0.01 MB | Lab-Item-Definitionen |

**Gesamt: 31 Dateien, ~10.1 GB komprimiert**

---

## 🎯 Für SOFA Score benötigt

### Minimal benötigte Dateien (9 von 31):

**ICU (5 Dateien):**
- ✅ `icustays.csv.gz` - ICU Zeitstempel
- ✅ `chartevents.csv.gz` - Vitals, GCS, MAP, FiO2
- ✅ `inputevents.csv.gz` - Vasopressoren
- ✅ `outputevents.csv.gz` - Urin
- ✅ `d_items.csv.gz` - Item-Labels

**HOSP (4 Dateien):**
- ✅ `labevents.csv.gz` - Bilirubin, Kreatinin, Thrombozyten
- ✅ `patients.csv.gz` - Patienten-Daten
- ✅ `admissions.csv.gz` - Hospital Admissions
- ✅ `d_labitems.csv.gz` - Lab-Item-Labels

**Größe für SOFA: ~6.3 GB komprimiert**

---

## ⚠️ Besonderheiten

### 1. Kompression
- **Alle Dateien sind GZIP-komprimiert** (.csv.gz)
- Pandas kann diese direkt lesen: `pd.read_csv("file.csv.gz")`
- Dekomprimiert sind sie ~2-3x größer

### 2. Sehr große Dateien
- **chartevents.csv.gz:** 3.3 GB komprimiert ≈ 15-20 GB unkomprimiert
- **labevents.csv.gz:** 2.5 GB komprimiert ≈ 10-15 GB unkomprimiert
- **→ Chunk-basiertes Laden ZWINGEND notwendig!**

### 3. Struktur
```
mimic-iv/3.1/
├── icu/          # Intensive Care Unit Daten
└── hosp/         # Hospital-wide Daten
```

---

## 🔧 Pipeline-Anpassungen

Die Pipeline wurde angepasst für:
- ✅ .csv.gz Dateien (statt .csv)
- ✅ Chunk-basiertes Laden für große Dateien
- ✅ Korrekte Pfade zur SSD

**Konfiguration:** `src/baseline_models/config.py`

---

## 📈 Erwartete Performance

Bei chunk_size=100000:

| Datei | Größe | Erwartete Ladezeit |
|-------|-------|-------------------|
| `icustays.csv.gz` | 3.2 MB | < 1 Sekunde |
| `chartevents.csv.gz` | 3.3 GB | **2-5 Minuten** |
| `labevents.csv.gz` | 2.5 GB | **1-3 Minuten** |
| `inputevents.csv.gz` | 382 MB | 10-30 Sekunden |
| `outputevents.csv.gz` | 47 MB | < 5 Sekunden |

**Gesamt für SOFA-Berechnung: ~5-10 Minuten** (abhängig von SSD-Geschwindigkeit)

---

**Scan durchgeführt:** 2025-12-05
**Status:** ✅ Alle benötigten Dateien vorhanden

