# Troubleshooting Guide - Bekannte Probleme und Fixes

Dieses Dokument sammelt alle Probleme, die während der Entwicklung aufgetreten sind, und deren Lösungen. Dies hilft bei zukünftigen Modellen und Sessions.

---

## 🔧 Server-Deployment Probleme

### Problem: DOS-Zeilenumbrüche in SLURM-Script
**Symptom:**
```
sbatch: error: Batch script contains DOS line breaks (\r\n)
```

**Lösung:**
```bash
# Auf dem Server
sed -i 's/\r$//' scripts/cluster/train_cnn_scratch.sbatch
```

**Ursache:** Windows erstellt Dateien mit `\r\n`, Linux erwartet `\n`.

**Fix:** Immer nach dem Übertragen von Scripts auf Server konvertieren.

---

### Problem: Keine Partition angegeben
**Symptom:**
```
sbatch: error: Batch job submission failed: No partition specified
```

**Lösung:**
Im SLURM-Script hinzufügen:
```bash
#SBATCH --partition=gpu_a100_il  # Für GPU-Training
```

**Verfügbare Partitionen auf bwUniCluster 3.0:**
- `gpu_a100_il` - Für GPU-Training (A100)
- `gpu_h100_il` - Für GPU-Training (H100)
- `dev_gpu_a100_il` - Für kurze Tests (max 30 Min)

---

### Problem: Module-Namen falsch
**Symptom:**
```
Lmod has detected the following error: The following module(s) are unknown: "python/3.9"
```

**Lösung:**
Verfügbare Module prüfen und korrekte Namen verwenden:
```bash
module avail python
module avail cuda
```

**Korrekte Module-Namen (Stand: Jan 2026):**
```bash
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8
# cuDNN nicht verfügbar oder nicht nötig
```

---

## 🔧 Code-Probleme

### Problem: Learning Rate als String statt Float
**Symptom:**
```
TypeError: '<=' not supported between instances of 'str' and 'float'
```

**Lösung:**
In `src/training/trainer.py`:
```python
lr = opt_config.get("lr", 5e-4)
# Ensure lr is a float (YAML might load scientific notation as string)
if isinstance(lr, str):
    lr = float(lr)
```

**Ursache:** YAML lädt manchmal wissenschaftliche Notation (`5e-4`) als String.

**Fix:** Immer Float-Konvertierung für numerische Werte aus Config.

---

### Problem: Scheduler min_lr als String
**Symptom:**
```
TypeError: '>' not supported between instances of 'str' and 'float'
```

**Lösung:**
In `src/training/trainer.py`:
```python
min_lr = sched_config.get("min_lr", 1e-6)
# Ensure min_lr is a float
if isinstance(min_lr, str):
    min_lr = float(min_lr)
```

**Fix:** Gleiche Lösung wie bei Learning Rate.

---

### Problem: YAML-Syntaxfehler in Augmentation-Config
**Symptom:**
```
yaml.scanner.ScannerError: mapping values are not allowed here
  in "configs/baseline.yaml", line 43, column 16
```

**Ursache:**
Falsche Einrückung in YAML. Parameter müssen auf derselben Ebene stehen:
```yaml
# FALSCH:
augmentation:
  gaussian_noise: true
    noise_std: 0.03  # ❌ Falsche Einrückung

# RICHTIG:
augmentation:
  gaussian_noise: true
  noise_std: 0.03  # ✅ Gleiche Einrückung
```

**Lösung:**
Alle Parameter in der `augmentation`-Sektion müssen auf derselben Einrückungsebene stehen (2 Leerzeichen nach `augmentation:`).

**Fix:** YAML-Syntax korrigiert, alle Parameter auf gleicher Ebene.

---

### Problem: .npy Dateien werden nicht geladen
**Symptom:**
```
RuntimeError: No demo ECG records found in /path/to/preprocessed_24h_1
```

**Lösung:**
Neue Funktionen in `src/data/ecg/ecg_loader.py`:
- `build_npy_index()` - Findet `.npy` Dateien
- `ECGNPYDataset` - Lädt `.npy` Dateien

**Automatische Erkennung:**
In `src/data/ecg/dataloader_factory.py`:
```python
# Try .npy first, fall back to .hea/.dat
try:
    records = build_npy_index(data_dir=data_dir)
except (FileNotFoundError, RuntimeError):
    records = build_demo_index(data_dir=data_dir)
```

**Fix:** Code unterstützt jetzt beide Formate automatisch.

---

### Problem: Timestamps fehlen in .npy Dateien
**Symptom:**
```
Class distribution: {0: 0, 1: 0, ...}  # Alle Labels = 0
```

**Lösung:**
Fallback-Mechanismus in `src/data/ecg/ecg_dataset.py`:
```python
# Fallback for .npy files without timestamps: use first ICU stay's intime
if ecg_time is None:
    subject_stays = self.icu_mapper.icustays_df[
        self.icu_mapper.icustays_df['subject_id'] == subject_id
    ]
    if len(subject_stays) > 0:
        ecg_time = pd.to_datetime(subject_stays.iloc[0]['intime'])
```

**Fix:** Labels werden jetzt auch ohne Timestamps generiert (verwendet intime des ersten ICU Stays).

---

### Problem: PyTorch DataLoader kann None-Werte nicht collaten
**Symptom:**
```
TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'NoneType'>
```

**Lösung:**
In `src/data/ecg/ecg_dataset.py`:
```python
# Remove None values from meta (PyTorch DataLoader can't collate None)
meta_clean = {k: v for k, v in meta.items() if v is not None}
return {
    "signal": signal,
    "label": label,
    "meta": meta_clean,
}
```

**Fix:** None-Werte werden vor dem Return entfernt.

---

### Problem: Meta als Dictionary statt Liste
**Symptom:**
```
KeyError: 0
# In train_loop.py: meta[i] schlägt fehl
```

**Lösung:**
Custom Collate-Function in `src/data/ecg/dataloader_factory.py`:
```python
def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function that preserves meta as a list of dicts."""
    signals = torch.stack([item["signal"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    meta = [item["meta"] for item in batch]  # Keep as list
    return {"signal": signals, "label": labels, "meta": meta}
```

**Verwendung:**
```python
DataLoader(..., collate_fn=custom_collate_fn)
```

**Fix:** Alle DataLoaders verwenden jetzt `custom_collate_fn`.

---

### Problem: NaN Loss während Training
**Symptom:**
```
Train Loss: nan, Train Acc: 0.1623
Warning: NaN/Inf loss detected! Skipping batch.
Signal stats: min=nan, max=nan, mean=nan, std=nan
```

**Lösung:**
NaN-Check in `src/training/train_loop.py`:
```python
# Check for NaN/Inf in loss
if torch.isnan(loss) or torch.isinf(loss):
    print(f"Warning: NaN/Inf loss detected! Skipping batch.")
    # Debug-Informationen ausgeben
    continue
```

**Mögliche Ursachen:**
- Nicht normalisierte Daten
- Zu große Learning Rate
- Gradient Explosion
- Ungültige Labels
- **Data Augmentation produziert NaN** (siehe nächster Abschnitt)

**Fix:** NaN-Batches werden übersprungen und Warnung ausgegeben.

---

### Problem: Data Augmentation produziert NaN-Werte
**Symptom:**
```
Warning: NaN/Inf loss detected! Skipping batch.
Signal stats: min=nan, max=nan, mean=nan, std=nan
Logits stats: min=nan, max=nan, mean=nan
```
Alle Batches werden übersprungen, Training läuft durch ohne zu lernen.

**Ursache:**
Data Augmentation (besonders `GaussianNoise`) produziert NaN wenn:
- `signal_std = 0` (konstantes Signal) → `noise_std * 0 = NaN`
- `signal_std = NaN` (Signal enthält bereits NaN)
- Eingabesignale enthalten bereits NaN/Inf

**Lösung:**
1. **NaN-Check nach Laden der Daten** in `src/data/ecg/ecg_dataset.py`:
```python
# Check for NaN/Inf in loaded data
if torch.isnan(signal).any() or torch.isinf(signal).any():
    # Replace NaN/Inf with zeros (fallback)
    signal = torch.where(torch.isnan(signal) | torch.isinf(signal), torch.tensor(0.0), signal)
```

2. **NaN-Check in allen Augmentation-Methoden** in `src/data/augmentation/ecg_augmentation.py`:
```python
# Check for NaN/Inf in input
if torch.isnan(signal).any() or torch.isinf(signal).any():
    return signal  # Return unchanged if input is invalid

# In GaussianNoise: Schutz vor std=0 oder NaN
signal_std = torch.clamp(signal_std, min=1e-6)
signal_std = torch.where(torch.isnan(signal_std), torch.tensor(1e-6, device=signal.device), signal_std)

# Check for NaN/Inf in output
if torch.isnan(result).any() or torch.isinf(result).any():
    return signal  # Return original if augmentation produced NaN
```

**Fix:** Alle Augmentation-Methoden prüfen jetzt auf NaN/Inf und geben bei Problemen das Original-Signal zurück.

---

### Problem: Class Weights auf CPU statt CUDA (Training)
**Symptom:**
```
RuntimeError: Expected all tensors to be on the same device, but got weight is on cpu, different from other tensors on cuda:0
```

**Ursache:**
Class Weights werden in `get_loss()` auf CPU erstellt, aber das Modell läuft auf CUDA.

**Lösung:**
In `src/training/trainer.py` nach dem Erstellen der Loss-Funktion:
```python
# Setup loss
self.criterion = get_loss(config)

# Move loss weights to device if they exist
if hasattr(self.criterion, 'weight') and self.criterion.weight is not None:
    self.criterion.weight = self.criterion.weight.to(self.device)
```

**Fix:** Weights werden automatisch auf das Device des Modells verschoben.

---

### Problem: Class Weights auf CPU in Test-Evaluation
**Symptom:**
```
RuntimeError: Expected all tensors to be on the same device, but got weight is on cpu, different from other tensors on cuda:0
# Fehler tritt in evaluate_with_detailed_metrics() auf
```

**Ursache:**
In der Test-Evaluation wird eine neue Loss-Funktion erstellt (`criterion = get_loss(config)`), die Weights auf CPU hat.

**Lösung:**
In `scripts/training/cnn_from_scratch/icu_24h/train_cnn_scratch_24h_weighted.py` (und `train_cnn_scratch.py`):
```python
# FALSCH:
criterion = get_loss(config)  # Neue Loss mit CPU-Weights
test_metrics = evaluate_with_detailed_metrics(..., criterion=criterion, ...)

# RICHTIG:
# Use the criterion from trainer (already has weights on correct device)
test_metrics = evaluate_with_detailed_metrics(..., criterion=trainer.criterion, ...)
```

**Fix:** Verwende `trainer.criterion` statt neue Loss-Funktion zu erstellen.

---

### Problem: evaluate_with_detailed_metrics fehlt auf Server
**Symptom:**
```
ImportError: cannot import name 'evaluate_with_detailed_metrics' from 'src.training.train_loop'
```

**Ursache:**
Die Funktion wurde lokal erstellt, aber nicht auf den Server übertragen.

**Lösung:**
```bash
# Datei auf Server übertragen
scp src/training/train_loop.py ka_zx9981@bwunicluster.scc.kit.edu:~/workspace/ma-thesis/MA-thesis-1/src/training/
```

**Fix:** Stelle sicher, dass alle geänderten Dateien auf den Server übertragen werden.

---

### Problem: sklearn Import-Fehler (trotz Installation)
**Symptom:**
```
ModuleNotFoundError: No module named 'sklearn'
# Obwohl sklearn bereits installiert ist
```

**Ursache:**
- venv wurde nicht aktiviert beim Test
- Oder veraltete `train_loop.py` ohne sklearn-Import

**Lösung:**
```bash
# Prüfe Installation
source venv/bin/activate
pip install scikit-learn
python3 -c "from sklearn.metrics import confusion_matrix; print('OK')"
```

**Fix:** Stelle sicher, dass venv aktiviert ist und `train_loop.py` aktuell ist.

---

## 📋 Checkliste für neue Modelle

### Vor dem Training:
- [ ] SLURM-Script: Partition gesetzt (`gpu_a100_il`)
- [ ] SLURM-Script: Module-Namen korrekt (`devel/python/...`)
- [ ] SLURM-Script: ICUSTAYS_PATH gesetzt (absoluter Pfad)
- [ ] SLURM-Script: Zeilenumbrüche konvertiert (`sed -i 's/\r$//'`)
- [ ] Config: `data_dir` auf absoluten Pfad gesetzt
- [ ] Config: Numerische Werte (lr, min_lr, etc.) als Float geparst

### Code-Checks:
- [ ] Custom Collate-Function verwendet (`custom_collate_fn`)
- [ ] None-Werte aus meta entfernt
- [ ] Float-Konvertierung für Config-Werte
- [ ] NaN-Checks im Training-Loop
- [ ] NaN-Checks in Data Augmentation (wenn aktiviert)
- [ ] NaN-Check nach Laden der .npy Dateien
- [ ] Class Weights auf Device verschoben (wenn weighted_ce verwendet)
- [ ] Test-Evaluation verwendet `trainer.criterion` (nicht neue Loss-Funktion)
- [ ] Alle geänderten Dateien auf Server übertragen

---

## 🔍 Debug-Befehle

### Auf Server prüfen:
```bash
# Job-Status
squeue -u ka_zx9981
sacct -u ka_zx9981 --starttime=today

# Logs prüfen
tail -100 outputs/logs/slurm_*.out
cat outputs/logs/slurm_*.err

# Code prüfen
python -c "from src.data.ecg import create_dataloaders; print('OK')"
```

### Daten prüfen:
```bash
# Struktur prüfen
ls preprocessed_24h_1/p1000/ | head -5
find preprocessed_24h_1/p1000 -type f | head -5

# Labels prüfen
python -c "
from src.data.ecg import create_dataloaders
# ... Test-Code
"
```

---

## 📝 Wichtige Pfade (Server)

- Projekt: `/home/ka/ka_aifb/ka_zx9981/workspace/ma-thesis/MA-thesis-1`
- Daten: `/home/ka/ka_aifb/ka_zx9981/workspace/ma-thesis/MA-thesis-1/preprocessed_24h_1`
- icustays.csv: `/home/ka/ka_aifb/ka_zx9981/workspace/ma-thesis/MA-thesis-1/icustays.csv`
- E-Mail: `zx9981@partner.kit.edu`

---

## 🎯 Für zukünftige Modelle

**Diese Fixes sind bereits implementiert und funktionieren für alle Modelle:**
- ✅ `.npy` Support
- ✅ Custom Collate-Function
- ✅ Float-Konvertierung für Config
- ✅ NaN-Checks im Training-Loop
- ✅ NaN-Checks in Data Augmentation
- ✅ NaN-Check nach Laden der Daten
- ✅ Timestamp-Fallback
- ✅ Class Weights automatisch auf Device verschoben
- ✅ Test-Evaluation verwendet trainer.criterion

**Bei neuen Modellen:**
1. Gleiche DataLoader-Factory verwenden (bereits fix)
2. Gleiche Trainer-Klasse verwenden (bereits fix)
3. Nur Model-Architektur ändern
4. Config anpassen (Pfade, Hyperparameter)

---

## 📚 Nützliche Links

- [bwUniCluster 3.0 Wiki](https://wiki.bwhpc.de/e/BwUniCluster3.0)
- [Running Jobs](https://wiki.bwhpc.de/e/BwUniCluster3.0/Running_Jobs)
- [SLURM Documentation](https://slurm.schedmd.com/documentation.html)

---

**Letzte Aktualisierung:** 2026-01-07 (Class Weights Device-Fix, Test-Evaluation Fix)
**Server:** bwUniCluster 3.0
**Python:** 3.12.3 (via `devel/python/3.12.3-gnu-14.2`)
**CUDA:** 12.8 (via `devel/cuda/12.8`)

