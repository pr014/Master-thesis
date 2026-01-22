# 🚀 Quick Start: Code auf bwUniCluster 3.0 übertragen

## Schnellübersicht (5-Minuten-Checkliste)

### ✅ Schritt 1: Lokale Vorbereitung (2 Min)
```powershell
# In PowerShell im Projektverzeichnis
cd C:\Users\trist\MA-thesis-1

# Prüfe ob alles committed (falls Git)
git status

# Notiere dir:
# - Projekt-Pfad: C:\Users\trist\MA-thesis-1
# - Username auf Server: <dein-username>
```

### ✅ Schritt 2: Server-Verbindung (1 Min)
```bash
# SSH-Verbindung
ssh <dein-username>@bwunicluster.scc.kit.edu

# Arbeitsverzeichnis erstellen
mkdir -p ~/workspace/ma-thesis
cd ~/workspace/ma-thesis
pwd  # Notiere diesen Pfad!
```

### ✅ Schritt 3: Code übertragen (2 Min)

**Option A: Git (empfohlen)**
```bash
# Auf Server
cd ~/workspace/ma-thesis
git clone <dein-repo-url>
cd MA-thesis-1
```

**Option B: SCP (PowerShell)**
```powershell
# Auf lokalem Rechner (PowerShell)
cd C:\Users\trist\MA-thesis-1
scp -r . <dein-username>@bwunicluster.scc.kit.edu:~/workspace/ma-thesis/MA-thesis-1/
```

**Option C: FileZilla/WinSCP**
- Verbindung: `bwunicluster.scc.kit.edu`
- Lokal: `C:\Users\trist\MA-thesis-1`
- Remote: `~/workspace/ma-thesis/MA-thesis-1`

### ✅ Schritt 4: Server-Setup (5 Min)
```bash
# Auf Server
cd ~/workspace/ma-thesis/MA-thesis-1

# Module prüfen
module avail python
module avail cuda

# Module laden (ANPASSEN nach verfügbaren Versionen!)
module load python/3.9
module load cuda/11.8
module load cudnn/8.6

# Virtual Environment
python -m venv venv
source venv/bin/activate

# Dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### ✅ Schritt 5: Konfiguration (3 Min)
```bash
# 1. Config anpassen
nano configs/baseline.yaml
# Ändere: data_dir: "/absoluter/pfad/zu/preprocessed/ecg/data"

# 2. SLURM-Script anpassen
nano scripts/cluster/train_cnn_scratch.sbatch
# Aktiviere Module-Load (Zeilen 24-26)
# Setze ICUSTAYS_PATH (Zeile 38)

# 3. Output-Verzeichnisse
mkdir -p outputs/logs outputs/checkpoints
```

### ✅ Schritt 6: Job starten (1 Min)
```bash
# Job einreichen
sbatch scripts/cluster/train_cnn_scratch.sbatch

# Status prüfen
squeue -u <dein-username>

# Logs anzeigen
tail -f outputs/logs/slurm_*.out
```

---

## 🔑 Wichtige Pfade (ANPASSEN!)

| Was | Pfad auf Server |
|-----|----------------|
| Projekt-Verzeichnis | `~/workspace/ma-thesis/MA-thesis-1` |
| Preprocessed ECG-Daten | `/path/to/preprocessed/ecg/data` ⚠️ |
| icustays.csv | `/path/to/icustays.csv` ⚠️ |
| Outputs | `~/workspace/ma-thesis/MA-thesis-1/outputs/` |

⚠️ **MUSS ANGEPASST WERDEN!**

---

## 📋 Minimal-Checkliste

- [ ] Code auf Server übertragen
- [ ] `venv` erstellt und aktiviert
- [ ] `pip install -r requirements.txt` erfolgreich
- [ ] `data_dir` in `configs/baseline.yaml` angepasst
- [ ] `ICUSTAYS_PATH` in `train_cnn_scratch.sbatch` gesetzt
- [ ] Module-Load-Befehle im SLURM-Script aktiviert
- [ ] `sbatch` erfolgreich ausgeführt

---

## 🆘 Bei Problemen

1. **Module nicht gefunden**: `module avail` → Verfügbare Versionen prüfen
2. **Import-Fehler**: `export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"`
3. **Datenpfad nicht gefunden**: Immer absolute Pfade verwenden!
4. **GPU nicht verfügbar**: `sinfo | grep gpu` → Partition prüfen

Siehe auch: `SERVER_DEPLOYMENT.md` für detaillierte Anleitung

