# Server Deployment Guide - bwUniCluster 3.0

## 📋 Ablaufplan: Code auf Server übertragen

### Phase 1: Vorbereitung (Lokal)

#### 1.1 Code-Status prüfen
- [ ] Alle Änderungen committed (falls Git verwendet wird)
- [ ] Code funktioniert lokal (optional: Smoke Test)
- [ ] Keine temporären Dateien oder Debug-Code enthalten

#### 1.2 Dateien für Übertragung vorbereiten
- [ ] `.gitignore` prüfen (sollte `__pycache__/`, `*.pyc`, `outputs/`, etc. ausschließen)
- [ ] Große Output-Dateien nicht übertragen (werden auf Server neu erstellt)
- [ ] Sensible Daten/API-Keys nicht enthalten

#### 1.3 Übertragungsmethode wählen
**Option A: Git (empfohlen für Code)**
- Vorteile: Versionierung, Updates einfach
- Nachteile: Große Daten müssen separat übertragen werden

**Option B: SCP/SFTP (für alles auf einmal)**
- Vorteile: Einfach, alles in einem Schritt
- Nachteile: Langsam bei Updates

**Option C: rsync (empfohlen für Updates)**
- Vorteile: Effizient, nur Änderungen
- Nachteile: Benötigt rsync auf Windows (WSL/Git Bash)

---

### Phase 2: Server-Verbindung herstellen

#### 2.1 SSH-Verbindung aufbauen
```bash
# Auf Windows (PowerShell, WSL, oder Git Bash)
ssh <dein-username>@bwunicluster.scc.kit.edu

# Oder falls andere Domain:
# ssh <dein-username>@login.bwunicluster.de
```

#### 2.2 Arbeitsverzeichnis erstellen
```bash
# Nach SSH-Login auf dem Server
cd ~  # oder cd /work/<dein-username> falls verfügbar
mkdir -p workspace/ma-thesis
cd workspace/ma-thesis
pwd  # Notiere dir den absoluten Pfad!
```

**WICHTIG:** Notiere dir den absoluten Pfad (z.B. `/home/<username>/workspace/ma-thesis`)

---

### Phase 3: Code übertragen

#### Option A: ZIP + SCP (empfohlen für erste Übertragung)
```powershell
# Auf deinem lokalen Rechner (PowerShell)
cd C:\Users\trist\MA-thesis-1

# Automatisch mit Script:
.\scripts\deployment\create_deployment_package.ps1
.\scripts\deployment\transfer_to_server.ps1 -Username <dein-username>

# Oder manuell:
# 1. ZIP erstellen
Compress-Archive -Path src,scripts,configs,requirements.txt,README.md,.gitignore `
    -DestinationPath MA-thesis-1-deployment.zip -CompressionLevel Optimal

# 2. Auf Server übertragen
scp MA-thesis-1-deployment.zip <username>@bwunicluster.scc.kit.edu:~/workspace/ma-thesis/

# 3. Auf Server entpacken (nach SSH-Login)
ssh <username>@bwunicluster.scc.kit.edu
cd ~/workspace/ma-thesis
unzip MA-thesis-1-deployment.zip -d MA-thesis-1
cd MA-thesis-1
```

**Vorteile**: Schnell, einfach, komprimiert, keine Git nötig  
**Siehe auch**: `DEPLOYMENT_TRANSFER.md` für Details

#### Option B: Git (empfohlen für Updates)
```bash
# Auf deinem lokalen Rechner (PowerShell/Terminal)
cd C:\Users\trist\MA-thesis-1

# Falls noch nicht initialisiert:
git init
git add .
git commit -m "Prepare for HPC deployment"

# Falls Remote noch nicht gesetzt:
git remote add origin <dein-git-repo-url>
git push -u origin main

# Auf dem Server:
cd ~/workspace/ma-thesis
git clone <dein-git-repo-url>
cd MA-thesis-1
```

#### Option C: SCP direkt (ohne ZIP)
```powershell
# Auf deinem lokalen Rechner (PowerShell)
# Stelle sicher, dass du im Projektverzeichnis bist
cd C:\Users\trist\MA-thesis-1

# Übertrage alles (außer .git falls vorhanden)
scp -r -o "StrictHostKeyChecking=no" `
    --exclude='.git' `
    --exclude='__pycache__' `
    --exclude='*.pyc' `
    --exclude='outputs/**' `
    . <dein-username>@bwunicluster.scc.kit.edu:~/workspace/ma-thesis/MA-thesis-1/
```

**Hinweis:** PowerShell SCP unterstützt `--exclude` nicht direkt. Besser: Git oder rsync verwenden.

#### Option D: rsync (mit WSL oder Git Bash)
```bash
# In WSL oder Git Bash auf Windows
cd /mnt/c/Users/trist/MA-thesis-1

rsync -avz --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='outputs/' \
    --exclude='.vscode/' \
    --exclude='.idea/' \
    ./ <dein-username>@bwunicluster.scc.kit.edu:~/workspace/ma-thesis/MA-thesis-1/
```

#### Option D: Manuell mit FileZilla/WinSCP
1. FileZilla oder WinSCP installieren
2. Verbindung zu `bwunicluster.scc.kit.edu` herstellen
3. Lokales Verzeichnis: `C:\Users\trist\MA-thesis-1`
4. Server-Verzeichnis: `~/workspace/ma-thesis/MA-thesis-1`
5. Dateien übertragen (ausschließen: `__pycache__`, `outputs/`, `.git`)

---

### Phase 4: Server-Setup

#### 4.1 Auf Server einloggen und Projekt prüfen
```bash
ssh <dein-username>@bwunicluster.scc.kit.edu
cd ~/workspace/ma-thesis/MA-thesis-1

# Prüfe ob alles da ist
ls -la
ls src/
ls scripts/training/
```

#### 4.2 Verfügbare Module prüfen
```bash
# Verfügbare Python-Versionen
module avail python

# Verfügbare CUDA-Versionen
module avail cuda

# Verfügbare cuDNN-Versionen
module avail cudnn

# Beispiel-Output notieren (z.B. python/3.9, cuda/11.8, cudnn/8.6)
```

#### 4.3 Python-Umgebung erstellen
```bash
# Module laden (anpassen nach verfügbaren Versionen)
module load python/3.9  # oder python/3.10, etc.
module load cuda/11.8    # anpassen
module load cudnn/8.6    # anpassen

# Virtual Environment erstellen
python -m venv venv

# Aktivieren
source venv/bin/activate

# pip upgraden
pip install --upgrade pip
```

#### 4.4 Dependencies installieren
```bash
# Im venv (sollte aktiviert sein)
cd ~/workspace/ma-thesis/MA-thesis-1
pip install -r requirements.txt

# Optional: PyTorch separat installieren (falls Version nicht passt)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 4.5 Datenpfade prüfen
```bash
# Prüfe wo deine preprocessed ECG-Daten liegen
# (sollten bereits auf dem Server sein)
ls /path/to/preprocessed/ecg/data  # Anpassen!

# Prüfe ob icustays.csv vorhanden ist
ls /path/to/icustays.csv  # Anpassen!
```

---

### Phase 5: Konfiguration anpassen

#### 5.1 Config-Dateien anpassen
```bash
# Öffne configs/baseline.yaml
nano configs/baseline.yaml
# oder
vim configs/baseline.yaml
```

**Anpassungen:**
- `data_dir`: Pfad zu preprocessed ECG-Daten auf Server
  ```yaml
  data:
    data_dir: "/path/to/preprocessed/ecg/data"  # ANPASSEN!
  ```

#### 5.2 SLURM-Script anpassen
```bash
nano scripts/cluster/train_cnn_scratch.sbatch
```

**Anpassungen:**
1. **Module-Load-Befehle aktivieren** (Zeilen 24-26):
   ```bash
   module load python/3.9      # Anpassen nach verfügbaren Modulen
   module load cuda/11.8       # Anpassen
   module load cudnn/8.6       # Anpassen
   ```

2. **ICUSTAYS_PATH setzen** (Zeile 38):
   ```bash
   export ICUSTAYS_PATH="/absoluter/pfad/zu/icustays.csv"
   ```

3. **Optional: GPU-Partition/Account** (falls nötig):
   ```bash
   #SBATCH --partition=gpu      # Falls spezielle Partition nötig
   #SBATCH --account=<account>   # Falls Account nötig
   ```

#### 5.3 Output-Verzeichnisse erstellen
```bash
mkdir -p outputs/logs
mkdir -p outputs/checkpoints
```

---

### Phase 6: Test-Run (optional, aber empfohlen)

#### 6.1 Kleiner Test lokal auf Server
```bash
# SSH auf Server
cd ~/workspace/ma-thesis/MA-thesis-1
source venv/bin/activate

# Environment-Variable setzen
export ICUSTAYS_PATH="/path/to/icustays.csv"

# Test ob Code lädt (ohne Training)
python -c "from src.models import CNNScratch; print('Import OK')"
```

#### 6.2 Smoke Test (falls vorhanden)
```bash
# Falls du einen Smoke Test hast
python test/smoke_test_pipeline.py
```

---

### Phase 7: Job einreichen

#### 7.1 SLURM-Job einreichen
```bash
cd ~/workspace/ma-thesis/MA-thesis-1
sbatch scripts/cluster/train_cnn_scratch.sbatch
```

**Output:** Du erhältst eine Job-ID (z.B. `123456`)

#### 7.2 Job-Status prüfen
```bash
# Job-Status anzeigen
squeue -u <dein-username>

# Oder spezifische Job-ID
squeue -j <job-id>

# Job-Details
scontrol show job <job-id>
```

#### 7.3 Logs überwachen
```bash
# Logs in Echtzeit anzeigen (während Job läuft)
tail -f outputs/logs/slurm_<job-id>.out

# Oder Fehler-Log
tail -f outputs/logs/slurm_<job-id>.err
```

---

### Phase 8: Nach dem Training

#### 8.1 Ergebnisse prüfen
```bash
# Checkpoints
ls -lh outputs/checkpoints/

# Logs
ls -lh outputs/logs/

# TensorBoard-Logs (falls aktiviert)
ls -lh outputs/logs/tensorboard/
```

#### 8.2 Ergebnisse herunterladen
```bash
# Auf deinem lokalen Rechner (PowerShell/WSL)
scp -r <dein-username>@bwunicluster.scc.kit.edu:~/workspace/ma-thesis/MA-thesis-1/outputs/checkpoints ./

# Oder mit rsync
rsync -avz <dein-username>@bwunicluster.scc.kit.edu:~/workspace/ma-thesis/MA-thesis-1/outputs/ ./outputs/
```

---

## 🔧 Troubleshooting

### Problem: Module nicht gefunden
```bash
# Verfügbare Module prüfen
module avail

# Module-Liste aktualisieren
module spider python
module spider cuda
```

### Problem: GPU nicht verfügbar
```bash
# Verfügbare GPU-Partitionen prüfen
sinfo | grep gpu

# GPU-Info im Job
nvidia-smi  # Funktioniert nur im laufenden Job
```

### Problem: Import-Fehler
```bash
# PYTHONPATH prüfen
echo $PYTHONPATH

# Im SLURM-Script sicherstellen:
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"
```

### Problem: Datenpfad nicht gefunden
```bash
# Absoluten Pfad prüfen
ls -la /path/to/data

# In Config verwenden
# Immer absolute Pfade verwenden!
```

---

## 📝 Checkliste vor Job-Einreichung

- [ ] Code auf Server übertragen
- [ ] Python-Umgebung erstellt und aktiviert
- [ ] Dependencies installiert (`pip install -r requirements.txt`)
- [ ] Module-Load-Befehle im SLURM-Script aktiviert
- [ ] `data_dir` in Config angepasst (absoluter Pfad)
- [ ] `ICUSTAYS_PATH` im SLURM-Script gesetzt (absoluter Pfad)
- [ ] Output-Verzeichnisse erstellt (`outputs/logs`, `outputs/checkpoints`)
- [ ] Test-Run erfolgreich (optional)
- [ ] SLURM-Script-Pfade korrekt
- [ ] Job eingereicht mit `sbatch`

---

## 🔗 Nützliche Links

- [bwUniCluster 3.0 Wiki](https://wiki.bwhpc.de/e/BwUniCluster3.0)
- [Getting Started Guide](https://wiki.bwhpc.de/e/BwUniCluster3.0/Getting_Started)
- [Running Jobs](https://wiki.bwhpc.de/e/BwUniCluster3.0/Running_Jobs)
- [SLURM Documentation](https://slurm.schedmd.com/documentation.html)

---

## 💡 Tipps

1. **Absolute Pfade verwenden**: Immer absolute Pfade in Configs und Scripts verwenden
2. **Module-Versionen notieren**: Notiere dir welche Module-Versionen funktionieren
3. **Logs regelmäßig prüfen**: Während des Trainings Logs überwachen
4. **Backup**: Wichtige Checkpoints regelmäßig herunterladen
5. **Resource-Anfragen**: Bei Problemen mit Memory/Time, SLURM-Parameter anpassen

