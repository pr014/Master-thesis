# 📦 Code-Übertragung auf Server - Anleitung

## Soll ich die Daten in eine ZIP packen?

### ✅ **JA, empfohlen für SCP!**

**Vorteile von ZIP:**
- ✅ **Schneller**: Weniger Overhead bei vielen kleinen Dateien (Code)
- ✅ **Kompression**: Reduziert Übertragungszeit (besonders bei Text-Dateien)
- ✅ **Einfacher**: Ein Befehl statt viele einzelne Dateien
- ✅ **Zuverlässiger**: Weniger Verbindungsprobleme

**Nachteile:**
- ⚠️ Komprimierung braucht Zeit (aber meist schneller gesamt)
- ⚠️ Muss auf Server entpackt werden

---

## 🚀 Schnellstart: ZIP-Methode

### Option 1: Automatisches Script (empfohlen)

```powershell
# 1. Package erstellen
cd C:\Users\trist\MA-thesis-1
.\scripts\deployment\create_deployment_package.ps1

# 2. Auf Server übertragen
.\scripts\deployment\transfer_to_server.ps1 -Username <dein-username>
```

### Option 2: Manuell

```powershell
# 1. ZIP erstellen (PowerShell)
cd C:\Users\trist\MA-thesis-1

# Erstelle ZIP mit wichtigen Dateien (ohne outputs, data, __pycache__)
Compress-Archive -Path src,scripts,configs,requirements.txt,README.md,.gitignore `
    -DestinationPath MA-thesis-1-deployment.zip `
    -CompressionLevel Optimal

# 2. Auf Server übertragen
scp MA-thesis-1-deployment.zip <username>@bwunicluster.scc.kit.edu:~/workspace/ma-thesis/

# 3. Auf Server entpacken (nach SSH-Login)
ssh <username>@bwunicluster.scc.kit.edu
cd ~/workspace/ma-thesis
unzip MA-thesis-1-deployment.zip -d MA-thesis-1
cd MA-thesis-1
```

---

## 📋 Was wird übertragen?

### ✅ Wird übertragen:
- `src/` - Alle Source-Dateien
- `scripts/` - Alle Scripts
- `configs/` - Konfigurationsdateien
- `requirements.txt` - Dependencies
- `README.md` - Dokumentation
- `.gitignore` - Git-Konfiguration
- `docs/` - Dokumentation (optional, ohne PDFs)

### ❌ Wird NICHT übertragen:
- `data/` - Große Daten (liegen bereits auf Server)
- `outputs/` - Generierte Dateien (werden neu erstellt)
- `__pycache__/` - Python Cache
- `*.pyc` - Kompilierte Python-Dateien
- `.vscode/`, `.idea/` - IDE-Einstellungen
- `*.pdf` - Große PDF-Dateien
- `.git/` - Git-Repository (falls vorhanden)

---

## 🔄 Alternative Methoden

### Option A: Git (wenn Repository vorhanden)
```bash
# Auf Server
git clone <dein-repo-url>
```
**Vorteil**: Einfache Updates, Versionierung  
**Nachteil**: Benötigt Git-Repository

### Option B: rsync (mit WSL/Git Bash)
```bash
# In WSL oder Git Bash
rsync -avz --exclude='outputs' --exclude='data' --exclude='__pycache__' \
    ./ <username>@bwunicluster.scc.kit.edu:~/workspace/ma-thesis/MA-thesis-1/
```
**Vorteil**: Nur Änderungen, sehr effizient  
**Nachteil**: Benötigt rsync auf Windows

### Option C: FileZilla/WinSCP (GUI)
- Verbindung: `bwunicluster.scc.kit.edu`
- Lokal: `C:\Users\trist\MA-thesis-1`
- Remote: `~/workspace/ma-thesis/MA-thesis-1`
- **Tipp**: ZIP erstellen und dann übertragen (schneller!)

---

## ⚡ Performance-Vergleich

| Methode | Geschwindigkeit | Einfachheit | Updates |
|---------|----------------|-------------|---------|
| **ZIP + SCP** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Git** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **rsync** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **SCP direkt** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **FileZilla** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

**Empfehlung für erste Übertragung**: ZIP + SCP  
**Empfehlung für Updates**: Git oder rsync

---

## 🎯 Empfohlener Workflow

### Erste Übertragung:
1. ✅ ZIP erstellen (automatisch oder manuell)
2. ✅ ZIP auf Server übertragen (`scp`)
3. ✅ Auf Server entpacken (`unzip`)
4. ✅ Setup durchführen (siehe `SERVER_DEPLOYMENT.md`)

### Spätere Updates:
1. ✅ **Git** (wenn Repository): `git pull` auf Server
2. ✅ **rsync** (wenn WSL verfügbar): Nur Änderungen übertragen
3. ✅ **ZIP** (falls nötig): Neu erstellen und übertragen

---

## 💡 Tipps

1. **ZIP-Größe prüfen**: Sollte < 50 MB sein (nur Code)
2. **Vor Übertragung testen**: ZIP lokal entpacken und prüfen
3. **Backup**: Alte Version auf Server behalten (umbenennen)
4. **Daten separat**: Preprocessed ECGs und `icustays.csv` liegen bereits auf Server

---

## 🆘 Troubleshooting

### Problem: ZIP zu groß
```powershell
# Prüfe was drin ist
Compress-Archive -Path src,scripts,configs -DestinationPath test.zip
# Prüfe Größe
(Get-Item test.zip).Length / 1MB
```

### Problem: SCP langsam
- Prüfe Internet-Verbindung
- Nutze `-C` Flag für Kompression: `scp -C file.zip ...`
- Nutze rsync statt SCP (falls verfügbar)

### Problem: Unzip fehlt auf Server
```bash
# Auf Server
module load unzip  # Falls verfügbar
# Oder
gunzip -c file.zip | tar -xvf -  # Alternative
```

---

## 📝 Checkliste

- [ ] ZIP-Datei erstellt (ohne große Dateien)
- [ ] ZIP-Größe < 50 MB (nur Code)
- [ ] ZIP auf Server übertragen
- [ ] ZIP auf Server entpackt
- [ ] Projekt-Struktur auf Server geprüft
- [ ] Setup gestartet (siehe `SERVER_DEPLOYMENT.md`)

