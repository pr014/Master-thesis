# Model Architecture Design - Methodisches Konstrukt

## Übersicht

Dieses Dokument beschreibt die methodische Struktur für den Aufbau von Deep-Learning-Modellen für die ECG-Klassifikation. Die Architekturen umfassen:
1. **CNN** (Convolutional Neural Network)
2. **LSTM** (Long Short-Term Memory)
3. **Hybrid CNN-LSTM**
4. **Transformer**
5. **Foundation Model** (optional)

## Prinzipien des Designs

### 1. Modularität
- **Getrennte Verantwortlichkeiten**: Jede Komponente hat eine klar definierte Aufgabe
- **Wiederverwendbarkeit**: Gemeinsame Komponenten werden von allen Modellen genutzt
- **Erweiterbarkeit**: Neue Modelle können einfach hinzugefügt werden

### 2. Shared Components (Gemeinsame Komponenten)
Viele Komponenten werden von allen Modellen geteilt:
- **Dataloader**: Einheitliches Interface für Dateneingabe
- **Preprocessing**: Standardisierte Signalvorverarbeitung
- **Augmentation**: Gemeinsame Augmentationsstrategien
- **Training Loop**: Gemeinsame Trainingsinfrastruktur
- **Evaluation**: Einheitliche Metriken und Logging
- **Configuration**: YAML-basierte Konfiguration

### 3. Model-Specific Components
Jedes Modell hat eigene spezifische Komponenten:
- **Model Architecture**: Modell-spezifische Architektur
- **Input Format**: Falls abweichend vom Standard
- **Loss Function**: Falls modell-spezifisch
- **Optimizer Settings**: Falls unterschiedlich

---

## Verzeichnisstruktur

```
src/
├── data/
│   ├── __init__.py
│   ├── ecg_loader.py              # Basis ECG-Loader (bereits vorhanden)
│   ├── ecg_dataset.py             # PyTorch Dataset-Wrapper
│   ├── dataloader_factory.py      # Factory für DataLoader-Erstellung
│   └── preprocessing/
│       ├── __init__.py
│       ├── signal_processing.py   # Filterung, Normalisierung, etc.
│       ├── augmentation.py        # Data Augmentation (zeitlich, frequenz-basiert)
│       └── transforms.py          # PyTorch Transforms
│
├── models/
│   ├── __init__.py
│   ├── base_model.py              # Basisklasse für alle Modelle
│   ├── cnn/
│   │   ├── __init__.py
│   │   ├── model.py               # CNN-Architektur
│   │   └── config.yaml            # CNN-spezifische Konfiguration
│   ├── lstm/
│   │   ├── __init__.py
│   │   ├── model.py               # LSTM-Architektur
│   │   └── config.yaml
│   ├── hybrid_cnn_lstm/
│   │   ├── __init__.py
│   │   ├── model.py               # Hybrid-Architektur
│   │   └── config.yaml
│   ├── transformer/
│   │   ├── __init__.py
│   │   ├── model.py               # Transformer-Architektur
│   │   └── config.yaml
│   └── foundation/
│       ├── __init__.py
│       ├── model.py               # Foundation Model (z.B. pre-trained)
│       └── config.yaml
│
├── training/
│   ├── __init__.py
│   ├── trainer.py                 # Basis Trainer-Klasse
│   ├── train_loop.py              # Gemeinsamer Training Loop
│   ├── callbacks.py               # Callbacks (EarlyStopping, Checkpointing, etc.)
│   └── losses.py                  # Loss-Funktionen
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                 # Metriken (Accuracy, F1, AUC, etc.)
│   ├── evaluator.py               # Evaluator-Klasse
│   └── visualization.py           # Visualisierungen für Ergebnisse
│
└── utils/
    ├── __init__.py
    ├── config_loader.py           # YAML-Konfigurationslader
    ├── logger.py                  # Logging-Utility
    └── device.py                  # GPU/CPU-Handling

configs/
├── base/
│   └── default.yaml               # Basis-Konfiguration für alle Modelle
├── visualization/
│   └── default_paths.yaml         # Pfade für Visualisierungs-Skripte
├── model/
│   ├── cnn.yaml
│   ├── lstm.yaml
│   ├── hybrid_cnn_lstm.yaml
│   ├── transformer.yaml
│   └── foundation.yaml
└── experiment/
    └── [experiment_name].yaml     # Experiment-spezifische Konfigurationen
```

---

## Komponenten-Details

### 1. Data Layer (Gemeinsam)

#### 1.1 ECG Dataset (`src/data/ecg_dataset.py`)
**Zweck**: PyTorch Dataset-Klasse, die die vorhandene `ECGDemoDataset` erweitert

**Verantwortlichkeiten**:
- Lädt ECG-Signale aus WFDB-Dateien
- Wendet Preprocessing an
- Wendet Augmentation an (nur im Training)
- Gibt einheitliches Format zurück: `{"signal": Tensor, "label": Tensor, "meta": Dict}`

**Interface**:
```python
class ECGDataset(Dataset):
    def __init__(
        self,
        records: List[Dict],
        labels: Optional[Dict] = None,
        preprocess: Optional[Callable] = None,
        augmentation: Optional[Callable] = None,
        split: str = "train"  # "train", "val", "test"
    ):
        ...
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        ...
```

#### 1.2 Preprocessing (`src/data/preprocessing/signal_processing.py`)
**Zweck**: Standardisierte Signalvorverarbeitung

**Funktionen**:
- **Normalisierung**: Z-Score Normalisierung, Min-Max Normalisierung
- **Filterung**: Bandpass-Filter, Notch-Filter (z.B. 50/60 Hz), Baseline-Wandering-Reduktion
- **Resampling**: Einheitliche Sampling-Rate (z.B. 500 Hz)
- **Windowing**: Fixe Fensterlänge (z.B. 10 Sekunden)
- **Lead Selection**: Auswahl relevanter Leads (12-Lead, einzelne Leads, etc.)

**Interface**:
```python
def preprocess_ecg(
    signal: np.ndarray,
    fs: float,
    target_fs: float = 500.0,
    normalize: str = "zscore",  # "zscore", "minmax", None
    filter_type: str = "bandpass",  # "bandpass", "notch", None
    remove_baseline: bool = True
) -> np.ndarray:
    ...
```

#### 1.3 Data Augmentation (`src/data/preprocessing/augmentation.py`)
**Zweck**: Data Augmentation für bessere Generalisierung

**Augmentationen**:
- **Zeitbereich**: Time Warping, Time Shift, Scaling, Gaussian Noise
- **Frequenzbereich**: Frequency Masking, Mixup
- **Lead-spezifisch**: Lead Dropout, Lead Permutation

**Interface**:
```python
class ECGAugmentation:
    def __init__(
        self,
        time_warp: bool = False,
        time_shift: bool = False,
        add_noise: bool = False,
        scale: bool = False,
        lead_dropout: bool = False,
        ...
    ):
        ...
    
    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        ...
```

#### 1.4 DataLoader Factory (`src/data/dataloader_factory.py`)
**Zweck**: Zentralisierte Erstellung von DataLoaders

**Verantwortlichkeiten**:
- Erstellt Train/Val/Test Splits
- Erstellt PyTorch DataLoaders mit korrekten Parametern
- Unterstützt unterschiedliche Batch-Sizes für verschiedene Modelle

**Interface**:
```python
def create_dataloaders(
    data_dir: Path,
    config: Dict,
    model_type: str = "cnn"
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns: train_loader, val_loader, test_loader
    """
    ...
```

---

### 2. Model Layer (Model-Spezifisch)

#### 2.1 Base Model (`src/models/base_model.py`)
**Zweck**: Abstrakte Basisklasse für alle Modelle

**Verantwortlichkeiten**:
- Definiert gemeinsames Interface (`forward()`, `predict()`)
- Gemeinsame Hilfsmethoden (Parameter-Counting, etc.)
- Enforced durch alle Modell-Implementierungen

**Interface**:
```python
class BaseECGModel(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        ...
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ECG signal tensor (B, C, T) or (B, T, C)
            B: Batch size
            C: Number of channels/leads
            T: Time steps
        Returns:
            logits: (B, num_classes)
        """
        raise NotImplementedError
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns class predictions"""
        ...
```

#### 2.2 CNN Model (`src/models/cnn/model.py`)
**Architektur-Überlegungen**:
- **Input Format**: (B, C, T) - Zeitreihe als 1D/2D Convolutions
- **Layers**: Conv1D/Conv2D → BatchNorm → ReLU → MaxPool → Dropout
- **Feature Extraction**: Mehrere Convolutional Blocks
- **Classification Head**: Global Average Pooling → Dense Layers → Output

**Typische Architektur**:
```
Input (B, 12, 5000)  # 12 leads, 10s @ 500Hz
  ↓
Conv1D Blocks (tiefe Feature-Extraktion)
  ↓
Global Average Pooling
  ↓
Dense Layers
  ↓
Output (B, num_classes)
```

#### 2.3 LSTM Model (`src/models/lstm/model.py`)
**Architektur-Überlegungen**:
- **Input Format**: (B, T, C) - Sequenz-basiert
- **Layers**: LSTM/BiLSTM → Dropout → Dense → Output
- **Varianten**: Single-Layer vs. Multi-Layer, Bidirectional

**Typische Architektur**:
```
Input (B, 5000, 12)  # Time-first für LSTM
  ↓
LSTM/BiLSTM Layers
  ↓
Last Hidden State oder Attention Pooling
  ↓
Dense Layers
  ↓
Output (B, num_classes)
```

#### 2.4 Hybrid CNN-LSTM (`src/models/hybrid_cnn_lstm/model.py`)
**Architektur-Überlegungen**:
- **Input Format**: (B, C, T) oder (B, T, C)
- **Zwei-Stufen**: CNN für Feature-Extraktion → LSTM für Sequenz-Modellierung
- **Fusion**: Verschiedene Fusion-Strategien (Concatenation, Attention)

**Typische Architektur**:
```
Input (B, 12, 5000)
  ↓
CNN Feature Extractor (B, 12, 5000) → (B, features, reduced_time)
  ↓
Reshape für LSTM (B, reduced_time, features)
  ↓
LSTM Layers
  ↓
Fusion + Dense
  ↓
Output (B, num_classes)
```

#### 2.5 Transformer Model (`src/models/transformer/model.py`)
**Architektur-Überlegungen**:
- **Input Format**: (B, T, C) oder (B, C, T) mit Patch Embedding
- **Components**: Patch Embedding → Positional Encoding → Transformer Blocks → Classification Head
- **Variants**: Vision Transformer (ViT)-Style, Time Series Transformer

**Typische Architektur**:
```
Input (B, 12, 5000)
  ↓
Patch Embedding (B, num_patches, embed_dim)
  ↓
+ Positional Encoding
  ↓
Transformer Encoder Blocks (Multi-Head Attention + FFN)
  ↓
CLS Token oder Mean Pooling
  ↓
Classification Head
  ↓
Output (B, num_classes)
```

#### 2.6 Foundation Model (`src/models/foundation/model.py`)
**Architektur-Überlegungen**:
- **Option 1**: Pre-trained Modell (z.B. aus HuggingFace, PhysioNet Challenge)
- **Option 2**: Self-Supervised Pre-training (z.B. Masked Autoencoder, Contrastive Learning)
- **Fine-tuning**: Transfer Learning auf spezifische Aufgabe

**Strategien**:
- **Feature Extractor**: Pre-trained Model als Feature-Extraktor + Task-spezifischer Head
- **Fine-tuning**: Vollständiges Fine-tuning aller Parameter
- **Partial Fine-tuning**: Nur bestimmte Layers fine-tunen

---

### 3. Training Layer (Gemeinsam)

#### 3.1 Trainer (`src/training/trainer.py`)
**Zweck**: Basisklasse für Training

**Verantwortlichkeiten**:
- Model Training
- Validation
- Checkpointing
- Logging (TensorBoard, Weights & Biases, etc.)
- Early Stopping

**Interface**:
```python
class BaseTrainer:
    def __init__(
        self,
        model: BaseECGModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device
    ):
        ...
    
    def train(self) -> Dict[str, List[float]]:
        """Returns training history"""
        ...
    
    def validate(self) -> Dict[str, float]:
        """Returns validation metrics"""
        ...
```

#### 3.2 Loss Functions (`src/training/losses.py`)
**Zweck**: Loss-Funktionen (gemeinsam, aber konfigurierbar pro Modell)

**Mögliche Losses**:
- Cross-Entropy Loss
- Focal Loss (für class imbalance)
- Weighted Cross-Entropy
- Label Smoothing

#### 3.3 Callbacks (`src/training/callbacks.py`)
**Zweck**: Callbacks für Training

**Callbacks**:
- EarlyStopping: Stoppt Training bei keinem Fortschritt
- ModelCheckpoint: Speichert beste Modelle
- LearningRateScheduler: Passt Learning Rate an
- TensorBoardLogger: Loggt Metriken

---

### 4. Evaluation Layer (Gemeinsam)

#### 4.1 Metrics (`src/evaluation/metrics.py`)
**Zweck**: Evaluationsmetriken

**Metriken**:
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC, AUC-PR
- Confusion Matrix
- Classification Report

#### 4.2 Evaluator (`src/evaluation/evaluator.py`)
**Zweck**: Zentrale Evaluationsklasse

**Verantwortlichkeiten**:
- Evaluiert Model auf Test-Set
- Berechnet alle Metriken
- Speichert Ergebnisse

---

### 5. Configuration Layer

#### 5.1 Konfigurationsstruktur
**Hierarchie**:
```
base/default.yaml          # Basis-Konfiguration
  ├── model/cnn.yaml       # Model-spezifische Overrides
  └── experiment/exp1.yaml # Experiment-spezifische Overrides
```

**Konfigurationsbereiche**:
- **Data**: Pfade, Preprocessing-Parameter, Augmentation
- **Model**: Architektur-Parameter (Layers, Hidden Units, etc.)
- **Training**: Optimizer, Learning Rate, Epochs, Batch Size
- **Evaluation**: Metriken, Thresholds
- **Logging**: Logging-Config, Checkpoint-Pfade

---

## Datenfluss

### Training Pipeline
```
1. Configuration laden (YAML)
   ↓
2. DataLoader erstellen (Factory)
   ├── Preprocessing anwenden
   ├── Augmentation (nur Training)
   └── Batch-Erstellung
   ↓
3. Modell instanziieren (Model Factory)
   ↓
4. Trainer erstellen
   ├── Loss Function
   ├── Optimizer
   ├── Scheduler
   └── Callbacks
   ↓
5. Training Loop
   ├── Forward Pass
   ├── Loss Berechnung
   ├── Backward Pass
   ├── Validation
   └── Checkpointing
   ↓
6. Evaluation auf Test-Set
   ↓
7. Ergebnisse speichern
```

### Inference Pipeline
```
1. Model laden (Checkpoint)
   ↓
2. ECG Signal laden
   ↓
3. Preprocessing (wie im Training)
   ↓
4. Forward Pass
   ↓
5. Postprocessing (Softmax, Argmax)
   ↓
6. Ergebnis zurückgeben
```

---

## Gemeinsame vs. Model-Spezifische Komponenten

### ✅ Gemeinsam (Shared)
- **Dataloader**: Einheitliches Interface für alle Modelle
- **Preprocessing**: Standardisierte Signalvorverarbeitung
- **Augmentation**: Gemeinsame Augmentationsstrategien
- **Training Loop**: Gemeinsame Infrastruktur
- **Evaluation**: Einheitliche Metriken
- **Configuration**: YAML-basiert, hierarchisch
- **Logging**: TensorBoard, etc.
- **Device Handling**: GPU/CPU

### 🔷 Model-Spezifisch
- **Architektur**: Jedes Modell hat eigene Architektur
- **Input Format**: 
  - CNN: (B, C, T) - Channel-first
  - LSTM: (B, T, C) - Time-first
  - Hybrid: Abhängig von Fusion-Strategie
  - Transformer: (B, T, C) mit Patches
- **Hyperparameter**: Learning Rate, Batch Size können unterschiedlich sein
- **Loss Function**: Kann modell-spezifisch sein (optional)
- **Optimizer Settings**: Kann unterschiedlich sein

---

## Implementierungsreihenfolge (Empfehlung)

### Phase 1: Foundation (Shared Components)
1. ✅ ECG Dataset erweitern (PyTorch Dataset)
2. ✅ Preprocessing-Modul
3. ✅ Augmentation-Modul
4. ✅ DataLoader Factory
5. ✅ Base Model Klasse
6. ✅ Configuration System

### Phase 2: Training Infrastructure
7. ✅ Trainer-Klasse
8. ✅ Loss Functions
9. ✅ Callbacks
10. ✅ Metrics

### Phase 3: Model Implementation
11. ✅ CNN Model (am einfachsten zu starten)
12. ✅ LSTM Model
13. ✅ Hybrid CNN-LSTM
14. ✅ Transformer Model
15. ✅ Foundation Model (optional)

### Phase 4: Evaluation & Integration
16. ✅ Evaluator
17. ✅ Experiment-Scripts
18. ✅ Inference-Pipeline

---

## Best Practices

### 1. Reproduzierbarkeit
- Random Seeds setzen (PyTorch, NumPy, Python)
- Deterministic Operations (wenn möglich)
- Configuration-Versioning

### 2. Experiment-Tracking
- Alle Konfigurationen speichern
- Logs für alle Experimente
- Model-Checkpoints mit Metadaten

### 3. Code-Organisation
- Klare Trennung zwischen Shared und Model-Specific Code
- Type Hints für bessere Dokumentation
- Docstrings für alle Funktionen/Klassen

### 4. Testing
- Unit Tests für Preprocessing
- Integration Tests für Training Pipeline
- Model-Spezifische Tests

---

## Offene Fragen / Design-Entscheidungen

1. **Input Format Standardisierung**:
   - Soll es einen einheitlichen Input-Format-Converter geben?
   - Oder akzeptiert jedes Modell sein bevorzugtes Format?

2. **Label Format**:
   - Single-Label oder Multi-Label Klassifikation?
   - Regression-Tasks?

3. **Multi-Task Learning**:
   - Sollen Modelle mehrere Tasks gleichzeitig lernen können?

4. **Ensemble Methods**:
   - Sollen Ensembles unterstützt werden?

5. **Model Zoo**:
   - Sollen pre-trained Modelle gespeichert/geteilt werden?

---

## Nächste Schritte

Nach Genehmigung dieses Designs:
1. Detaillierte Spezifikationen für jede Komponente
2. API-Design für Interfaces
3. Konfigurationsschemas definieren
4. Implementierung starten (Phase 1)

