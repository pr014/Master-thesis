"""
MIMIC-IV Data Pipeline

Wiederverwendbare, allgemeingültige Pipeline für MIMIC-IV Daten.
Kann für verschiedene Zwecke genutzt werden:
- Baseline Models (SOFA, APACHE, etc.)
- Feature Engineering
- ML-Training
- Datenanalyse

Author: MA Thesis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Union
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# MIMIC-IV PIPELINE KLASSE
# =============================================================================

class MimicIVPipeline:
    """
    Allgemeingültige Pipeline für MIMIC-IV Datenverarbeitung.
    
    Features:
    - Automatische Pfad-Erkennung
    - Chunk-basiertes Laden großer Dateien
    - Zeitfenster-Filterung
    - ICU Stay Verknüpfung
    - Progress Tracking
    
    Usage:
        pipeline = MimicIVPipeline(base_path="D:/MA/physionet.org/files/mimic-iv/3.1")
        
        # Lade ICU Stays
        icustays = pipeline.load_icustays()
        
        # Lade Vitals für erste 24h
        vitals = pipeline.load_chartevents(
            stay_ids=icustays['stay_id'],
            itemids=[220045, 220050],
            time_window_hours=24
        )
    """
    
    def __init__(self, base_path: Union[str, Path]):
        """
        Initialisiert die Pipeline mit MIMIC-IV Basis-Pfad.
        
        Args:
            base_path: Pfad zu MIMIC-IV Root (z.B. "D:/MA/physionet.org/files/mimic-iv/3.1")
        """
        self.base_path = Path(base_path)
        self.validate_path()
        
        # Definiere Standard-Pfade
        self.paths = {
            # ICU Daten
            'icustays': self.base_path / 'icu' / 'icustays.csv',
            'chartevents': self.base_path / 'icu' / 'chartevents.csv',
            'inputevents': self.base_path / 'icu' / 'inputevents.csv',
            'outputevents': self.base_path / 'icu' / 'outputevents.csv',
            'd_items': self.base_path / 'icu' / 'd_items.csv',
            
            # Hospital Daten
            'labevents': self.base_path / 'hosp' / 'labevents.csv',
            'd_labitems': self.base_path / 'hosp' / 'd_labitems.csv',
            'patients': self.base_path / 'hosp' / 'patients.csv',
            'admissions': self.base_path / 'hosp' / 'admissions.csv',
        }
    
    def validate_path(self):
        """Validiert, ob MIMIC-IV Pfad existiert und korrekte Struktur hat."""
        if not self.base_path.exists():
            raise FileNotFoundError(f"MIMIC-IV Pfad nicht gefunden: {self.base_path}")
        
        # Prüfe ob icu/ und hosp/ Ordner existieren
        required_dirs = ['icu', 'hosp']
        for dir_name in required_dirs:
            dir_path = self.base_path / dir_name
            if not dir_path.exists():
                raise FileNotFoundError(f"MIMIC-IV Unterordner nicht gefunden: {dir_path}")
        
        print(f"✓ MIMIC-IV Pfad validiert: {self.base_path}")
    
    # =========================================================================
    # ICU STAYS
    # =========================================================================
    
    def load_icustays(
        self,
        subject_ids: Optional[List[int]] = None,
        min_los_hours: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Lädt ICU Stays.
        
        Args:
            subject_ids: Optional - Filter auf bestimmte Patienten
            min_los_hours: Optional - Minimale Length of Stay in Stunden
            
        Returns:
            DataFrame mit ICU Stays
        """
        print("📋 Lade ICU Stays...")
        
        icustays = pd.read_csv(
            self.paths['icustays'],
            usecols=['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime', 'los']
        )
        
        # Filter auf Patienten
        if subject_ids is not None:
            icustays = icustays[icustays['subject_id'].isin(subject_ids)]
        
        # Filter auf minimale Length of Stay
        if min_los_hours is not None:
            icustays = icustays[icustays['los'] >= (min_los_hours / 24)]
        
        # Konvertiere Zeitstempel
        icustays['intime'] = pd.to_datetime(icustays['intime'])
        icustays['outtime'] = pd.to_datetime(icustays['outtime'])
        
        print(f"  ✓ {len(icustays)} ICU Stays geladen")
        return icustays
    
    # =========================================================================
    # CHART EVENTS (Vitals, Monitoring)
    # =========================================================================
    
    def load_chartevents(
        self,
        stay_ids: List[int],
        itemids: List[int],
        time_window_hours: Optional[int] = None,
        chunk_size: int = 100000
    ) -> pd.DataFrame:
        """
        Lädt Chart Events (Vitals, Monitoring-Daten).
        
        Args:
            stay_ids: Liste von ICU Stay IDs
            itemids: Liste von Item IDs (zu ladende Parameter)
            time_window_hours: Optional - Zeitfenster nach ICU Admission
            chunk_size: Chunk-Größe für große Datei
            
        Returns:
            DataFrame mit Chart Events
        """
        print(f"📊 Lade Chart Events ({len(itemids)} items)...")
        
        # Lade in Chunks (chartevents ist riesig!)
        chunks = []
        total_rows = 0
        
        for chunk in pd.read_csv(self.paths['chartevents'], chunksize=chunk_size):
            # Filter
            chunk = chunk[
                chunk['stay_id'].isin(stay_ids) &
                chunk['itemid'].isin(itemids)
            ]
            
            if len(chunk) > 0:
                chunks.append(chunk)
                total_rows += len(chunk)
                print(f"  → {total_rows} Zeilen gefunden", end='\r')
        
        if not chunks:
            print("\n  ⚠️  Keine Chart Events gefunden!")
            return pd.DataFrame()
        
        chart_df = pd.concat(chunks, ignore_index=True)
        chart_df['charttime'] = pd.to_datetime(chart_df['charttime'])
        
        print(f"\n  ✓ {len(chart_df)} Chart Events geladen")
        return chart_df
    
    # =========================================================================
    # LAB EVENTS (Laborwerte)
    # =========================================================================
    
    def load_labevents(
        self,
        subject_ids: List[int],
        itemids: List[int],
        time_window_hours: Optional[int] = None,
        chunk_size: int = 100000
    ) -> pd.DataFrame:
        """
        Lädt Lab Events (Laborwerte).
        
        Args:
            subject_ids: Liste von Patient IDs
            itemids: Liste von Lab Item IDs
            time_window_hours: Optional - Zeitfenster
            chunk_size: Chunk-Größe
            
        Returns:
            DataFrame mit Lab Events
        """
        print(f"🧪 Lade Lab Events ({len(itemids)} items)...")
        
        chunks = []
        total_rows = 0
        
        for chunk in pd.read_csv(self.paths['labevents'], chunksize=chunk_size):
            chunk = chunk[
                chunk['subject_id'].isin(subject_ids) &
                chunk['itemid'].isin(itemids)
            ]
            
            if len(chunk) > 0:
                chunks.append(chunk)
                total_rows += len(chunk)
                print(f"  → {total_rows} Zeilen gefunden", end='\r')
        
        if not chunks:
            print("\n  ⚠️  Keine Lab Events gefunden!")
            return pd.DataFrame()
        
        lab_df = pd.concat(chunks, ignore_index=True)
        lab_df['charttime'] = pd.to_datetime(lab_df['charttime'])
        
        print(f"\n  ✓ {len(lab_df)} Lab Events geladen")
        return lab_df
    
    # =========================================================================
    # INPUT EVENTS (Medikamente, Vasopressoren)
    # =========================================================================
    
    def load_inputevents(
        self,
        stay_ids: List[int],
        itemids: List[int]
    ) -> pd.DataFrame:
        """
        Lädt Input Events (Medikamente, Infusionen).
        
        Args:
            stay_ids: Liste von ICU Stay IDs
            itemids: Liste von Item IDs
            
        Returns:
            DataFrame mit Input Events
        """
        print(f"💉 Lade Input Events ({len(itemids)} items)...")
        
        input_df = pd.read_csv(self.paths['inputevents'])
        input_df = input_df[
            input_df['stay_id'].isin(stay_ids) &
            input_df['itemid'].isin(itemids)
        ]
        
        input_df['starttime'] = pd.to_datetime(input_df['starttime'])
        
        print(f"  ✓ {len(input_df)} Input Events geladen")
        return input_df
    
    # =========================================================================
    # OUTPUT EVENTS (Urin, Drainage)
    # =========================================================================
    
    def load_outputevents(
        self,
        stay_ids: List[int],
        itemids: List[int]
    ) -> pd.DataFrame:
        """
        Lädt Output Events (Urin, Drainagen).
        
        Args:
            stay_ids: Liste von ICU Stay IDs
            itemids: Liste von Item IDs
            
        Returns:
            DataFrame mit Output Events
        """
        print(f"💧 Lade Output Events ({len(itemids)} items)...")
        
        output_df = pd.read_csv(self.paths['outputevents'])
        output_df = output_df[
            output_df['stay_id'].isin(stay_ids) &
            output_df['itemid'].isin(itemids)
        ]
        
        output_df['charttime'] = pd.to_datetime(output_df['charttime'])
        
        print(f"  ✓ {len(output_df)} Output Events geladen")
        return output_df
    
    # =========================================================================
    # HILFSFUNKTIONEN
    # =========================================================================
    
    def filter_by_time_window(
        self,
        df: pd.DataFrame,
        icustays: pd.DataFrame,
        time_column: str = 'charttime',
        time_window_hours: int = 24
    ) -> pd.DataFrame:
        """
        Filtert DataFrame auf Zeitfenster nach ICU Admission.
        
        Args:
            df: DataFrame mit Daten
            icustays: ICU Stays DataFrame
            time_column: Name der Zeit-Spalte
            time_window_hours: Zeitfenster in Stunden
            
        Returns:
            Gefilterter DataFrame
        """
        # Merge mit ICU times
        df = df.merge(
            icustays[['subject_id', 'hadm_id', 'stay_id', 'intime']],
            on=['subject_id', 'hadm_id', 'stay_id'],
            how='inner'
        )
        
        # Berechne Zeitfenster
        df['endtime'] = df['intime'] + timedelta(hours=time_window_hours)
        
        # Filter
        df = df[
            (df[time_column] >= df['intime']) &
            (df[time_column] <= df['endtime'])
        ]
        
        return df.drop(columns=['endtime'])
    
    def get_item_labels(self, table: str = 'chartevents') -> pd.DataFrame:
        """
        Lädt Item-Labels (d_items oder d_labitems).
        
        Args:
            table: 'chartevents' oder 'labevents'
            
        Returns:
            DataFrame mit Item-Labels
        """
        if table == 'chartevents':
            return pd.read_csv(self.paths['d_items'])
        elif table == 'labevents':
            return pd.read_csv(self.paths['d_labitems'])
        else:
            raise ValueError(f"Unbekannte Tabelle: {table}")

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_pipeline(base_path: Optional[str] = None) -> MimicIVPipeline:
    """
    Erstellt Pipeline-Instanz mit Standard-Pfad.
    
    Args:
        base_path: Optional - Falls None, wird versucht aus config zu laden
        
    Returns:
        MimicIVPipeline Instanz
    """
    if base_path is None:
        # Versuche aus config zu laden
        try:
            from ...baseline_models import config
            base_path = config.MIMIC_IV_BASE_PATH
        except ImportError:
            raise ValueError("base_path muss angegeben werden oder in config.py definiert sein")
    
    return MimicIVPipeline(base_path)

