"""
MIMIC-IV Data Loader für SOFA Score Berechnung

Lädt und extrahiert alle benötigten Daten aus MIMIC-IV für die SOFA-Berechnung.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# tqdm optional (für Progress Bars)
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

from .. import config
from . import itemid_mappings as itemids

# =============================================================================
# HAUPTFUNKTION: SOFA-DATEN LADEN
# =============================================================================

def load_sofa_data(
    subject_ids: Optional[List[int]] = None,
    time_window_hours: int = 24
) -> pd.DataFrame:
    """
    Lädt alle benötigten Daten für SOFA Score Berechnung aus MIMIC-IV.
    
    Args:
        subject_ids: Optional - Liste von subject_ids (falls nur Teilmenge)
        time_window_hours: Zeitfenster nach ICU Admission in Stunden
        
    Returns:
        DataFrame mit aggregierten Werten pro Patient für SOFA-Berechnung
    """
    print("\n" + "="*70)
    print("MIMIC-IV Daten laden für SOFA Score Berechnung")
    print("="*70)
    
    # 1. Lade ICU Stays (Basis für Zeitfenster)
    print("\n📋 Lade ICU Stays...")
    icustays = load_icustays(subject_ids)
    print(f"  ✓ {len(icustays)} ICU Aufenthalte geladen")
    
    # 2. Lade Labor-Werte (langsam, mit Chunks)
    print("\n🧪 Lade Labor-Werte (labevents)...")
    lab_data = load_labevents(icustays, time_window_hours)
    print(f"  ✓ Laborwerte aggregiert")
    
    # 3. Lade Chart-Events (Vitals, GCS, etc.)
    print("\n📊 Lade Chart-Events (Vitals, GCS, etc.)...")
    chart_data = load_chartevents(icustays, time_window_hours)
    print(f"  ✓ Chart-Events aggregiert")
    
    # 4. Lade Vasopressor-Daten
    print("\n💉 Lade Vasopressor-Daten (inputevents)...")
    vasopressor_data = load_vasopressors(icustays, time_window_hours)
    print(f"  ✓ Vasopressor-Daten aggregiert")
    
    # 5. Lade Urin-Output
    print("\n💧 Lade Urin-Output (outputevents)...")
    urine_data = load_urine_output(icustays, time_window_hours)
    print(f"  ✓ Urin-Output aggregiert")
    
    # 6. Merge alle Daten zusammen
    print("\n🔗 Merge Daten zusammen...")
    sofa_data = merge_sofa_data(
        icustays, lab_data, chart_data, vasopressor_data, urine_data
    )
    print(f"  ✓ {len(sofa_data)} vollständige Patienten-Records")
    
    print("\n" + "="*70)
    print("✅ Daten erfolgreich geladen!")
    print("="*70 + "\n")
    
    return sofa_data

# =============================================================================
# 1. ICU STAYS LADEN
# =============================================================================

def load_icustays(subject_ids: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Lädt ICU Stays als Basis für Zeitfenster.
    
    Returns:
        DataFrame mit subject_id, hadm_id, stay_id, intime, outtime, los
    """
    icustays = pd.read_csv(
        config.MIMIC_IV_PATHS['icustays'],
        usecols=['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime', 'los']
    )
    
    # Filter auf bestimmte Patienten falls gewünscht
    if subject_ids is not None:
        icustays = icustays[icustays['subject_id'].isin(subject_ids)]
    
    # Konvertiere Zeitstempel
    icustays['intime'] = pd.to_datetime(icustays['intime'])
    icustays['outtime'] = pd.to_datetime(icustays['outtime'])
    
    return icustays

# =============================================================================
# 2. LABORWERTE LADEN (labevents)
# =============================================================================

def load_labevents(
    icustays: pd.DataFrame,
    time_window_hours: int
) -> pd.DataFrame:
    """
    Lädt Laborwerte: PaO2, Bilirubin, Kreatinin, Thrombozyten.
    
    Verwendet Chunks wegen großer Datei.
    """
    relevant_itemids = itemids.get_all_labevents_itemids()
    stay_ids = icustays['subject_id'].unique()
    
    # Zeitfenster definieren
    icustays_time = icustays[['subject_id', 'hadm_id', 'stay_id', 'intime']].copy()
    icustays_time['endtime'] = icustays_time['intime'] + timedelta(hours=time_window_hours)
    
    # Initialisiere leere Liste für Chunks
    chunk_list = []
    
    # Lade labevents in Chunks
    print(f"  Lade labevents (große Datei, kann dauern)...")
    chunk_iter = pd.read_csv(
        config.MIMIC_IV_PATHS['labevents'],
        usecols=['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum'],
        chunksize=config.CHUNK_SIZE
    )
    
    for chunk in tqdm(chunk_iter, desc="  Processing chunks"):
        # Filter: Nur relevante subject_ids und itemids
        chunk = chunk[
            chunk['subject_id'].isin(stay_ids) &
            chunk['itemid'].isin(relevant_itemids)
        ]
        
        if len(chunk) > 0:
            chunk['charttime'] = pd.to_datetime(chunk['charttime'])
            chunk_list.append(chunk)
    
    # Kombiniere alle Chunks
    if len(chunk_list) == 0:
        print("  ⚠️  Keine Laborwerte gefunden!")
        return pd.DataFrame()
    
    lab_df = pd.concat(chunk_list, ignore_index=True)
    
    # Merge mit ICU Stay Zeiten
    lab_df = lab_df.merge(icustays_time, on=['subject_id', 'hadm_id'], how='inner')
    
    # Filter auf Zeitfenster
    lab_df = lab_df[
        (lab_df['charttime'] >= lab_df['intime']) &
        (lab_df['charttime'] <= lab_df['endtime'])
    ]
    
    # Aggregiere: Worst case Werte pro Patient
    lab_aggregated = aggregate_lab_values(lab_df)
    
    return lab_aggregated

def aggregate_lab_values(lab_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregiert Laborwerte: Worst case (Min/Max je nach Parameter).
    """
    results = []
    
    for (subject_id, hadm_id, stay_id), group in lab_df.groupby(['subject_id', 'hadm_id', 'stay_id']):
        patient_labs = {
            'subject_id': subject_id,
            'hadm_id': hadm_id,
            'stay_id': stay_id,
        }
        
        # PaO2: Minimum (schlechtester Wert)
        pao2_values = group[group['itemid'].isin(itemids.PAO2_ITEMIDS)]['valuenum']
        if len(pao2_values) > 0:
            patient_labs['pao2'] = pao2_values.min()
        
        # Bilirubin: Maximum (schlechtester Wert)
        bili_values = group[group['itemid'].isin(itemids.BILIRUBIN_ITEMIDS)]['valuenum']
        if len(bili_values) > 0:
            patient_labs['bilirubin'] = bili_values.max()
        
        # Kreatinin: Maximum (schlechtester Wert)
        creat_values = group[group['itemid'].isin(itemids.CREATININE_ITEMIDS)]['valuenum']
        if len(creat_values) > 0:
            patient_labs['creatinine'] = creat_values.max()
        
        # Thrombozyten: Minimum (schlechtester Wert)
        plt_values = group[group['itemid'].isin(itemids.PLATELETS_ITEMIDS)]['valuenum']
        if len(plt_values) > 0:
            patient_labs['platelets'] = plt_values.min()
        
        results.append(patient_labs)
    
    return pd.DataFrame(results)

# =============================================================================
# 3. CHART EVENTS LADEN (Vitals, GCS, FiO2)
# =============================================================================

def load_chartevents(
    icustays: pd.DataFrame,
    time_window_hours: int
) -> pd.DataFrame:
    """
    Lädt Chart-Events: MAP, GCS, FiO2, Beatmung, Gewicht.
    """
    relevant_itemids = itemids.get_all_chartevents_itemids()
    stay_ids = icustays['stay_id'].unique()
    
    # Zeitfenster
    icustays_time = icustays[['subject_id', 'hadm_id', 'stay_id', 'intime']].copy()
    icustays_time['endtime'] = icustays_time['intime'] + timedelta(hours=time_window_hours)
    
    chunk_list = []
    
    print(f"  Lade chartevents (große Datei, kann dauern)...")
    chunk_iter = pd.read_csv(
        config.MIMIC_IV_PATHS['chartevents'],
        usecols=['subject_id', 'hadm_id', 'stay_id', 'itemid', 'charttime', 'valuenum', 'value'],
        chunksize=config.CHUNK_SIZE
    )
    
    for chunk in tqdm(chunk_iter, desc="  Processing chunks"):
        chunk = chunk[
            chunk['stay_id'].isin(stay_ids) &
            chunk['itemid'].isin(relevant_itemids)
        ]
        
        if len(chunk) > 0:
            chunk['charttime'] = pd.to_datetime(chunk['charttime'])
            chunk_list.append(chunk)
    
    if len(chunk_list) == 0:
        print("  ⚠️  Keine Chart-Events gefunden!")
        return pd.DataFrame()
    
    chart_df = pd.concat(chunk_list, ignore_index=True)
    
    # Merge mit Zeitfenster
    chart_df = chart_df.merge(icustays_time, on=['subject_id', 'hadm_id', 'stay_id'], how='inner')
    chart_df = chart_df[
        (chart_df['charttime'] >= chart_df['intime']) &
        (chart_df['charttime'] <= chart_df['endtime'])
    ]
    
    # Aggregiere
    chart_aggregated = aggregate_chart_values(chart_df)
    
    return chart_aggregated

def aggregate_chart_values(chart_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregiert Chart-Werte: Worst case.
    """
    results = []
    
    for (subject_id, hadm_id, stay_id), group in chart_df.groupby(['subject_id', 'hadm_id', 'stay_id']):
        patient_chart = {
            'subject_id': subject_id,
            'hadm_id': hadm_id,
            'stay_id': stay_id,
        }
        
        # MAP: Minimum (schlechtester Wert)
        map_values = group[group['itemid'].isin(itemids.MAP_ITEMIDS)]['valuenum']
        if len(map_values) > 0:
            patient_chart['map'] = map_values.min()
        
        # GCS: Minimum (schlechtester Wert)
        gcs_values = group[group['itemid'].isin(itemids.GCS_TOTAL_ITEMIDS)]['valuenum']
        if len(gcs_values) > 0:
            patient_chart['gcs'] = gcs_values.min()
        
        # FiO2: Maximum (worst case für PaO2/FiO2 Ratio)
        fio2_values = group[group['itemid'].isin(itemids.FIO2_ITEMIDS)]['valuenum']
        if len(fio2_values) > 0:
            # FiO2 kann als Prozent (0-100) oder Dezimal (0-1) gespeichert sein
            fio2_max = fio2_values.max()
            # Normalisiere auf Dezimal
            if fio2_max > 1:
                fio2_max = fio2_max / 100.0
            patient_chart['fio2'] = fio2_max
        
        # Beatmung: Ja/Nein
        vent_values = group[group['itemid'].isin(itemids.MECHANICAL_VENTILATION_ITEMIDS)]
        patient_chart['is_ventilated'] = len(vent_values) > 0
        
        # Gewicht (für Vasopressor-Dosierung)
        weight_values = group[group['itemid'].isin(itemids.WEIGHT_ITEMIDS)]['valuenum']
        if len(weight_values) > 0:
            patient_chart['weight_kg'] = weight_values.median()  # Median als robuster Schätzer
        
        results.append(patient_chart)
    
    return pd.DataFrame(results)

# =============================================================================
# 4. VASOPRESSOREN LADEN (inputevents)
# =============================================================================

def load_vasopressors(
    icustays: pd.DataFrame,
    time_window_hours: int
) -> pd.DataFrame:
    """
    Lädt Vasopressor-Dosierungen aus inputevents.
    """
    relevant_itemids = itemids.get_all_inputevents_itemids()
    stay_ids = icustays['stay_id'].unique()
    
    # Zeitfenster
    icustays_time = icustays[['subject_id', 'hadm_id', 'stay_id', 'intime']].copy()
    icustays_time['endtime'] = icustays_time['intime'] + timedelta(hours=time_window_hours)
    
    # Lade inputevents
    print(f"  Lade inputevents...")
    input_df = pd.read_csv(
        config.MIMIC_IV_PATHS['inputevents'],
        usecols=['subject_id', 'hadm_id', 'stay_id', 'itemid', 'starttime', 'endtime', 'rate', 'amount']
    )
    
    # Filter
    input_df = input_df[
        input_df['stay_id'].isin(stay_ids) &
        input_df['itemid'].isin(relevant_itemids)
    ]
    
    if len(input_df) == 0:
        print("  ⚠️  Keine Vasopressor-Daten gefunden!")
        return pd.DataFrame()
    
    input_df['starttime'] = pd.to_datetime(input_df['starttime'])
    
    # Merge mit Zeitfenster
    input_df = input_df.merge(icustays_time[['subject_id', 'hadm_id', 'stay_id', 'intime', 'endtime']], 
                              on=['subject_id', 'hadm_id', 'stay_id'], 
                              how='inner',
                              suffixes=('', '_icu'))
    
    input_df = input_df[
        (input_df['starttime'] >= input_df['intime']) &
        (input_df['starttime'] <= input_df['endtime_icu'])
    ]
    
    # Aggregiere Vasopressor-Dosen
    vaso_aggregated = aggregate_vasopressor_doses(input_df)
    
    return vaso_aggregated

def aggregate_vasopressor_doses(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregiert Vasopressor-Dosierungen: Maximum (worst case).
    """
    results = []
    
    for (subject_id, hadm_id, stay_id), group in input_df.groupby(['subject_id', 'hadm_id', 'stay_id']):
        patient_vaso = {
            'subject_id': subject_id,
            'hadm_id': hadm_id,
            'stay_id': stay_id,
            'dopamine': 0.0,
            'norepinephrine': 0.0,
            'epinephrine': 0.0,
            'dobutamine': 0.0,
        }
        
        # Dopamin
        dopa_rates = group[group['itemid'].isin(itemids.DOPAMINE_ITEMIDS)]['rate']
        if len(dopa_rates) > 0:
            patient_vaso['dopamine'] = dopa_rates.max()
        
        # Noradrenalin
        norepi_rates = group[group['itemid'].isin(itemids.NOREPINEPHRINE_ITEMIDS)]['rate']
        if len(norepi_rates) > 0:
            patient_vaso['norepinephrine'] = norepi_rates.max()
        
        # Adrenalin
        epi_rates = group[group['itemid'].isin(itemids.EPINEPHRINE_ITEMIDS)]['rate']
        if len(epi_rates) > 0:
            patient_vaso['epinephrine'] = epi_rates.max()
        
        # Dobutamin
        dobuta_rates = group[group['itemid'].isin(itemids.DOBUTAMINE_ITEMIDS)]['rate']
        if len(dobuta_rates) > 0:
            patient_vaso['dobutamine'] = dobuta_rates.max()
        
        results.append(patient_vaso)
    
    return pd.DataFrame(results)

# =============================================================================
# 5. URIN-OUTPUT LADEN (outputevents)
# =============================================================================

def load_urine_output(
    icustays: pd.DataFrame,
    time_window_hours: int
) -> pd.DataFrame:
    """
    Lädt Urin-Output aus outputevents.
    """
    relevant_itemids = itemids.get_all_outputevents_itemids()
    stay_ids = icustays['stay_id'].unique()
    
    # Zeitfenster
    icustays_time = icustays[['subject_id', 'hadm_id', 'stay_id', 'intime']].copy()
    icustays_time['endtime'] = icustays_time['intime'] + timedelta(hours=time_window_hours)
    
    print(f"  Lade outputevents...")
    output_df = pd.read_csv(
        config.MIMIC_IV_PATHS['outputevents'],
        usecols=['subject_id', 'hadm_id', 'stay_id', 'itemid', 'charttime', 'value']
    )
    
    # Filter
    output_df = output_df[
        output_df['stay_id'].isin(stay_ids) &
        output_df['itemid'].isin(relevant_itemids)
    ]
    
    if len(output_df) == 0:
        print("  ⚠️  Keine Urin-Output-Daten gefunden!")
        return pd.DataFrame()
    
    output_df['charttime'] = pd.to_datetime(output_df['charttime'])
    
    # Merge mit Zeitfenster
    output_df = output_df.merge(icustays_time, on=['subject_id', 'hadm_id', 'stay_id'], how='inner')
    output_df = output_df[
        (output_df['charttime'] >= output_df['intime']) &
        (output_df['charttime'] <= output_df['endtime'])
    ]
    
    # Aggregiere: Summe über Zeitfenster
    urine_aggregated = output_df.groupby(['subject_id', 'hadm_id', 'stay_id']).agg({
        'value': 'sum'  # Totaler Urin-Output
    }).reset_index()
    
    urine_aggregated.rename(columns={'value': 'urine_output_24h'}, inplace=True)
    
    return urine_aggregated

# =============================================================================
# 6. MERGE ALLE DATEN
# =============================================================================

def merge_sofa_data(
    icustays: pd.DataFrame,
    lab_data: pd.DataFrame,
    chart_data: pd.DataFrame,
    vasopressor_data: pd.DataFrame,
    urine_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Merged alle Teil-DataFrames zu einem kompletten SOFA-DataFrame.
    """
    # Starte mit ICU Stays
    sofa_df = icustays[['subject_id', 'hadm_id', 'stay_id']].copy()
    
    # Merge Labor-Daten
    if len(lab_data) > 0:
        sofa_df = sofa_df.merge(lab_data, on=['subject_id', 'hadm_id', 'stay_id'], how='left')
    
    # Merge Chart-Daten
    if len(chart_data) > 0:
        sofa_df = sofa_df.merge(chart_data, on=['subject_id', 'hadm_id', 'stay_id'], how='left')
    
    # Merge Vasopressor-Daten
    if len(vasopressor_data) > 0:
        sofa_df = sofa_df.merge(vasopressor_data, on=['subject_id', 'hadm_id', 'stay_id'], how='left')
        # Fülle fehlende Vasopressor-Werte mit 0
        for col in ['dopamine', 'norepinephrine', 'epinephrine', 'dobutamine']:
            if col in sofa_df.columns:
                sofa_df[col].fillna(0, inplace=True)
    
    # Merge Urin-Daten
    if len(urine_data) > 0:
        sofa_df = sofa_df.merge(urine_data, on=['subject_id', 'hadm_id', 'stay_id'], how='left')
    
    # Berechne PaO2/FiO2 Ratio
    if 'pao2' in sofa_df.columns and 'fio2' in sofa_df.columns:
        sofa_df['pao2_fio2_ratio'] = sofa_df['pao2'] / sofa_df['fio2']
        # Ersetze inf und sehr hohe Werte
        sofa_df['pao2_fio2_ratio'] = sofa_df['pao2_fio2_ratio'].replace([np.inf, -np.inf], np.nan)
        sofa_df.loc[sofa_df['pao2_fio2_ratio'] > 600, 'pao2_fio2_ratio'] = 600  # Cap bei 600
    
    # Normalisiere Vasopressor-Dosen auf μg/kg/min
    if 'weight_kg' in sofa_df.columns:
        for vaso in ['dopamine', 'norepinephrine', 'epinephrine', 'dobutamine']:
            if vaso in sofa_df.columns:
                # Falls Gewicht vorhanden, normalisiere
                sofa_df[vaso] = sofa_df.apply(
                    lambda row: row[vaso] / row['weight_kg'] if pd.notna(row.get('weight_kg')) and row.get('weight_kg', 0) > 0 
                                else row[vaso] / config.DEFAULT_PATIENT_WEIGHT_KG,
                    axis=1
                )
    else:
        # Falls kein Gewicht: Nutze Default
        for vaso in ['dopamine', 'norepinephrine', 'epinephrine', 'dobutamine']:
            if vaso in sofa_df.columns:
                sofa_df[vaso] = sofa_df[vaso] / config.DEFAULT_PATIENT_WEIGHT_KG
    
    return sofa_df

# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'load_sofa_data',
    'load_icustays',
    'load_labevents',
    'load_chartevents',
    'load_vasopressors',
    'load_urine_output',
    'merge_sofa_data',
]

