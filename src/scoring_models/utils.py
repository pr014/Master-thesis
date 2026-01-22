"""
Gemeinsame Hilfsfunktionen für alle Clinical Baseline Models

Enthält wiederverwendbare Funktionen für:
- Datenvalidierung
- Statistiken
- Export/Save
- Plausibilitäts-Checks
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import os
from datetime import datetime

# =============================================================================
# DATENVALIDIERUNG
# =============================================================================

def validate_data(
    df: pd.DataFrame,
    required_columns: List[str],
    score_name: str = "Score"
) -> Dict:
    """
    Validiert Datenqualität für Score-Berechnung.
    
    Args:
        df: DataFrame mit Daten
        required_columns: Liste benötigter Spalten
        score_name: Name des Scores (für Output)
        
    Returns:
        Dictionary mit Validierungs-Statistiken
    """
    print("\n" + "="*70)
    print(f"{score_name.upper()} - DATENQUALITÄT VALIDIERUNG")
    print("="*70)
    
    stats = {
        'total_patients': len(df),
        'missing_columns': [],
        'missing_values': {},
        'available_data': {},
    }
    
    # Prüfe fehlende Spalten
    for col in required_columns:
        if col not in df.columns:
            stats['missing_columns'].append(col)
            print(f"  ⚠️  Spalte fehlt: {col}")
    
    # Prüfe Missing Values
    print("\nFehlende Werte:")
    print("-" * 70)
    
    for col in required_columns:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            available = len(df) - missing_count
            
            stats['missing_values'][col] = missing_count
            stats['available_data'][col] = available
            
            status = "✓" if missing_pct < 20 else "⚠️"
            print(f"  {status} {col:30s}: {available:5d} / {len(df):5d} ({100-missing_pct:5.1f}%)")
        else:
            stats['missing_values'][col] = len(df)
            stats['available_data'][col] = 0
            print(f"  ❌ {col:30s}: NICHT VORHANDEN!")
    
    print("="*70 + "\n")
    
    return stats

# =============================================================================
# STATISTIKEN
# =============================================================================

def calculate_score_statistics(
    df: pd.DataFrame,
    score_column: str = 'total_score',
    component_columns: Optional[List[str]] = None
) -> Dict:
    """
    Berechnet deskriptive Statistiken für Scores.
    
    Args:
        df: DataFrame mit Scores
        score_column: Spaltenname des Gesamt-Scores
        component_columns: Optional - Liste der Komponenten-Spalten
        
    Returns:
        Dictionary mit Statistiken
    """
    if score_column not in df.columns:
        print(f"⚠️  Score-Spalte '{score_column}' nicht gefunden!")
        return {}
    
    stats = {
        'n': len(df),
        'mean': df[score_column].mean(),
        'std': df[score_column].std(),
        'median': df[score_column].median(),
        'min': df[score_column].min(),
        'max': df[score_column].max(),
        'q25': df[score_column].quantile(0.25),
        'q75': df[score_column].quantile(0.75),
        'distribution': df[score_column].value_counts().sort_index().to_dict(),
    }
    
    # Komponenten-Statistiken
    if component_columns:
        stats['components'] = {}
        for col in component_columns:
            if col in df.columns:
                stats['components'][col] = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std(),
                }
    
    return stats

def print_statistics(
    stats: Dict,
    score_name: str = "Score"
):
    """Gibt Statistiken formatiert aus."""
    print("\n" + "="*70)
    print(f"{score_name.upper()} - STATISTIKEN")
    print("="*70)
    
    print(f"\nGesamt-{score_name} (n={stats['n']}):")
    print("-" * 70)
    print(f"  Mean ± SD:  {stats['mean']:.2f} ± {stats['std']:.2f}")
    print(f"  Median:     {stats['median']:.1f}")
    print(f"  Range:      {stats['min']:.0f} - {stats['max']:.0f}")
    print(f"  IQR:        {stats['q25']:.1f} - {stats['q75']:.1f}")
    
    if 'components' in stats:
        print("\nKomponenten (Mittelwert ± SD):")
        print("-" * 70)
        for comp_name, comp_stats in stats['components'].items():
            print(f"  {comp_name:30s}: {comp_stats['mean']:.2f} ± {comp_stats['std']:.2f}")
    
    print("="*70 + "\n")

# =============================================================================
# EXPORT / SAVE
# =============================================================================

def save_results(
    df: pd.DataFrame,
    output_path: str,
    score_name: str = "score",
    score_columns: List[str] = ['total_score'],
    save_components: bool = True
):
    """
    Speichert Score-Ergebnisse als CSV.
    
    Args:
        df: DataFrame mit Scores
        output_path: Ausgabe-Pfad
        score_name: Name des Scores (für Dateinamen)
        score_columns: Liste der Score-Spalten
        save_components: Ob Komponenten auch gespeichert werden sollen
    """
    print("\n💾 Speichere Ergebnisse...")
    
    # 1. Haupt-Output: Total Scores
    main_cols = ['subject_id', 'hadm_id', 'stay_id'] + score_columns
    available_cols = [col for col in main_cols if col in df.columns]
    
    if available_cols:
        output_file = os.path.join(output_path, f'{score_name}_scores.csv')
        df[available_cols].to_csv(output_file, index=False)
        print(f"  ✓ Gespeichert: {output_file}")
    
    # 2. Vollständiger Datensatz
    full_file = os.path.join(output_path, f'{score_name}_complete_data.csv')
    df.to_csv(full_file, index=False)
    print(f"  ✓ Gespeichert: {full_file}")
    
    print()

def save_statistics(
    stats: Dict,
    output_path: str,
    score_name: str = "Score",
    validation_stats: Optional[Dict] = None
):
    """Speichert Statistiken als Text-Datei."""
    output_file = os.path.join(output_path, f'{score_name.lower()}_statistics.txt')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(f"{score_name.upper()} BERECHNUNG - STATISTIKEN\n")
        f.write(f"Generiert am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        
        # Datenqualität
        if validation_stats:
            f.write("DATENQUALITÄT\n")
            f.write("-"*70 + "\n")
            f.write(f"Gesamt Patienten: {validation_stats['total_patients']}\n\n")
        
        # Statistiken
        f.write(f"{score_name.upper()} STATISTIKEN\n")
        f.write("-"*70 + "\n")
        f.write(f"N: {stats['n']}\n")
        f.write(f"Mean ± SD: {stats['mean']:.2f} ± {stats['std']:.2f}\n")
        f.write(f"Median: {stats['median']:.1f}\n")
        f.write(f"Range: {stats['min']:.0f} - {stats['max']:.0f}\n")
        f.write(f"IQR: {stats['q25']:.1f} - {stats['q75']:.1f}\n\n")
        
        # Komponenten
        if 'components' in stats:
            f.write("KOMPONENTEN (Mittelwert ± SD):\n")
            f.write("-"*70 + "\n")
            for comp_name, comp_stats in stats['components'].items():
                f.write(f"  {comp_name:30s}: {comp_stats['mean']:.2f} ± {comp_stats['std']:.2f}\n")
    
    print(f"  ✓ Statistiken gespeichert: {output_file}\n")

# =============================================================================
# PLAUSIBILITÄTS-CHECKS
# =============================================================================

def sanity_check_scores(
    df: pd.DataFrame,
    score_column: str = 'total_score',
    valid_range: tuple = (0, 24),
    expected_mean_range: tuple = (2, 15),
    score_name: str = "Score"
) -> bool:
    """
    Führt Plausibilitäts-Checks für berechnete Scores durch.
    
    Args:
        df: DataFrame mit Scores
        score_column: Spaltenname des Scores
        valid_range: Gültiger Wertebereich (min, max)
        expected_mean_range: Erwarteter Mittelwert-Bereich
        score_name: Name des Scores (für Output)
        
    Returns:
        True wenn alle Checks bestanden, False sonst
    """
    print(f"\n🔍 Plausibilitäts-Checks für {score_name}...")
    print("-" * 70)
    
    issues = []
    
    # Check 1: Score im gültigen Bereich?
    if score_column in df.columns:
        min_val, max_val = valid_range
        invalid_scores = df[(df[score_column] < min_val) | (df[score_column] > max_val)]
        if len(invalid_scores) > 0:
            issues.append(f"⚠️  {len(invalid_scores)} Patienten mit ungültigem {score_name} (außerhalb {min_val}-{max_val})")
    
    # Check 2: Plausible Verteilung?
    if score_column in df.columns:
        mean_score = df[score_column].mean()
        min_expected, max_expected = expected_mean_range
        if mean_score < min_expected or mean_score > max_expected:
            issues.append(f"⚠️  Ungewöhnlicher Mittelwert: {mean_score:.1f} (erwartet: {min_expected}-{max_expected})")
    
    # Output
    if len(issues) == 0:
        print("  ✓ Alle Plausibilitäts-Checks bestanden!")
    else:
        print("  Gefundene Probleme:")
        for issue in issues:
            print(f"    {issue}")
    
    print("-" * 70 + "\n")
    
    return len(issues) == 0

# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'validate_data',
    'calculate_score_statistics',
    'print_statistics',
    'save_results',
    'save_statistics',
    'sanity_check_scores',
]

