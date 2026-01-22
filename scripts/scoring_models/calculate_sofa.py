"""
SOFA Score Berechnung - Main Script

Berechnet SOFA Scores aus MIMIC-IV Daten für alle ICU-Patienten.

Usage:
    python scripts/baseline_models/calculate_sofa.py

Output:
    outputs/baseline_models/sofa/
    ├── sofa_scores.csv
    ├── sofa_complete_data.csv
    └── sofa_statistics.txt
"""

import sys
import os
from datetime import datetime
from pathlib import Path

# Füge Projekt-Root zu Python Path hinzu
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Importiere Module
from src.baseline_models import config, utils
from src.baseline_models.sofa import (
    calculator,
    itemid_mappings
)
# Import data_loader functions directly
from src.baseline_models.sofa.data_loader import load_icustays, load_sofa_data
import src.baseline_models.sofa.data_loader as data_loader

# =============================================================================
# MAIN FUNKTION
# =============================================================================

def main():
    """Hauptfunktion: Führt SOFA Score Berechnung komplett durch."""
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  SOFA SCORE BERECHNUNG".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    print(f"\nStart: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # =========================================================================
    # 1. SETUP & VALIDIERUNG
    # =========================================================================
    
    print("📁 Schritt 1: Setup & Validierung")
    print("-" * 70)
    
    # Erstelle Output-Verzeichnisse
    output_path = config.create_output_dirs("sofa")
    
    # Validiere Pfade
    if not config.validate_paths():
        print("\n❌ FEHLER: MIMIC-IV Pfade nicht konfiguriert!")
        print("   → Bitte passe die Pfade in src/baseline_models/config.py an.\n")
        sys.exit(1)
    
    # Validiere itemids
    print("\n🔍 Validiere itemid Mappings...")
    itemid_mappings.validate_itemids()
    
    # Teste SOFA-Berechnung
    print("\n🧪 Teste SOFA Score Berechnung...")
    if not calculator.validate_sofa_calculation():
        print("⚠️  Warnung: Einige Validierungs-Tests fehlgeschlagen!")
        response = input("Trotzdem fortfahren? (j/n): ")
        if response.lower() != 'j':
            sys.exit(1)
    
    print("\n✅ Setup abgeschlossen!\n")
    
    # =========================================================================
    # 2. DATEN LADEN
    # =========================================================================
    
    print("📊 Schritt 2: MIMIC-IV Daten laden")
    print("-" * 70)
    
    # SOFA für N zufällige ICU-Patienten
    # Nach offizieller Methodik: erste 24h nach ICU-Admission, worst values
    n_patients = 10  # Start mit 10 für schnellen Test
    random_state = 42  # Reproduzierbar
    
    try:
        # 1. Lade alle ICU Stays
        print("📋 Lade ICU Stays...")
        all_icustays = data_loader.load_icustays()
        print(f"  ✓ {len(all_icustays)} ICU Stays total")
        
        # 2. Filter: Mindestens 24h LOS (für valide 24h SOFA)
        icustays_24h = all_icustays[all_icustays['los'] >= 1.0]
        print(f"  ✓ {len(icustays_24h)} ICU Stays mit LOS ≥24h")
        
        # 3. Zufällige Auswahl von N Patienten
        import numpy as np
        np.random.seed(random_state)
        
        if len(icustays_24h) < n_patients:
            print(f"  ⚠️  Nur {len(icustays_24h)} Patienten verfügbar, nutze alle")
            sample_icustays = icustays_24h
        else:
            sample_icustays = icustays_24h.sample(n=n_patients, random_state=random_state)
            print(f"  ✓ {n_patients} zufällige Patienten ausgewählt (random_state={random_state})")
        
        # 4. Lade SOFA-Daten für diese Patienten (erste 24h, worst values)
        subject_ids = sample_icustays['subject_id'].unique().tolist()
        print(f"\n📊 Lade Daten für {len(subject_ids)} Patienten...")
        
        sofa_data = data_loader.load_sofa_data(
            subject_ids=subject_ids,
            time_window_hours=24  # Erste 24h nach ICU-Admission
        )
    except Exception as e:
        print(f"\n❌ FEHLER beim Laden: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    if len(sofa_data) == 0:
        print("\n❌ FEHLER: Keine Daten geladen!")
        sys.exit(1)
    
    print(f"\n✅ {len(sofa_data)} Patienten geladen\n")
    
    # =========================================================================
    # 3. DATENQUALITÄT PRÜFEN
    # =========================================================================
    
    print("🔍 Schritt 3: Datenqualität prüfen")
    print("-" * 70)
    
    required_cols = [
        'pao2_fio2_ratio', 'platelets', 'bilirubin',
        'map', 'gcs', 'creatinine'
    ]
    validation_stats = utils.validate_data(sofa_data, required_cols, "SOFA")
    
    # =========================================================================
    # 4. SOFA SCORES BERECHNEN
    # =========================================================================
    
    print("🧮 Schritt 4: SOFA Scores berechnen")
    print("-" * 70)
    
    try:
        sofa_results = calculator.calculate_sofa_batch(sofa_data)
        print(f"\n✅ SOFA Scores berechnet!\n")
    except Exception as e:
        print(f"\n❌ FEHLER bei Berechnung: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # =========================================================================
    # 5. PLAUSIBILITÄTS-CHECKS
    # =========================================================================
    
    print("✅ Schritt 5: Plausibilitäts-Checks")
    print("-" * 70)
    
    is_valid = utils.sanity_check_scores(
        sofa_results,
        score_column='sofa_total',
        valid_range=(0, 24),
        expected_mean_range=(2, 15),
        score_name="SOFA"
    )
    
    if not is_valid:
        print("⚠️  Warnung: Plausibilitäts-Checks haben Probleme gefunden!")
    
    # =========================================================================
    # 6. STATISTIKEN
    # =========================================================================
    
    print("📈 Schritt 6: Statistiken berechnen")
    print("-" * 70)
    
    component_cols = [
        'sofa_respiration', 'sofa_coagulation', 'sofa_liver',
        'sofa_cardiovascular', 'sofa_cns', 'sofa_renal'
    ]
    
    stats = utils.calculate_score_statistics(
        sofa_results,
        score_column='sofa_total',
        component_columns=component_cols
    )
    
    utils.print_statistics(stats, "SOFA")
    
    # =========================================================================
    # 7. ERGEBNISSE SPEICHERN
    # =========================================================================
    
    print("💾 Schritt 7: Ergebnisse speichern")
    print("-" * 70)
    
    try:
        utils.save_results(
            sofa_results,
            output_path,
            score_name="sofa",
            score_columns=['sofa_total'] + component_cols
        )
        
        utils.save_statistics(
            stats,
            output_path,
            score_name="SOFA",
            validation_stats=validation_stats
        )
        
        print("✅ Alle Ergebnisse gespeichert!\n")
        
    except Exception as e:
        print(f"\n❌ FEHLER beim Speichern: {e}")
        sys.exit(1)
    
    # =========================================================================
    # 8. ZUSAMMENFASSUNG
    # =========================================================================
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  SOFA SCORE - FERTIG!".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    print(f"\n📊 ZUSAMMENFASSUNG:")
    print("-" * 70)
    print(f"  Methodik:               Erste 24h nach ICU-Admission, worst values")
    print(f"  Patienten:              {len(sofa_results)}")
    print(f"  SOFA Score (Mittel):    {stats['mean']:.2f} ± {stats['std']:.2f}")
    print(f"  SOFA Score (Median):    {stats['median']:.1f}")
    print(f"  SOFA Score (Range):     {stats['min']:.0f} - {stats['max']:.0f}")
    print("-" * 70)
    
    print(f"\n📁 OUTPUT:")
    print(f"  {output_path}/sofa_scores.csv")
    print(f"  {output_path}/sofa_complete_data.csv")
    print(f"  {output_path}/sofa_statistics.txt")
    
    print(f"\n⏱️  Ende: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n✅ Erfolgreich abgeschlossen!\n")

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Abbruch durch Benutzer (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ UNERWARTETER FEHLER: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

