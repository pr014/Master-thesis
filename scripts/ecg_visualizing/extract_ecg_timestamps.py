"""Extract timestamps from ECG records (base_date, base_time from WFDB format).

Based on PhysioNet Usage Notes:
- ECG timestamps are from the machine's internal clock
- May not be synchronized with other MIMIC-IV databases
- Useful for linking ECG records to Clinical Database admissions
"""

import argparse
from pathlib import Path

import yaml

from src.data.ecg import extract_timestamps_from_directory, extract_timestamp_from_record


def get_default_output_dir() -> Path:
    """Get default output directory for ECG timestamps."""
    config_path = Path("configs/data/default_paths.yaml")
    base_dir = Path("outputs/ecg_visualizing/timestamps")
    
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            base_dir = Path(config.get('outputs', {}).get('ecg_visualizing_timestamps', str(base_dir)))
        except Exception:
            pass
    
    return base_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract timestamps (base_date, base_time) from ECG records",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract timestamps from all records in directory
  python scripts/ecg_visualizing/extract_ecg_timestamps.py --data_dir data/raw/demo/ecg/mimic-iv-ecg-demo --output outputs/ecg_visualizing/timestamps/timestamps.csv

  # Extract timestamp from single record
  python scripts/ecg_visualizing/extract_ecg_timestamps.py --base_path data/raw/demo/ecg/mimic-iv-ecg-demo/files/p10000032/s107143276/107143276

Note: Timestamps may not be synchronized with other MIMIC-IV databases.
See PhysioNet Usage Notes for limitations.
        """,
    )

    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--data_dir',
        type=str,
        help='Directory containing ECG records (.hea/.dat pairs)',
    )
    input_group.add_argument(
        '--base_path',
        type=str,
        help='Path to single ECG record (without .hea/.dat extension)',
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of records to process (when using --data_dir)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV path. If not specified for --data_dir, saves to outputs/ecg_visualizing/timestamps/',
    )

    args = parser.parse_args()

    try:
        if args.base_path:
            # Single record
            timestamp_info = extract_timestamp_from_record(args.base_path)
            print("Timestamp information:")
            for key, value in timestamp_info.items():
                print(f"  {key}: {value}")

            if args.output:
                import pandas as pd
                df = pd.DataFrame([timestamp_info])
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_path, index=False)
                print(f"\nSaved to {output_path}")

        else:
            # Directory
            if not args.output:
                # Auto-generate output path
                output_dir = get_default_output_dir()
                output_dir.mkdir(parents=True, exist_ok=True)
                data_dir_name = Path(args.data_dir).name
                limit_suffix = f"_limit{args.limit}" if args.limit else ""
                args.output = str(output_dir / f"{data_dir_name}_timestamps{limit_suffix}.csv")

            df = extract_timestamps_from_directory(
                data_dir=args.data_dir,
                limit=args.limit,
                output_csv=args.output,
            )
            print(f"\nExtracted timestamps from {len(df)} records")
            print(f"\nFirst few records:")
            print(df.head())

    except Exception as e:
        print(f"Error: {e}", flush=True)
        raise


if __name__ == '__main__':
    main()

