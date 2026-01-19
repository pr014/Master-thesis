"""Visualize ECG records from various sources with configurable paths."""

import argparse
from pathlib import Path
from typing import Optional

import yaml

from src.data.ecg import ECGDemoDataset, build_demo_index
from src.visualization.plot_ecg import plot_12lead_ecg, plot_single_lead


def extract_patient_id_from_path(base_path: str) -> Optional[str]:
    """Extract patient ID (subject_id) from ECG record path.
    
    Args:
        base_path: Path like 'data/.../files/p10000032/s107143276/107143276'
        
    Returns:
        Patient ID (e.g., '10000032') or None if not found
    """
    path_parts = Path(base_path).parts
    for i, part in enumerate(path_parts):
        if part.startswith('p') and part[1:].isdigit():
            # Extract numeric part after 'p'
            return part[1:]
    return None


def get_default_output_dir(subdir: str = "") -> Path:
    """Get default output directory for ECG visualizations.
    
    Args:
        subdir: Optional subdirectory (e.g., '12lead', 'single_lead')
        
    Returns:
        Path to output directory
    """
    config_path = Path("configs/data/default_paths.yaml")
    base_dir = Path("outputs/ecg_visualizing")
    
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            base_dir = Path(config.get('outputs', {}).get('ecg_visualizing', str(base_dir)))
        except Exception:
            pass
    
    if subdir:
        return base_dir / subdir
    return base_dir


def visualize_from_wfdb(
    base_path: str,
    window_seconds: Optional[float] = None,
    lead_idx: Optional[int] = None,
    output_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Visualize an ECG record from a WFDB base path.

    Args:
        base_path: Path to ECG record (without .hea/.dat extension)
        window_seconds: Optional window length in seconds (from start)
        lead_idx: If specified, plot only this lead (0-indexed)
        output_path: Optional path to save the plot
        show: Whether to display the plot
    """
    try:
        import wfdb  # type: ignore
    except ImportError as e:
        raise RuntimeError("wfdb is required. Install with: pip install wfdb") from e

    # Load record
    record = wfdb.rdrecord(base_path)
    signal = record.p_signal  # (T, C)
    fs = float(record.fs)
    lead_names = record.sig_name

    # Apply windowing if requested
    if window_seconds is not None:
        window_samples = int(window_seconds * fs)
        signal = signal[:window_samples]

    # Extract record name for title
    record_name = Path(base_path).name

    # Plot
    if lead_idx is not None:
        plot_single_lead(
            signal=signal,
            fs=fs,
            lead_idx=lead_idx,
            lead_name=lead_names[lead_idx] if lead_idx < len(lead_names) else None,
            title=f"ECG Record: {record_name}",
            output_path=output_path,
            show=show,
        )
    else:
        plot_12lead_ecg(
            signal=signal,
            fs=fs,
            lead_names=lead_names,
            title=f"ECG Record: {record_name}",
            output_path=output_path,
            show=show,
        )


def visualize_from_directory(
    data_dir: str,
    record_idx: int = 0,
    limit: Optional[int] = None,
    window_seconds: Optional[float] = None,
    lead_idx: Optional[int] = None,
    output_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Visualize an ECG record from a directory of records.

    Args:
        data_dir: Directory containing ECG records (.hea/.dat pairs)
        record_idx: Index of the record to visualize (default: 0)
        limit: Optional limit on number of records to discover
        window_seconds: Optional window length in seconds
        lead_idx: If specified, plot only this lead
        output_path: Optional path to save the plot
        show: Whether to display the plot
    """
    # Discover records
    records = build_demo_index(data_dir=data_dir, limit=limit)

    if record_idx >= len(records):
        raise ValueError(f"Record index {record_idx} out of range. Found {len(records)} records.")

    # Load record using dataset
    dataset = ECGDemoDataset(records, window_seconds=window_seconds)
    item = dataset[record_idx]

    signal = item["signal"]
    fs = item["meta"]["fs"]
    base_path = item["meta"]["base_path"]
    record_name = Path(base_path).name

    # Default lead names for 12-lead
    lead_names = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    if signal.shape[1] <= len(lead_names):
        lead_names = lead_names[:signal.shape[1]]
    else:
        lead_names = [f"Lead {i+1}" for i in range(signal.shape[1])]

    # Plot
    if lead_idx is not None:
        plot_single_lead(
            signal=signal,
            fs=fs,
            lead_idx=lead_idx,
            lead_name=lead_names[lead_idx] if lead_idx < len(lead_names) else None,
            title=f"ECG Record: {record_name}",
            output_path=output_path,
            show=show,
        )
    else:
        plot_12lead_ecg(
            signal=signal,
            fs=fs,
            lead_names=lead_names,
            title=f"ECG Record: {record_name}",
            output_path=output_path,
            show=show,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize ECG records with configurable paths",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize first record from directory
  python scripts/ecg_visualizing/visualize_ecg.py --data_dir data/raw/demo/ecg/mimic-iv-ecg-demo

  # Visualize specific record by base path
  python scripts/ecg_visualizing/visualize_ecg.py --base_path data/raw/demo/ecg/mimic-iv-ecg-demo/files/p10000032/s107143276/107143276

  # Visualize with windowing and save to file
  python scripts/ecg_visualizing/visualize_ecg.py --data_dir data/raw/demo/ecg/mimic-iv-ecg-demo --window_seconds 5 --output outputs/ecg_visualizing/ecg.png

  # Visualize single lead only
  python scripts/ecg_visualizing/visualize_ecg.py --data_dir data/raw/demo/ecg/mimic-iv-ecg-demo --lead_idx 0
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
        help='Direct path to ECG record base (without .hea/.dat extension)',
    )

    # Visualization options
    parser.add_argument(
        '--record_idx',
        type=int,
        default=0,
        help='Index of record to visualize when using --data_dir (default: 0)',
    )
    parser.add_argument(
        '--window_seconds',
        type=float,
        default=None,
        help='Window length in seconds (from start). If not specified, shows full signal.',
    )
    parser.add_argument(
        '--lead_idx',
        type=int,
        default=None,
        help='If specified, plot only this lead (0-indexed). Otherwise plot all leads.',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path to save the plot. If not specified, saves to outputs/ecg_visualizing/ with auto-generated name.',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (auto-generates filename if --output not specified)',
    )
    parser.add_argument(
        '--no_show',
        action='store_true',
        help='Do not display the plot (useful when only saving)',
    )

    args = parser.parse_args()

    show = not args.no_show

    # Determine output path
    output_path = args.output
    if output_path is None and not show:
        # Auto-generate output path if --no_show but no --output specified
        output_dir = Path(args.output_dir) if args.output_dir else get_default_output_dir()
        if args.lead_idx is not None:
            output_dir = output_dir / "single_lead"
        else:
            output_dir = output_dir / "12lead"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename based on patient ID
        patient_id = None
        if args.base_path:
            patient_id = extract_patient_id_from_path(args.base_path)
            if not patient_id:
                # Fallback to record name if patient ID not found
                patient_id = Path(args.base_path).name
        else:
            # Load record to get base_path and extract patient ID
            records = build_demo_index(data_dir=args.data_dir, limit=args.record_idx + 1)
            if records and args.record_idx < len(records):
                base_path = records[args.record_idx]["base_path"]
                patient_id = extract_patient_id_from_path(base_path)
                if not patient_id:
                    # Fallback to record name
                    patient_id = Path(base_path).name
        
        if not patient_id:
            patient_id = f"record_{args.record_idx}"
        
        suffix = f"_lead{args.lead_idx}" if args.lead_idx is not None else ""
        window_suffix = f"_window{args.window_seconds}s" if args.window_seconds else ""
        output_path = str(output_dir / f"p{patient_id}{suffix}{window_suffix}.png")

    try:
        if args.base_path:
            visualize_from_wfdb(
                base_path=args.base_path,
                window_seconds=args.window_seconds,
                lead_idx=args.lead_idx,
                output_path=output_path,
                show=show,
            )
        else:
            visualize_from_directory(
                data_dir=args.data_dir,
                record_idx=args.record_idx,
                window_seconds=args.window_seconds,
                lead_idx=args.lead_idx,
                output_path=output_path,
                show=show,
            )
    except Exception as e:
        print(f"Error: {e}", flush=True)
        raise


if __name__ == '__main__':
    main()

