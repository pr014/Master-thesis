"""Baseline preprocessing pipeline for ECG datasets.

This script applies the standardized baseline preprocessing pipeline to any ECG dataset:
1. Loads all ECG files from the source directory
2. Applies preprocessing pipeline (resample → filter → segment → normalize)
3. Saves preprocessed signals as NumPy arrays (.npy files)
4. Preserves directory structure and metadata

The preprocessed dataset can then be used for training all models with consistent preprocessing.

Pipeline (filtering before resampling to avoid aliasing):
- Bandpass filtering (0.5-50 Hz, Butterworth 4th order)
- Notch filtering (60 Hz for US power line frequency)
- Resampling to 500 Hz
- Segmentation to 10 seconds (5000 samples)
- Z-score normalization per lead

Supports large datasets (60k+ ECGs) with multiprocessing.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import numpy as np
import wfdb
from multiprocessing import Pool, cpu_count
from functools import partial

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocessing import preprocess_ecg_signal
from src.data.ecg import build_demo_index


def load_ecg_record(base_path: Path) -> Tuple[np.ndarray, float, Optional[str], Optional[str]]:
    """Load ECG record from PhysioNet format.
    
    Args:
        base_path: Path to ECG record (without .hea/.dat extension).
    
    Returns:
        Tuple of (signal, sampling_rate, base_date, base_time).
        signal: shape (T, C) where T is time samples, C is channels/leads.
    """
    record = wfdb.rdrecord(str(base_path))
    x = record.p_signal  # shape: (T, C) as float64
    fs = float(record.fs)
    base_date = getattr(record, 'base_date', None)
    base_time = getattr(record, 'base_time', None)
    return x, fs, base_date, base_time


def preprocess_and_save(
    base_path: Path,
    source_dir: Path,
    output_dir: Path,
    target_fs: float = 500.0,
    window_seconds: float = 10.0,
    filter_lowcut: float = 0.5,
    filter_highcut: float = 50.0,
    filter_order: int = 4,
    notch_freq: float = 60.0,
    normalize: bool = True,
    skip_existing: bool = False,
) -> Dict:
    """Preprocess a single ECG record and save it.
    
    Args:
        base_path: Path to ECG record (without .hea/.dat extension).
        source_dir: Source directory root (for computing relative paths).
        output_dir: Output directory root.
        target_fs: Target sampling rate in Hz.
        window_seconds: Target window length in seconds.
        filter_lowcut: Low cutoff frequency in Hz.
        filter_highcut: High cutoff frequency in Hz.
        filter_order: Butterworth filter order.
        normalize: Whether to apply Z-score normalization.
    
    Returns:
        Dictionary with processing results (success, error message, etc.).
    """
    try:
        # Load ECG record
        x, fs, base_date, base_time = load_ecg_record(base_path)
        
        # Apply preprocessing pipeline
        x_processed, effective_fs = preprocess_ecg_signal(
            x=x,
            fs=fs,
            target_fs=target_fs,
            window_seconds=window_seconds,
            filter_lowcut=filter_lowcut,
            filter_highcut=filter_highcut,
            filter_order=filter_order,
            notch_freq=notch_freq,
            normalize=normalize,
        )
        
        # Determine output path (preserve relative structure from source_dir)
        try:
            # Get relative path from source_dir
            rel_path = base_path.relative_to(source_dir)
            # Create output path preserving directory structure
            output_path = output_dir / rel_path.with_suffix('.npy')
        except ValueError:
            # If base_path is not relative to source_dir, use just the filename
            output_path = output_dir / (base_path.name + ".npy")
        
        # Skip if file already exists (for resume functionality)
        if skip_existing and output_path.exists():
            return {
                'success': True,
                'output_path': str(output_path),
                'skipped': True,
                'metadata': {'base_path': str(base_path)},
            }
        
        # Create parent directories if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save preprocessed signal
        np.save(output_path, x_processed)
        
        # Save metadata (optional, for reference)
        metadata = {
            'base_path': str(base_path),
            'original_fs': fs,
            'effective_fs': effective_fs,
            'original_shape': x.shape,
            'processed_shape': x_processed.shape,
            'base_date': base_date,
            'base_time': base_time,
        }
        
        return {
            'success': True,
            'output_path': str(output_path),
            'metadata': metadata,
            'skipped': False,
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'base_path': str(base_path),
        }


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(
        description="Apply baseline preprocessing pipeline to ECG dataset"
    )
    parser.add_argument(
        '--source',
        type=str,
        default=r"D:\MA\data\mimic-iv-ecg\icustay_ecgs_24h\files",
        help='Source directory containing ECG files (.hea/.dat). '
             'Can be icustay_ecgs_24h/files or icustay_ecgs/files. '
             'For icustay_ecgs_24h, output will be automatically set to preprocessed_24h_1'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for preprocessed signals (.npy files). '
             'If not specified, will be set automatically based on source path. '
             'Example: icustay_ecgs_24h/files -> icustay_ecgs_24h/preprocessed_24h_1'
    )
    parser.add_argument(
        '--target-fs',
        type=float,
        default=500.0,
        help='Target sampling rate in Hz (default: 500.0)'
    )
    parser.add_argument(
        '--window-seconds',
        type=float,
        default=10.0,
        help='Target window length in seconds (default: 10.0)'
    )
    parser.add_argument(
        '--filter-lowcut',
        type=float,
        default=0.5,
        help='Low cutoff frequency in Hz (default: 0.5)'
    )
    parser.add_argument(
        '--filter-highcut',
        type=float,
        default=50.0,
        help='High cutoff frequency in Hz (default: 50.0)'
    )
    parser.add_argument(
        '--filter-order',
        type=int,
        default=4,
        help='Butterworth filter order (default: 4)'
    )
    parser.add_argument(
        '--notch-freq',
        type=float,
        default=60.0,
        help='Notch filter frequency in Hz for power line removal (default: 60.0 for US)'
    )
    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='Skip Z-score normalization (default: normalize is enabled)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of files to process (for testing)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: number of CPU cores)'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip files that already exist in output directory (for resume)'
    )
    
    args = parser.parse_args()
    
    # Paths
    source_dir = Path(args.source)
    
    # Auto-detect output directory if not specified
    if args.output is None:
        # If source is in icustay_ecgs_24h/files, use icustay_ecgs_24h/preprocessed_24h_1
        source_str = str(source_dir).replace("\\", "/")
        if "icustay_ecgs_24h/files" in source_str:
            # Replace /files with /preprocessed_24h_1
            output_str = source_str.replace("/files", "/preprocessed_24h_1")
            output_dir = Path(output_str)
        elif "icustay_ecgs/files" in source_str and "24h" not in source_str:
            # For icustay_ecgs (without 24h), use icustay_ecgs/preprocessed
            output_str = source_str.replace("/files", "/preprocessed")
            output_dir = Path(output_str)
        elif source_dir.name == "files" and "icustay_ecgs_24h" in str(source_dir.parent):
            # If source_dir is .../icustay_ecgs_24h/files, use .../icustay_ecgs_24h/preprocessed_24h_1
            output_dir = source_dir.parent / "preprocessed_24h_1"
        elif source_dir.name == "files" and "icustay_ecgs" in str(source_dir.parent):
            # If source_dir is .../icustay_ecgs/files, use .../icustay_ecgs/preprocessed
            output_dir = source_dir.parent / "preprocessed"
        else:
            # Default: add _preprocessed suffix to source directory
            output_dir = Path(str(source_dir) + "_preprocessed")
    else:
        output_dir = Path(args.output)
    
    # Verify source directory
    if not source_dir.exists():
        print(f"ERROR: Source directory not found: {source_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("ECG Dataset Preprocessing (Baseline Pipeline)")
    print("=" * 70)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Target sampling rate: {args.target_fs} Hz")
    print(f"Window length: {args.window_seconds} seconds")
    print(f"Bandpass filter: {args.filter_lowcut}-{args.filter_highcut} Hz (order {args.filter_order})")
    print(f"Notch filter: {args.notch_freq} Hz (US power line frequency)")
    print(f"Normalization: {'enabled' if not args.no_normalize else 'disabled'}")
    print("=" * 70)
    
    # Find all ECG files
    print("\nScanning for ECG files...")
    records = build_demo_index(data_dir=str(source_dir), limit=args.limit)
    print(f"Found {len(records):,} ECG records")
    
    if len(records) == 0:
        print("No ECG files found. Exiting.")
        return
    
    # Determine number of workers
    num_workers = args.num_workers if args.num_workers is not None else cpu_count()
    print(f"\nUsing {num_workers} parallel workers")
    
    # Process records
    print("\nProcessing ECG records...")
    
    # Create partial function with fixed parameters
    process_func = partial(
        preprocess_and_save,
        source_dir=source_dir,
        output_dir=output_dir,
        target_fs=args.target_fs,
        window_seconds=args.window_seconds,
        filter_lowcut=args.filter_lowcut,
        filter_highcut=args.filter_highcut,
        filter_order=args.filter_order,
        notch_freq=args.notch_freq,
        normalize=not args.no_normalize,
        skip_existing=args.skip_existing,
    )
    
    # Prepare arguments for parallel processing
    base_paths = [Path(record["base_path"]) for record in records]
    
    # Process with multiprocessing
    results = []
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    if num_workers > 1:
        # Parallel processing
        with Pool(processes=num_workers) as pool:
            # Use imap for progress tracking
            for idx, result in enumerate(pool.imap(process_func, base_paths), 1):
                results.append(result)
                
                if result.get('skipped', False):
                    skipped_count += 1
                elif result['success']:
                    success_count += 1
                else:
                    error_count += 1
                    if error_count <= 10:  # Print first 10 errors
                        base_path = result.get('base_path', 'unknown')
                        print(f"  ERROR processing {Path(base_path).name}: {result.get('error', 'Unknown error')}")
                
                # Progress update every 100 records
                if idx % 100 == 0:
                    print(f"  Processed {idx:,}/{len(records):,} records... "
                          f"(success: {success_count:,}, skipped: {skipped_count:,}, errors: {error_count:,})")
    else:
        # Sequential processing (for debugging or single-core systems)
        for idx, base_path in enumerate(base_paths, 1):
            result = process_func(base_path)
            results.append(result)
            
            if result.get('skipped', False):
                skipped_count += 1
            elif result['success']:
                success_count += 1
            else:
                error_count += 1
                if error_count <= 10:
                    print(f"  ERROR processing {base_path.name}: {result.get('error', 'Unknown error')}")
            
            # Progress update every 100 records
            if idx % 100 == 0:
                print(f"  Processed {idx:,}/{len(records):,} records... "
                      f"(success: {success_count:,}, skipped: {skipped_count:,}, errors: {error_count:,})")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Preprocessing Summary")
    print("=" * 70)
    print(f"Total records: {len(records):,}")
    print(f"Successfully processed: {success_count:,}")
    if skipped_count > 0:
        print(f"Skipped (already exist): {skipped_count:,}")
    print(f"Errors: {error_count:,}")
    print(f"Output directory: {output_dir}")
    
    # Check output shape (sample a few files)
    if success_count > 0:
        print("\nVerifying output files...")
        sample_results = [r for r in results if r.get('success')]
        if sample_results:
            sample_meta = sample_results[0].get('metadata', {})
            processed_shape = sample_meta.get('processed_shape')
            target_length = int(args.window_seconds * args.target_fs)
            print(f"  Expected shape: ({target_length}, num_leads)")
            if processed_shape:
                print(f"  Sample processed shape: {processed_shape}")
                if processed_shape[0] == target_length:
                    print("  [OK] Output shape matches expected length")
                else:
                    print(f"  [WARNING] Output shape mismatch!")
    
    print("\n" + "=" * 70)
    print("Preprocessing complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()