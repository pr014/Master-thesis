"""CLI tool to create timestamp mapping from original .hea/.dat files.

This script extracts timestamps from original ECG files and creates a CSV mapping
that can be used when loading preprocessed .npy files.
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import modules - need to import in order to handle relative imports
import src.data.ecg.ecg_loader
import src.data.ecg.ecg_metadata

# Now import timestamp_mapping (it will use the already imported modules)
from src.data.ecg.timestamp_mapping import (
    create_timestamp_mapping,
    get_timestamp_mapping_path,
    auto_detect_original_path,
    load_timestamp_mapping,
)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Create timestamp mapping from original .hea/.dat ECG files"
    )
    parser.add_argument(
        "--original_dir",
        type=str,
        help="Path to directory containing original .hea/.dat files",
    )
    parser.add_argument(
        "--preprocessed_dir",
        type=str,
        help="Path to preprocessed data directory (for auto-detection of original_dir)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for CSV mapping file (default: auto-generated in data/labeling/labels_csv/)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of records to process (for testing)",
    )
    
    args = parser.parse_args()
    
    # Determine original data directory
    original_data_dir = args.original_dir
    
    if original_data_dir is None:
        if args.preprocessed_dir:
            # Try auto-detection
            original_data_dir_path = auto_detect_original_path(args.preprocessed_dir)
            if original_data_dir_path:
                original_data_dir = str(original_data_dir_path)
            else:
                print("Error: Could not auto-detect original data directory.")
                print(f"  Preprocessed directory: {args.preprocessed_dir}")
                print("  Please specify --original_dir manually")
                return 1
        else:
            print("Error: Must specify either --original_dir or --preprocessed_dir")
            parser.print_help()
            return 1
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Use preprocessed_dir if available, otherwise use original_dir
        data_dir = args.preprocessed_dir or original_data_dir
        output_path = str(get_timestamp_mapping_path(data_dir))
    
    # Create mapping
    try:
        print("=" * 60)
        print("Creating Timestamp Mapping")
        print("=" * 60)
        print(f"Original data directory: {original_data_dir}")
        print(f"Output path: {output_path}")
        if args.limit:
            print(f"Limit: {args.limit} records (testing mode)")
        print("=" * 60)
        
        df = create_timestamp_mapping(
            original_data_dir=original_data_dir,
            output_path=output_path,
            limit=args.limit,
        )
        
        # Verify the mapping was created correctly
        print("\n" + "=" * 60)
        print("Verifying timestamp mapping...")
        print("=" * 60)
        try:
            loaded_mapping = load_timestamp_mapping(output_path)
            print(f"✅ Timestamp mapping created successfully!")
            print(f"   Total entries: {len(loaded_mapping):,}")
            print(f"   File: {output_path}")
            print("\nFirst 5 entries:")
            for i, (base_path, timestamps) in enumerate(list(loaded_mapping.items())[:5]):
                print(f"   {base_path}: {timestamps['base_date']} {timestamps['base_time']}")
        except Exception as e:
            print(f"⚠️  Warning: Could not verify mapping: {e}")
        
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n❌ Error creating timestamp mapping: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

