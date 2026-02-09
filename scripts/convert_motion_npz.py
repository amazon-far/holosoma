#!/usr/bin/env python3
"""
Convert motion npz files from numpy 2.x format to numpy 1.x compatible format.
This script loads an npz file and resaves it without using numpy._core module.
"""
import sys
import numpy as np
from pathlib import Path

def convert_npz(input_file: str, output_file: str = None):
    """Convert npz file to numpy 1.x compatible format."""
    input_path = Path(input_file)

    if output_file is None:
        output_file = str(input_path.parent / f"{input_path.stem}_converted.npz")

    print(f"Loading {input_file}...")

    # Load with allow_pickle for compatibility
    try:
        data = np.load(input_file, allow_pickle=True)
    except Exception as e:
        print(f"Error loading file: {e}")
        print("Trying with different numpy version...")
        # Try loading with fix_imports
        data = np.load(input_file, allow_pickle=True, fix_imports=True, encoding='latin1')

    # Extract all arrays and force conversion of string arrays
    data_dict = {}
    for key in data.files:
        arr = data[key]
        # For string arrays, convert to ensure numpy 1.x compatibility
        if arr.dtype.kind in ('U', 'S', 'O'):  # Unicode, bytes, or object
            # Convert to list and back to numpy array to avoid pickle issues
            arr = np.array(arr.tolist())
        data_dict[key] = arr

    print(f"Found {len(data_dict)} arrays: {list(data_dict.keys())}")

    # Save with numpy 1.x compatible format (without pickle for string arrays)
    print(f"Saving to {output_file}...")
    np.savez(output_file, **data_dict)

    print(f"✓ Conversion complete!")
    print(f"  Input:  {input_file}")
    print(f"  Output: {output_file}")

    # Verify the output
    print("\nVerifying output file...")
    test_data = np.load(output_file, allow_pickle=True)
    print(f"✓ Output file can be loaded successfully")
    print(f"  Arrays: {test_data.files}")

    return output_file

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_motion_npz.py <input_file.npz> [output_file.npz]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    convert_npz(input_file, output_file)
