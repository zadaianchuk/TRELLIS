#!/usr/bin/env python3
"""
Prepare ABO dataset split for training and validation.

This script creates separate data directories for training and validation
based on the train.csv and val.csv files, using symlinks to save disk space.

Usage:
    python prepare_abo_split.py --abo_root /path/to/datasets/ABO
"""
import os
import sys
import argparse
import pandas as pd
from pathlib import Path


def create_split_metadata(abo_root: str, split_csv: str, output_dir: str, split_name: str):
    """
    Create metadata.csv for a specific split by filtering the original metadata.
    
    Args:
        abo_root: Root directory of ABO dataset
        split_csv: Path to train.csv or val.csv
        output_dir: Output directory for the split
        split_name: Name of the split ('train' or 'val')
    """
    # Read the full metadata
    metadata_path = os.path.join(abo_root, 'metadata.csv')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.csv not found at {metadata_path}")
    
    metadata = pd.read_csv(metadata_path)
    print(f"Loaded metadata with {len(metadata)} entries")
    
    # Read the split CSV
    split_df = pd.read_csv(split_csv)
    split_sha256s = set(split_df['sha256'].values)
    print(f"Split CSV ({split_name}) has {len(split_sha256s)} entries")
    
    # Filter metadata to only include entries in the split
    filtered_metadata = metadata[metadata['sha256'].isin(split_sha256s)]
    print(f"Filtered metadata has {len(filtered_metadata)} entries")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save filtered metadata
    output_metadata_path = os.path.join(output_dir, 'metadata.csv')
    filtered_metadata.to_csv(output_metadata_path, index=False)
    print(f"Saved filtered metadata to {output_metadata_path}")
    
    # Create symlinks to shared data directories
    shared_dirs = ['ss_latents', 'renders_cond', 'voxels', 'renders', 'raw', 'features', 'latents']
    for dir_name in shared_dirs:
        src_path = os.path.join(abo_root, dir_name)
        dst_path = os.path.join(output_dir, dir_name)
        
        if os.path.exists(src_path):
            if os.path.exists(dst_path) or os.path.islink(dst_path):
                os.remove(dst_path) if os.path.islink(dst_path) else None
            
            if not os.path.exists(dst_path):
                os.symlink(src_path, dst_path)
                print(f"Created symlink: {dst_path} -> {src_path}")
        else:
            print(f"Warning: Source directory {src_path} does not exist")
    
    return len(filtered_metadata)


def main():
    parser = argparse.ArgumentParser(description='Prepare ABO dataset splits for training and validation')
    parser.add_argument('--abo_root', type=str, default='./datasets/ABO',
                        help='Root directory of ABO dataset')
    parser.add_argument('--train_csv', type=str, default=None,
                        help='Path to train.csv (default: {abo_root}/train.csv)')
    parser.add_argument('--val_csv', type=str, default=None,
                        help='Path to val.csv (default: {abo_root}/val.csv)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Base output directory (default: {abo_root}/splits)')
    args = parser.parse_args()
    
    abo_root = os.path.abspath(args.abo_root)
    
    if not os.path.exists(abo_root):
        raise FileNotFoundError(f"ABO root directory not found: {abo_root}")
    
    train_csv = args.train_csv or os.path.join(abo_root, 'train.csv')
    val_csv = args.val_csv or os.path.join(abo_root, 'val.csv')
    output_base = args.output_dir or os.path.join(abo_root, 'splits')
    
    print(f"ABO root: {abo_root}")
    print(f"Train CSV: {train_csv}")
    print(f"Val CSV: {val_csv}")
    print(f"Output base: {output_base}")
    print()
    
    # Create train split
    print("=" * 60)
    print("Creating training split...")
    print("=" * 60)
    train_output = os.path.join(output_base, 'train')
    train_count = create_split_metadata(abo_root, train_csv, train_output, 'train')
    print()
    
    # Create val split
    print("=" * 60)
    print("Creating validation split...")
    print("=" * 60)
    val_output = os.path.join(output_base, 'val')
    val_count = create_split_metadata(abo_root, val_csv, val_output, 'val')
    print()
    
    print("=" * 60)
    print("Summary:")
    print(f"  Train: {train_count} samples -> {train_output}")
    print(f"  Val: {val_count} samples -> {val_output}")
    print("=" * 60)


if __name__ == '__main__':
    main()


