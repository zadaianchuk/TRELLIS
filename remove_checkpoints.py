#!/usr/bin/env python3
"""Remove checkpoint files by specifying step numbers."""

import argparse
import os
from pathlib import Path


def remove_checkpoints(ckpt_dir: str, steps: list[int], dry_run: bool = False):
    """Remove all checkpoint files matching the specified steps."""
    ckpt_path = Path(ckpt_dir)
    
    if not ckpt_path.exists():
        print(f"Error: Directory {ckpt_dir} does not exist")
        return
    
    removed = []
    not_found = []
    
    for step in steps:
        step_str = f"step{step:07d}"  # Format as step0005000
        
        # Find all files matching this step
        matching_files = list(ckpt_path.glob(f"*_{step_str}.pt"))
        
        if not matching_files:
            not_found.append(step)
            continue
        
        for f in matching_files:
            if dry_run:
                print(f"[DRY RUN] Would remove: {f}")
            else:
                f.unlink()
                print(f"Removed: {f}")
            removed.append(f.name)
    
    print(f"\n{'Would remove' if dry_run else 'Removed'} {len(removed)} files")
    if not_found:
        print(f"No files found for steps: {not_found}")


def main():
    parser = argparse.ArgumentParser(description="Remove checkpoint files by step numbers")
    parser.add_argument(
        "ckpt_dir",
        type=str,
        help="Path to the checkpoint directory"
    )
    parser.add_argument(
        "steps",
        type=int,
        nargs="+",
        help="Step numbers to remove (e.g., 5000 10000 15000)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without actually deleting"
    )
    
    args = parser.parse_args()
    remove_checkpoints(args.ckpt_dir, args.steps, args.dry_run)


if __name__ == "__main__":
    main()

