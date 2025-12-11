#!/usr/bin/env python3
"""
Script to upload checkpoints to LocalWorldModels Hugging Face repository.
Usage: python upload_checkpoint.py <checkpoint_path> [additional_paths...]
"""

import argparse
import os
from huggingface_hub import HfApi
import glob

def find_latest_step(checkpoint_path):
    files = glob.glob(os.path.join(checkpoint_path, "ckpts", "*.pt"))
    if len(files) == 0:
        return None
    return max([int(os.path.basename(f).split("step")[-1].split(".")[0]) for f in files])


def main():
    parser = argparse.ArgumentParser(
        description="Upload checkpoints to LocalWorldModels Hugging Face repository",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload single checkpoint
  python upload_checkpoint.py /path/to/denoiser_step0007000.pt
  
  # Upload multiple checkpoints simultaneously
  python upload_checkpoint.py /path/to/denoiser_step0007000.pt /path/to/denoiser_ema0.9999_step0007000.pt /path/to/misc_step0007000.pt
  
  # Upload with custom repository
  python upload_checkpoint.py /path/to/checkpoint.pt --repo-id "your-org/your-repo"
        """
    )
    
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="Path to the checkpoint directory to upload"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=-1,
        help="Step of the checkpoint to upload"
    )
    
    parser.add_argument(
        "--repo-id",
        default="LocalWorldModels/checkpoints",
        help="Hugging Face repository ID (default: LocalWorldModels/LocalWorldModels)"
    )

    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="checkpoints",
        help="Name of the checkpoint to upload"
    )

    
    args = parser.parse_args()
    
    # Convert to absolute paths
    checkpoint_path = os.path.abspath(args.checkpoint_dir) 
    if args.step == -1:
        args.step = find_latest_step(checkpoint_path)
    # Only upload denoiser_*.pt files (not misc_*.pt)
    all_checkpoints_files = [os.path.join(checkpoint_path, "ckpts", f"denoiser_step{args.step:07d}.pt"), os.path.join(checkpoint_path, "ckpts", f"denoiser_ema0.9999_step{args.step:07d}.pt")]
    # Validate checkpoint path
    for checkpoint_file in all_checkpoints_files:
        if not os.path.exists(checkpoint_file):
            return False, os.path.basename(checkpoint_file), f"Checkpoint file not found: {checkpoint_file}"
        filename = os.path.basename(checkpoint_file)
    
        api = HfApi()
        
        # Upload the file
        api.upload_file(
            path_or_fileobj=checkpoint_file,
            path_in_repo=f"checkpoints/{args.checkpoint_name}/{filename}",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"Upload checkpoint: {filename}"
        )

if __name__ == "__main__":
    main()
