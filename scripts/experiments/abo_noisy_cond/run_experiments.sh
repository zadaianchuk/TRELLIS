#!/bin/bash
#
# Run ABO Noisy Conditioning Experiments
#
# This script runs two experiments:
# 1. Baseline: Standard finetuning without noise
# 2. Noisy: RAE-style noisy conditioning finetuning
#
# Both experiments finetune the SS encoder on ABO train set for 30k iterations
# and can be evaluated on the ABO validation set.
#
# Usage:
#   ./run_experiments.sh [prepare|train_baseline|train_noisy|eval_baseline|eval_noisy|all]
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ABO_ROOT="${PROJECT_ROOT}/datasets/ABO"

# Output directories
BASELINE_OUTPUT="${PROJECT_ROOT}/outputs/experiments/abo_noisy_cond/baseline"
NOISY_OUTPUT="${PROJECT_ROOT}/outputs/experiments/abo_noisy_cond/noisy"

# Config files
BASELINE_CONFIG="${PROJECT_ROOT}/configs/experiments/abo_noisy_cond/finetune_ss_baseline.json"
NOISY_CONFIG="${PROJECT_ROOT}/configs/experiments/abo_noisy_cond/finetune_ss_noisy_cond.json"

# Number of GPUs (adjust based on your setup)
NUM_GPUS="${NUM_GPUS:-1}"

prepare_data() {
    echo "=========================================="
    echo "Preparing ABO dataset splits..."
    echo "=========================================="
    
    cd "${PROJECT_ROOT}"
    python scripts/experiments/abo_noisy_cond/prepare_abo_split.py \
        --abo_root "${ABO_ROOT}"
    
    echo "Data preparation complete!"
    echo "Train data: ${ABO_ROOT}/splits/train"
    echo "Val data: ${ABO_ROOT}/splits/val"
}

train_baseline() {
    echo "=========================================="
    echo "Training BASELINE (no noise)..."
    echo "=========================================="
    echo "Config: ${BASELINE_CONFIG}"
    echo "Output: ${BASELINE_OUTPUT}"
    echo "=========================================="
    
    cd "${PROJECT_ROOT}"
    mkdir -p "${BASELINE_OUTPUT}"
    
    if [ "${NUM_GPUS}" -gt 1 ]; then
        python -m torch.distributed.launch \
            --nproc_per_node=${NUM_GPUS} \
            train.py \
            --config "${BASELINE_CONFIG}" \
            --output_dir "${BASELINE_OUTPUT}" \
            --data_dir "${ABO_ROOT}/splits/train" \
            --num_gpus ${NUM_GPUS}
    else
        python train.py \
            --config "${BASELINE_CONFIG}" \
            --output_dir "${BASELINE_OUTPUT}" \
            --data_dir "${ABO_ROOT}/splits/train" \
            --num_gpus 1
    fi
    
    echo "Baseline training complete!"
}

train_noisy() {
    echo "=========================================="
    echo "Training NOISY (RAE-style noise injection)..."
    echo "=========================================="
    echo "Config: ${NOISY_CONFIG}"
    echo "Output: ${NOISY_OUTPUT}"
    echo "=========================================="
    
    cd "${PROJECT_ROOT}"
    mkdir -p "${NOISY_OUTPUT}"
    
    if [ "${NUM_GPUS}" -gt 1 ]; then
        python -m torch.distributed.launch \
            --nproc_per_node=${NUM_GPUS} \
            train.py \
            --config "${NOISY_CONFIG}" \
            --output_dir "${NOISY_OUTPUT}" \
            --data_dir "${ABO_ROOT}/splits/train" \
            --num_gpus ${NUM_GPUS}
    else
        python train.py \
            --config "${NOISY_CONFIG}" \
            --output_dir "${NOISY_OUTPUT}" \
            --data_dir "${ABO_ROOT}/splits/train" \
            --num_gpus 1
    fi
    
    echo "Noisy conditioning training complete!"
}

eval_baseline() {
    echo "=========================================="
    echo "Evaluating BASELINE on validation set..."
    echo "=========================================="
    
    cd "${PROJECT_ROOT}"
    
    # Find latest checkpoint
    LATEST_CKPT=$(ls -t "${BASELINE_OUTPUT}/ckpts/denoiser_ema_0_step"*.pt 2>/dev/null | head -1)
    if [ -z "${LATEST_CKPT}" ]; then
        LATEST_CKPT=$(ls -t "${BASELINE_OUTPUT}/ckpts/denoiser_step"*.pt 2>/dev/null | head -1)
    fi
    
    if [ -z "${LATEST_CKPT}" ]; then
        echo "No checkpoint found in ${BASELINE_OUTPUT}/ckpts/"
        exit 1
    fi
    
    echo "Using checkpoint: ${LATEST_CKPT}"
    
    python scripts/experiments/abo_noisy_cond/evaluate.py \
        --config "${BASELINE_CONFIG}" \
        --checkpoint "${LATEST_CKPT}" \
        --data_dir "${ABO_ROOT}/splits/val" \
        --output_dir "${BASELINE_OUTPUT}/eval" \
        --num_samples 50
    
    echo "Baseline evaluation complete!"
    echo "Results: ${BASELINE_OUTPUT}/eval"
}

eval_noisy() {
    echo "=========================================="
    echo "Evaluating NOISY on validation set..."
    echo "=========================================="
    
    cd "${PROJECT_ROOT}"
    
    # Find latest checkpoint
    LATEST_CKPT=$(ls -t "${NOISY_OUTPUT}/ckpts/denoiser_ema_0_step"*.pt 2>/dev/null | head -1)
    if [ -z "${LATEST_CKPT}" ]; then
        LATEST_CKPT=$(ls -t "${NOISY_OUTPUT}/ckpts/denoiser_step"*.pt 2>/dev/null | head -1)
    fi
    
    if [ -z "${LATEST_CKPT}" ]; then
        echo "No checkpoint found in ${NOISY_OUTPUT}/ckpts/"
        exit 1
    fi
    
    echo "Using checkpoint: ${LATEST_CKPT}"
    
    python scripts/experiments/abo_noisy_cond/evaluate.py \
        --config "${NOISY_CONFIG}" \
        --checkpoint "${LATEST_CKPT}" \
        --data_dir "${ABO_ROOT}/splits/val" \
        --output_dir "${NOISY_OUTPUT}/eval" \
        --num_samples 50
    
    echo "Noisy conditioning evaluation complete!"
    echo "Results: ${NOISY_OUTPUT}/eval"
}

print_usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  prepare         Prepare ABO dataset train/val splits"
    echo "  train_baseline  Train baseline model (no noise)"
    echo "  train_noisy     Train noisy conditioning model"
    echo "  eval_baseline   Evaluate baseline model on val set"
    echo "  eval_noisy      Evaluate noisy model on val set"
    echo "  all             Run all steps (prepare, train both, eval both)"
    echo ""
    echo "Environment variables:"
    echo "  NUM_GPUS        Number of GPUs to use (default: 1)"
}

# Main
case "${1:-}" in
    prepare)
        prepare_data
        ;;
    train_baseline)
        train_baseline
        ;;
    train_noisy)
        train_noisy
        ;;
    eval_baseline)
        eval_baseline
        ;;
    eval_noisy)
        eval_noisy
        ;;
    all)
        prepare_data
        train_baseline
        train_noisy
        eval_baseline
        eval_noisy
        ;;
    *)
        print_usage
        exit 1
        ;;
esac


