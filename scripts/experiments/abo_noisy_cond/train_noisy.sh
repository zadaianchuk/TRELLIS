#!/bin/bash
#
# Train SS encoder with RAE-style noisy conditioning on ABO
#
# Usage:
#   ./train_noisy.sh [NUM_GPUS]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

NUM_GPUS="${1:-1}"
ABO_ROOT="${PROJECT_ROOT}/datasets/ABO"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/experiments/abo_noisy_cond/noisy"
CONFIG="${PROJECT_ROOT}/configs/experiments/abo_noisy_cond/finetune_ss_noisy_cond.json"

echo "=========================================="
echo "ABO SS Finetuning - NOISY (RAE-style)"
echo "=========================================="
echo "Project root: ${PROJECT_ROOT}"
echo "ABO root: ${ABO_ROOT}"
echo "Output: ${OUTPUT_DIR}"
echo "Config: ${CONFIG}"
echo "GPUs: ${NUM_GPUS}"
echo ""
echo "Noise settings (from config):"
echo "  - Schedule: uniform"
echo "  - Std range: [0.0, 0.2]"
echo "=========================================="

# Check if splits exist, if not create them
if [ ! -d "${ABO_ROOT}/splits/train" ]; then
    echo "Creating train/val splits..."
    cd "${PROJECT_ROOT}"
    python scripts/experiments/abo_noisy_cond/prepare_abo_split.py --abo_root "${ABO_ROOT}"
fi

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

echo ""
echo "Starting training with noisy conditioning..."
echo ""

python train.py \
    --config "${CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "${ABO_ROOT}/splits/train" \
    --num_gpus ${NUM_GPUS}

echo ""
echo "Training complete!"
echo "Output: ${OUTPUT_DIR}"


