# ABO Noisy Conditioning Experiment

This experiment tests whether adding RAE-style noise to conditioning features during fine-tuning improves model robustness. The experiment compares two training setups:

1. **Baseline**: Standard fine-tuning without noise
2. **Noisy**: RAE-style Gaussian noise injection on conditioning features

## Background

The [RAE (Representation Autoencoders)](https://github.com/bytetriper/RAE) paper shows that adding noise to latent representations during training can improve model robustness. This experiment applies a similar idea to the conditioning features in TRELLIS.

### Noise Injection Method

During training with noisy conditioning:
- Encode images using DINOv2 (as usual)
- Add Gaussian noise to the encoded features: `noisy_features = features + σ * ε`
- The noise scale σ is sampled uniformly from [0.0, 0.2]
- No noise is added during inference

## Setup

### Dataset

The experiment uses the ABO dataset with the following splits:
- **Train**: `/datasets/ABO/train.csv` (~4000 samples)
- **Val**: `/datasets/ABO/val.csv` (~486 samples)

### Training Configuration

Both experiments use:
- 30,000 iterations
- Batch size: 8 per GPU
- Learning rate: 1e-4
- Fine-tuning from: `pretrained_models/TRELLIS-image-large/ckpts/ss_flow_img_dit_L_16l8_fp16.pt`
- Sampling every 2000 steps
- Checkpoints every 5000 steps

## Running the Experiment

### 1. Prepare Data Splits

```bash
cd /shared/home/AZA0761/projects/TRELLIS
python scripts/experiments/abo_noisy_cond/prepare_abo_split.py \
    --abo_root ./datasets/ABO
```

This creates:
- `datasets/ABO/splits/train/` - Training data with symlinks
- `datasets/ABO/splits/val/` - Validation data with symlinks

### 2. Train Baseline (No Noise)

```bash
./scripts/experiments/abo_noisy_cond/train_baseline.sh [NUM_GPUS]
```

Or directly:
```bash
python train.py \
    --config configs/experiments/abo_noisy_cond/finetune_ss_baseline.json \
    --output_dir outputs/experiments/abo_noisy_cond/baseline \
    --data_dir datasets/ABO/splits/train \
    --num_gpus 1
```

### 3. Train Noisy (RAE-style)

```bash
./scripts/experiments/abo_noisy_cond/train_noisy.sh [NUM_GPUS]
```

Or directly:
```bash
python train.py \
    --config configs/experiments/abo_noisy_cond/finetune_ss_noisy_cond.json \
    --output_dir outputs/experiments/abo_noisy_cond/noisy \
    --data_dir datasets/ABO/splits/train \
    --num_gpus 1
```

### 4. Evaluate

```bash
# Evaluate baseline
python scripts/experiments/abo_noisy_cond/evaluate.py \
    --config configs/experiments/abo_noisy_cond/finetune_ss_baseline.json \
    --checkpoint outputs/experiments/abo_noisy_cond/baseline/ckpts/denoiser_ema_0_step030000.pt \
    --data_dir datasets/ABO/splits/val \
    --output_dir outputs/experiments/abo_noisy_cond/baseline/eval \
    --num_samples 50

# Evaluate noisy
python scripts/experiments/abo_noisy_cond/evaluate.py \
    --config configs/experiments/abo_noisy_cond/finetune_ss_noisy_cond.json \
    --checkpoint outputs/experiments/abo_noisy_cond/noisy/ckpts/denoiser_ema_0_step030000.pt \
    --data_dir datasets/ABO/splits/val \
    --output_dir outputs/experiments/abo_noisy_cond/noisy/eval \
    --num_samples 50
```

## File Structure

```
scripts/experiments/abo_noisy_cond/
├── README.md                 # This file
├── prepare_abo_split.py      # Creates train/val splits
├── evaluate.py               # Evaluation script
├── run_experiments.sh        # Master script for all steps
├── train_baseline.sh         # Train baseline only
└── train_noisy.sh            # Train noisy only

configs/experiments/abo_noisy_cond/
├── finetune_ss_baseline.json     # Baseline config
└── finetune_ss_noisy_cond.json   # Noisy conditioning config

outputs/experiments/abo_noisy_cond/
├── baseline/                 # Baseline training outputs
│   ├── ckpts/               # Checkpoints
│   ├── samples/             # Training samples
│   ├── tb_logs/             # TensorBoard logs
│   └── eval/                # Evaluation results
└── noisy/                    # Noisy training outputs
    ├── ckpts/
    ├── samples/
    ├── tb_logs/
    └── eval/
```

## Noise Configuration

The noisy conditioning trainer supports several noise schedules:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `cond_noise_enabled` | Enable/disable noise | `true` |
| `cond_noise_std` | Fixed noise std | `0.1` |
| `cond_noise_std_min` | Min std for uniform/lognormal | `0.0` |
| `cond_noise_std_max` | Max std for uniform/lognormal | `0.2` |
| `cond_noise_schedule` | Schedule type: `fixed`, `uniform`, `lognormal` | `uniform` |

## Expected Outcomes

The hypothesis is that adding noise during training will:
1. Make the model more robust to variations in conditioning features
2. Potentially improve generalization to out-of-distribution conditioning
3. Allow recovery from noisy features during inference

Compare:
- Training curves (loss, samples) between baseline and noisy
- Quality of generated samples on the validation set
- Robustness when adding noise to conditioning features at inference time

## Implementation Details

The noise injection is implemented in:
- `trellis/trainers/flow_matching/mixins/noisy_image_conditioned.py` - New mixin class
- `trellis/trainers/flow_matching/sparse_flow_matching.py` - New trainer class `NoisyImageConditionedSparseFlowMatchingCFGTrainer`

Key implementation choices:
- Noise is added **after** DINOv2 encoding and layer normalization
- Noise is only added during training, not inference
- The noise scale is sampled per-batch for variety


