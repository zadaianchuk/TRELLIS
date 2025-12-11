# LoRA Fine-tuning Tutorial for TRELLIS

This tutorial covers how to use **Low-Rank Adaptation (LoRA)** for efficient fine-tuning of TRELLIS 3D generation models. LoRA allows you to adapt large models to your specific data with significantly fewer trainable parameters and reduced memory requirements.

## Table of Contents

1. [What is LoRA?](#what-is-lora)
2. [When to Use LoRA](#when-to-use-lora)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Training with LoRA](#training-with-lora)
6. [LoRA Parameters Explained](#lora-parameters-explained)
7. [Using Trained LoRA Weights](#using-trained-lora-weights)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)

---

## What is LoRA?

**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning technique that:

- Freezes the pretrained model weights
- Injects trainable low-rank matrices into attention layers
- Reduces trainable parameters by **~95-99%** compared to full fine-tuning
- Significantly lowers GPU memory requirements
- Produces small checkpoint files (typically a few MB vs. GB)

In TRELLIS, LoRA is applied to the attention projection layers:
- `to_q` - Query projection (cross-attention)
- `to_kv` - Key-Value projection (cross-attention)
- `to_out` - Output projection
- `to_qkv` - Combined QKV projection (self-attention)

---

## When to Use LoRA

**Use LoRA when:**
- You have limited GPU memory
- You want to fine-tune on a small-to-medium custom dataset
- You need to create multiple specialized models from the same base
- You want faster training with lower resource requirements
- You need portable/shareable model adaptations

**Use full fine-tuning when:**
- You have abundant GPU resources
- You need maximum model capacity
- Your target domain is significantly different from training data
- You're doing large-scale training

---

## Installation

LoRA support in TRELLIS requires the `peft` library:

```bash
pip install peft
```

Verify installation:

```python
from trellis.utils.lora_utils import PEFT_AVAILABLE
print(f"PEFT available: {PEFT_AVAILABLE}")
```

---

## Configuration

### Config File Structure

LoRA training uses specialized config files. Here's the key structure:

```json
{
    "models": {
        "denoiser": {
            "name": "ElasticSLatFlowModel",
            "args": { ... }
        }
    },
    "dataset": {
        "name": "ImageConditionedSLat",
        "args": { ... }
    },
    "trainer": {
        "name": "LoRAImageConditionedSparseFlowMatchingCFGTrainer",
        "args": {
            // Standard training args
            "max_steps": 100000,
            "batch_size_per_gpu": 8,
            "optimizer": { ... },
            
            // LoRA-specific args
            "use_lora": true,
            "lora_r": 64,
            "lora_alpha": 128,
            "lora_dropout": 0.0,
            "lora_target_modules": ["to_q", "to_kv", "to_out", "to_qkv"],
            "lora_models": ["denoiser"],
            
            // Base model to fine-tune
            "finetune_ckpt": {
                "denoiser": "pretrained_models/TRELLIS-image-large/ckpts/slat_flow_img_dit_L_64l8p2_fp16.pt"
            }
        }
    }
}
```

### Available LoRA Trainers

| Trainer | Use Case |
|---------|----------|
| `LoRAImageConditionedFlowMatchingCFGTrainer` | Dense structured latent flow model (SLat) |
| `LoRAImageConditionedSparseFlowMatchingCFGTrainer` | Sparse structure flow model |
| `LoRANoisyImageConditionedFlowMatchingCFGTrainer` | SLat with noisy conditioning |
| `LoRANoisyImageConditionedSparseFlowMatchingCFGTrainer` | Sparse with noisy conditioning |

### Example Config Files

TRELLIS includes example LoRA configs:

- `configs/finetune_slat_flow_img_dit_L_64l8p2_fp16_lora.json` - SLat Flow Model
- `configs/finetune_ss_flow_img_dit_L_16l8_fp16_lora.json` - Sparse Structure Flow Model

---

## Training with LoRA

### Basic Training Command

```bash
python train.py \
    --config configs/finetune_slat_flow_img_dit_L_64l8p2_fp16_lora.json \
    --output_dir outputs/my_lora_experiment \
    --data_dir /path/to/your/dataset
```

### Multi-GPU Training

```bash
python train.py \
    --config configs/finetune_slat_flow_img_dit_L_64l8p2_fp16_lora.json \
    --output_dir outputs/my_lora_experiment \
    --data_dir /path/to/your/dataset \
    --num_gpus 4
```

### Multi-Node Training

```bash
# Node 0
python train.py \
    --config configs/finetune_slat_flow_img_dit_L_64l8p2_fp16_lora.json \
    --output_dir outputs/my_lora_experiment \
    --data_dir /path/to/your/dataset \
    --num_nodes 2 \
    --node_rank 0 \
    --master_addr $MASTER_ADDR \
    --master_port 12345

# Node 1
python train.py \
    --config configs/finetune_slat_flow_img_dit_L_64l8p2_fp16_lora.json \
    --output_dir outputs/my_lora_experiment \
    --data_dir /path/to/your/dataset \
    --num_nodes 2 \
    --node_rank 1 \
    --master_addr $MASTER_ADDR \
    --master_port 12345
```

### What to Expect During Training

When training starts, you'll see LoRA initialization info:

```
============================================================
Applying LoRA adapters
============================================================
  LoRA rank: 64
  LoRA alpha: 128
  LoRA dropout: 0.0
  Target modules: ['to_q', 'to_kv', 'to_out', 'to_qkv']
  Models: ['denoiser']

  denoiser:
    Original params: 1,200,000,000
    Trainable params: 12,000,000
    Reduction: 99.00%
============================================================
```

### Output Structure

After training, your output directory will contain:

```
outputs/my_lora_experiment/
├── ckpts/
│   ├── denoiser_step0010000.pt      # Full checkpoints
│   ├── misc_step0010000.pt
│   └── ...
├── lora_ckpts/
│   ├── denoiser_lora_step0010000.pt # LoRA-only weights (small!)
│   └── ...
├── command.txt
├── config.json
└── denoiser_model_summary.txt
```

---

## LoRA Parameters Explained

### `lora_r` (Rank)

The rank of the low-rank matrices. This is the most important hyperparameter.

- **Lower rank (4-16)**: Fewer parameters, less capacity, faster training
- **Higher rank (64-128)**: More parameters, more capacity, better adaptation

| Rank | Trainable Params | Use Case |
|------|-----------------|----------|
| 8 | ~0.1% of base | Small adaptations, style transfer |
| 32 | ~0.5% of base | Medium adaptations |
| 64 | ~1% of base | **Recommended default** |
| 128 | ~2% of base | Complex domain shifts |

**Recommended:** Start with `lora_r=64` and adjust based on results.

### `lora_alpha` (Scaling Factor)

Controls the magnitude of LoRA updates. Typically set to 2× the rank.

```
effective_scaling = lora_alpha / lora_r
```

- Higher alpha = Stronger LoRA influence
- Lower alpha = More conservative updates

**Recommended:** `lora_alpha = 2 * lora_r` (e.g., if r=64, alpha=128)

### `lora_dropout`

Dropout applied to LoRA layers during training.

- `0.0` - No dropout (default, recommended for most cases)
- `0.05-0.1` - Light regularization for overfitting prevention
- `0.2+` - Heavy regularization (rarely needed)

### `lora_target_modules`

List of module names to apply LoRA to:

```json
"lora_target_modules": ["to_q", "to_kv", "to_out", "to_qkv"]
```

Available targets in TRELLIS:
- `to_q` - Query projection
- `to_kv` - Key-Value projection
- `to_out` - Output projection
- `to_qkv` - Combined QKV projection

**Recommended:** Use all four targets (default) for best results.

### `lora_models`

Which models to apply LoRA to:

```json
"lora_models": ["denoiser"]
```

Currently, `denoiser` is the primary target. The VAE models are typically kept frozen.

---

## Using Trained LoRA Weights

### Method 1: Using LoRA Utilities

```python
from trellis.utils.lora_utils import (
    apply_lora_to_model,
    load_lora_weights,
    LoRAModelWrapper,
    merge_lora_weights,
)
from trellis import models

# Load base model
base_model = models.ElasticSLatFlowModel(
    resolution=64,
    in_channels=8,
    out_channels=8,
    # ... other args
).cuda()

# Load LoRA weights
model = load_lora_weights(
    model=base_model,
    lora_path="outputs/my_lora_experiment/lora_ckpts/denoiser_lora_step0010000.pt",
    r=64,
    lora_alpha=128,
    target_modules=["to_q", "to_kv", "to_out", "to_qkv"],
)

# Use for inference
model.eval()
output = model(x_t, t, cond)
```

### Method 2: Using LoRAModelWrapper

```python
from trellis.utils.lora_utils import LoRAModelWrapper
from trellis import models

# Create base model
base_model = models.SparseStructureFlowModel(
    resolution=16,
    in_channels=8,
    # ... other args
).cuda()

# Load with wrapper
wrapper = LoRAModelWrapper.load(
    base_model=base_model,
    lora_path="outputs/my_experiment/lora_ckpts/denoiser_lora_step0010000.pt",
    r=64,
    lora_alpha=128,
)

# Get the model for inference
model = wrapper.get_model()
model.eval()
```

### Method 3: Merge LoRA Weights into Base Model

For faster inference (no PEFT dependency needed):

```python
from trellis.utils.lora_utils import load_lora_weights, merge_lora_weights

# Load model with LoRA
model = load_lora_weights(base_model, lora_path, r=64, lora_alpha=128)

# Merge LoRA weights into base model
merged_model = merge_lora_weights(model)

# Save merged model (standard PyTorch format)
torch.save(merged_model.state_dict(), "merged_model.pt")
```

---

## Advanced Usage

### Custom LoRA Configuration

Create your own config by modifying the trainer section:

```json
{
    "trainer": {
        "name": "LoRAImageConditionedSparseFlowMatchingCFGTrainer",
        "args": {
            // ... standard args ...
            
            // Custom LoRA settings for your use case
            "use_lora": true,
            "lora_r": 32,              // Lower rank for style transfer
            "lora_alpha": 64,          // 2x rank
            "lora_dropout": 0.05,      // Light regularization
            "lora_target_modules": ["to_q", "to_kv"],  // Only cross-attention
            "lora_models": ["denoiser"]
        }
    }
}
```

### Combining with Noisy Conditioning

For improved robustness, combine LoRA with noisy image conditioning:

```json
{
    "trainer": {
        "name": "LoRANoisyImageConditionedSparseFlowMatchingCFGTrainer",
        "args": {
            // LoRA args
            "use_lora": true,
            "lora_r": 64,
            "lora_alpha": 128,
            
            // Noisy conditioning args
            "cond_noise_std": 0.1,
            "cond_noise_schedule": "uniform",
            "cond_noise_std_min": 0.0,
            "cond_noise_std_max": 0.2
        }
    }
}
```

### Programmatic Usage

```python
from trellis.utils.lora_utils import (
    get_lora_config,
    apply_lora_to_model,
    print_trainable_parameters,
    save_lora_weights,
)

# Create custom LoRA config
lora_config = get_lora_config(
    r=64,
    lora_alpha=128,
    lora_dropout=0.0,
    target_modules=["to_q", "to_kv", "to_out", "to_qkv"],
)

# Apply to model
peft_model = apply_lora_to_model(
    model=base_model,
    r=64,
    lora_alpha=128,
)

# Check parameters
print_trainable_parameters(peft_model, "My LoRA Model")
# Output:
# My LoRA Model Parameters:
#   Total:     1,200,000,000
#   Trainable: 12,000,000 (1.00%)
#   Frozen:    1,188,000,000

# Train your model...
# ...

# Save only LoRA weights
save_lora_weights(peft_model, "my_lora_weights.safetensors")
```

### Training Tips

1. **Learning Rate**: Use a lower learning rate than full fine-tuning (e.g., `5e-6` instead of `1e-4`)

2. **Batch Size**: You can often use larger batch sizes due to reduced memory

3. **Gradient Accumulation**: Reduce `batch_split` since memory usage is lower

4. **Early Stopping**: LoRA can converge faster - monitor validation loss

5. **Multiple LoRA Adapters**: Train different LoRA weights for different domains

---

## Troubleshooting

### "peft library required for LoRA"

Install PEFT:
```bash
pip install peft
```

### Out of Memory Errors

Even with LoRA, you might run out of memory. Try:
1. Reduce `batch_size_per_gpu`
2. Increase `batch_split` for gradient accumulation
3. Reduce `lora_r`

### Poor Generation Quality

1. Increase `lora_r` (try 128)
2. Train for more steps
3. Check your dataset quality
4. Try adding more `lora_target_modules`

### Loading Errors

Ensure LoRA parameters match when loading:
```python
# Must match training config!
model = load_lora_weights(
    base_model,
    lora_path,
    r=64,           # Same as training
    lora_alpha=128, # Same as training
    target_modules=["to_q", "to_kv", "to_out", "to_qkv"],  # Same as training
)
```

### State Dict Key Mismatches

If you get key errors when loading:
```python
# Use strict=False for partial loading
model = load_lora_weights(base_model, lora_path, strict=False, ...)
```

---

## Comparison: LoRA vs Full Fine-tuning

| Aspect | LoRA | Full Fine-tuning |
|--------|------|------------------|
| Trainable Params | ~1% | 100% |
| GPU Memory | ~50% reduction | Full |
| Training Speed | Faster | Slower |
| Checkpoint Size | ~MB | ~GB |
| Model Capacity | Limited by rank | Full |
| Best For | Domain adaptation, style | Major changes |

---

## References

- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Low-Rank Adaptation of Large Language Models
- [PEFT Library](https://github.com/huggingface/peft) - Parameter-Efficient Fine-Tuning
- [DSO Paper](https://github.com/RuiningLi/dso) - Inspiration for TRELLIS LoRA implementation

---

## Quick Reference

### Recommended Default Config

```json
{
    "use_lora": true,
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.0,
    "lora_target_modules": ["to_q", "to_kv", "to_out", "to_qkv"],
    "lora_models": ["denoiser"]
}
```

### Quick Training Command

```bash
python train.py \
    --config configs/finetune_slat_flow_img_dit_L_64l8p2_fp16_lora.json \
    --output_dir outputs/lora_experiment \
    --data_dir /path/to/dataset
```

### Quick Loading Code

```python
from trellis.utils.lora_utils import LoRAModelWrapper

wrapper = LoRAModelWrapper.load(
    base_model=my_model,
    lora_path="outputs/lora_experiment/lora_ckpts/denoiser_lora_step0010000.pt",
    r=64, lora_alpha=128,
)
model = wrapper.get_model()
```

