"""
LoRA (Low-Rank Adaptation) utilities for TRELLIS models.

This module provides utilities for applying LoRA to TRELLIS sparse structure flow
and structured latent flow models, following the approach from DSO paper.

Reference: https://github.com/RuiningLi/dso
"""
from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn

try:
    from peft import LoraConfig, get_peft_model, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: peft library not available. Install with: pip install peft")


# Default target modules for TRELLIS attention layers
# These are the projection layers in MultiHeadAttention
TRELLIS_LORA_TARGET_MODULES = [
    "to_q",      # Query projection (cross-attention)
    "to_kv",     # Key-Value projection (cross-attention)
    "to_out",    # Output projection
    "to_qkv",    # QKV projection (self-attention)
]

# Additional modules that can be targeted
TRELLIS_EXTENDED_TARGET_MODULES = TRELLIS_LORA_TARGET_MODULES + [
    "self_attn.to_qkv",
    "cross_attn.to_q",
    "cross_attn.to_kv",
]


def get_lora_config(
    r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    modules_to_save: Optional[List[str]] = None,
    **kwargs
) -> 'LoraConfig':
    """
    Create a LoRA configuration for TRELLIS models.
    
    Args:
        r: Rank of the low-rank matrices. Higher = more parameters, more capacity.
        lora_alpha: Scaling factor for LoRA. Typically 2x the rank.
        lora_dropout: Dropout rate for LoRA layers.
        target_modules: List of module names to apply LoRA to.
                       If None, uses TRELLIS_LORA_TARGET_MODULES.
        modules_to_save: Additional modules to save (trained fully, not with LoRA).
        **kwargs: Additional arguments passed to LoraConfig.
    
    Returns:
        LoraConfig object.
    """
    if not PEFT_AVAILABLE:
        raise ImportError("peft library required for LoRA. Install with: pip install peft")
    
    if target_modules is None:
        target_modules = TRELLIS_LORA_TARGET_MODULES
    
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        **kwargs
    )
    
    return config


def apply_lora_to_model(
    model: nn.Module,
    r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    **kwargs
) -> nn.Module:
    """
    Apply LoRA adapters to a TRELLIS model.
    
    Args:
        model: The TRELLIS model (SparseStructureFlowModel or SLatFlowModel).
        r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: LoRA dropout rate.
        target_modules: Target module names.
        **kwargs: Additional LoraConfig arguments.
    
    Returns:
        Model wrapped with LoRA adapters (PeftModel).
    """
    if not PEFT_AVAILABLE:
        raise ImportError("peft library required for LoRA. Install with: pip install peft")
    
    lora_config = get_lora_config(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        **kwargs
    )
    
    peft_model = get_peft_model(model, lora_config)
    
    return peft_model


def load_lora_weights(
    model: nn.Module,
    lora_path: str,
    r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    strict: bool = True,
) -> nn.Module:
    """
    Load LoRA weights into a model.
    
    Args:
        model: Base TRELLIS model (without LoRA).
        lora_path: Path to LoRA weights file (.pt or .safetensors).
        r: LoRA rank (must match saved weights).
        lora_alpha: LoRA alpha (must match saved weights).
        lora_dropout: LoRA dropout.
        target_modules: Target modules (must match saved weights).
        strict: Whether to enforce strict state dict loading.
    
    Returns:
        Model with LoRA weights loaded.
    """
    if not PEFT_AVAILABLE:
        raise ImportError("peft library required for LoRA. Install with: pip install peft")
    
    # Apply LoRA structure to model
    peft_model = apply_lora_to_model(
        model,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    
    # Load weights
    if lora_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(lora_path)
    else:
        state_dict = torch.load(lora_path, map_location='cpu', weights_only=True)
    
    peft_model.load_state_dict(state_dict, strict=strict)
    
    return peft_model


def save_lora_weights(
    model: nn.Module,
    save_path: str,
    use_safetensors: bool = True,
):
    """
    Save only the LoRA weights from a PEFT model.
    
    Args:
        model: PeftModel with LoRA adapters.
        save_path: Path to save weights (.pt or .safetensors).
        use_safetensors: Whether to use safetensors format.
    """
    if hasattr(model, 'save_pretrained'):
        # If it's a PeftModel, use built-in saving
        import os
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir, safe_serialization=use_safetensors)
    else:
        # Manual save - get only trainable parameters
        state_dict = {
            k: v for k, v in model.state_dict().items()
            if 'lora_' in k or 'modules_to_save' in k
        }
        
        if use_safetensors and save_path.endswith('.safetensors'):
            from safetensors.torch import save_file
            save_file(state_dict, save_path)
        else:
            if not save_path.endswith('.pt'):
                save_path = save_path + '.pt'
            torch.save(state_dict, save_path)


def get_trainable_params(model: nn.Module) -> List[nn.Parameter]:
    """
    Get only the trainable parameters from a model.
    For LoRA models, this returns only the LoRA parameters.
    
    Args:
        model: Model (possibly wrapped with PEFT).
    
    Returns:
        List of trainable parameters.
    """
    return [p for p in model.parameters() if p.requires_grad]


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: The model.
    
    Returns:
        Dictionary with 'total', 'trainable', and 'frozen' parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
        'trainable_percent': 100 * trainable / total if total > 0 else 0,
    }


def print_trainable_parameters(model: nn.Module, model_name: str = "Model"):
    """
    Print parameter statistics for a model.
    
    Args:
        model: The model.
        model_name: Name to display.
    """
    stats = count_parameters(model)
    print(f"\n{model_name} Parameters:")
    print(f"  Total:     {stats['total']:,}")
    print(f"  Trainable: {stats['trainable']:,} ({stats['trainable_percent']:.2f}%)")
    print(f"  Frozen:    {stats['frozen']:,}")


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge LoRA weights into the base model for faster inference.
    
    After merging, the model no longer needs the PEFT library for inference,
    but you lose the ability to train the LoRA weights further.
    
    Args:
        model: PeftModel with LoRA adapters.
    
    Returns:
        Model with merged weights.
    """
    if hasattr(model, 'merge_and_unload'):
        return model.merge_and_unload()
    else:
        print("Warning: Model does not support merge_and_unload. Returning as-is.")
        return model


class LoRAModelWrapper:
    """
    Wrapper class for managing LoRA-enabled TRELLIS models.
    
    Example usage:
        wrapper = LoRAModelWrapper(
            base_model=sparse_structure_flow_model,
            r=64,
            lora_alpha=128,
        )
        
        # Get model for training
        model = wrapper.get_model()
        
        # After training, save weights
        wrapper.save('path/to/lora_weights.safetensors')
        
        # Load for inference
        wrapper = LoRAModelWrapper.load(
            base_model=fresh_model,
            lora_path='path/to/lora_weights.safetensors',
            r=64,
            lora_alpha=128,
        )
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or TRELLIS_LORA_TARGET_MODULES
        
        self.model = apply_lora_to_model(
            base_model,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
        )
        
        print_trainable_parameters(self.model, "LoRA Model")
    
    def get_model(self) -> nn.Module:
        """Get the LoRA-wrapped model."""
        return self.model
    
    def save(self, save_path: str, use_safetensors: bool = True):
        """Save LoRA weights."""
        save_lora_weights(self.model, save_path, use_safetensors)
    
    @classmethod
    def load(
        cls,
        base_model: nn.Module,
        lora_path: str,
        r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
    ) -> 'LoRAModelWrapper':
        """Load a LoRA model from saved weights."""
        wrapper = cls.__new__(cls)
        wrapper.r = r
        wrapper.lora_alpha = lora_alpha
        wrapper.lora_dropout = lora_dropout
        wrapper.target_modules = target_modules or TRELLIS_LORA_TARGET_MODULES
        
        wrapper.model = load_lora_weights(
            base_model,
            lora_path,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
        )
        
        return wrapper
    
    def merge(self) -> nn.Module:
        """Merge LoRA weights into base model."""
        return merge_lora_weights(self.model)

