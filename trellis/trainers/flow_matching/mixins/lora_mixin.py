"""
LoRA Mixin for TRELLIS trainers.

This mixin adds LoRA (Low-Rank Adaptation) support to flow matching trainers,
allowing efficient fine-tuning with significantly fewer trainable parameters.

Reference: DSO paper - https://github.com/RuiningLi/dso
"""
from typing import Dict, List, Optional, Any
import os
import torch
import torch.nn as nn

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


class LoRAMixin:
    """
    Mixin that adds LoRA support to trainers.
    
    To use, inherit from this mixin before the trainer class:
        class MyLoRATrainer(LoRAMixin, ImageConditionedFlowMatchingCFGTrainer):
            pass
    
    Config args:
        use_lora (bool): Whether to use LoRA adapters.
        lora_r (int): LoRA rank.
        lora_alpha (int): LoRA scaling factor.
        lora_dropout (float): LoRA dropout rate.
        lora_target_modules (list): List of module names to apply LoRA to.
        lora_models (list): List of model names to apply LoRA to (default: ['denoiser']).
    """
    
    def __init__(
        self,
        *args,
        use_lora: bool = False,
        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.0,
        lora_target_modules: Optional[List[str]] = None,
        lora_models: Optional[List[str]] = None,
        **kwargs
    ):
        # Store LoRA config before calling parent init
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or ["to_q", "to_kv", "to_out", "to_qkv"]
        self.lora_models = lora_models or ["denoiser"]
        
        # For LoRA, we handle finetune_ckpt ourselves before applying LoRA
        # Pop it from kwargs so parent doesn't try to load after LoRA wrapping
        self._lora_finetune_ckpt = kwargs.pop('finetune_ckpt', None) if use_lora else None
        
        # Call parent init
        super().__init__(*args, **kwargs)
    
    def init_models_and_more(self, **kwargs):
        """
        Override to apply LoRA adapters before DDP wrapping.
        For LoRA, we load the finetune checkpoint BEFORE applying LoRA adapters.
        """
        if self.use_lora:
            if not PEFT_AVAILABLE:
                raise ImportError("peft library required for LoRA. Install with: pip install peft")
            
            # Load finetune checkpoint BEFORE applying LoRA
            if self._lora_finetune_ckpt:
                if self.is_master:
                    print('\n' + '=' * 60)
                    print('Loading base model checkpoint for LoRA fine-tuning:')
                    for name, path in self._lora_finetune_ckpt.items():
                        print(f'  - {name}: {path}')
                
                from trellis.utils.dist_utils import read_file_dist
                
                for name, model in self.models.items():
                    if name in self._lora_finetune_ckpt:
                        ckpt_path = self._lora_finetune_ckpt[name]
                        model_ckpt = torch.load(read_file_dist(ckpt_path), map_location=self.device, weights_only=True)
                        model_state_dict = model.state_dict()
                        
                        # Check for shape mismatches
                        for k, v in model_ckpt.items():
                            if k in model_state_dict and model_ckpt[k].shape != model_state_dict[k].shape:
                                if self.is_master:
                                    print(f'  Warning: {k} shape mismatch, {model_ckpt[k].shape} vs {model_state_dict[k].shape}, skipped.')
                                model_ckpt[k] = model_state_dict[k]
                        
                        model.load_state_dict(model_ckpt)
                        if self.is_master:
                            print(f'  Loaded {name} checkpoint.')
                
                if self.is_master:
                    print('=' * 60)
            
            if self.is_master:
                print("\n" + "=" * 60)
                print("Applying LoRA adapters")
                print("=" * 60)
                print(f"  LoRA rank: {self.lora_r}")
                print(f"  LoRA alpha: {self.lora_alpha}")
                print(f"  LoRA dropout: {self.lora_dropout}")
                print(f"  Target modules: {self.lora_target_modules}")
                print(f"  Models: {self.lora_models}")
            
            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.lora_target_modules,
            )
            
            for model_name in self.lora_models:
                if model_name in self.models:
                    original_model = self.models[model_name]
                    
                    # Count original params
                    original_params = sum(p.numel() for p in original_model.parameters())
                    
                    # Apply LoRA
                    self.models[model_name] = get_peft_model(original_model, lora_config)
                    
                    # Count trainable params
                    trainable_params = sum(
                        p.numel() for p in self.models[model_name].parameters() 
                        if p.requires_grad
                    )
                    
                    if self.is_master:
                        print(f"\n  {model_name}:")
                        print(f"    Original params: {original_params:,}")
                        print(f"    Trainable params: {trainable_params:,}")
                        print(f"    Reduction: {100 * (1 - trainable_params / original_params):.2f}%")
                else:
                    if self.is_master:
                        print(f"  Warning: Model '{model_name}' not found, skipping LoRA")
            
            if self.is_master:
                print("=" * 60 + "\n")
        
        # Call parent init
        super().init_models_and_more(**kwargs)
    
    def _master_params_to_state_dicts(self, master_params):
        """
        Override to handle PEFT model state dicts properly.
        """
        if self.use_lora:
            # For LoRA models, we need to handle the state dict differently
            from ..utils import unflatten_master_params
            
            if self.fp16_mode == 'inflat_all':
                master_params = unflatten_master_params(self.model_params, master_params)
            
            state_dicts = {}
            for name, model in self.models.items():
                if hasattr(model, 'base_model'):
                    # It's a PEFT model - get full state dict
                    state_dicts[name] = model.state_dict()
                else:
                    state_dicts[name] = model.state_dict()
            
            master_params_names = sum(
                [[(name, n) for n, p in model.named_parameters() if p.requires_grad] 
                 for name, model in self.models.items()]
            , [])
            for i, (model_name, param_name) in enumerate(master_params_names):
                state_dicts[model_name][param_name] = master_params[i]
            
            return state_dicts
        else:
            return super()._master_params_to_state_dicts(master_params)
    
    def save_lora_only(self, save_dir: Optional[str] = None):
        """
        Save only the LoRA weights (much smaller than full model).
        
        Args:
            save_dir: Directory to save to. Defaults to output_dir/lora_ckpts.
        """
        if not self.use_lora:
            print("Warning: use_lora is False, skipping LoRA save")
            return
        
        assert self.is_master, 'save_lora_only() should be called only by rank 0'
        
        if save_dir is None:
            save_dir = os.path.join(self.output_dir, 'lora_ckpts')
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f'\nSaving LoRA weights at step {self.step}...')
        
        for model_name in self.lora_models:
            if model_name in self.models:
                model = self.models[model_name]
                
                # Get only LoRA parameters
                lora_state_dict = {
                    k: v for k, v in model.state_dict().items()
                    if 'lora_' in k or 'modules_to_save' in k
                }
                
                save_path = os.path.join(
                    save_dir, 
                    f'{model_name}_lora_step{self.step:07d}.pt'
                )
                torch.save(lora_state_dict, save_path)
                print(f'  Saved: {save_path}')
                print(f'    Parameters: {sum(v.numel() for v in lora_state_dict.values()):,}')
        
        print('Done.\n')
    
    def save(self):
        """
        Override save to also save LoRA-only weights.
        """
        super().save()
        
        if self.use_lora and self.is_master:
            self.save_lora_only()


class LoRAImageConditionedMixin(LoRAMixin):
    """
    Convenience mixin for image-conditioned LoRA training.
    Combines LoRA support with image conditioning.
    """
    pass


class LoRANoisyImageConditionedMixin(LoRAMixin):
    """
    Convenience mixin for noisy image-conditioned LoRA training.
    Combines LoRA support with noisy image conditioning.
    """
    pass

