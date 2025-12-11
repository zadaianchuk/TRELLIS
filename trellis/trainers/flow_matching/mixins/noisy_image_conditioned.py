"""
Noisy Image Conditioning Mixin for RAE-style noise injection.

This mixin adds Gaussian noise to the encoded image features during training,
similar to the approach in RAE (Representation Autoencoders) paper.
The noise injection makes the model more robust to variations in conditioning features.

Reference: https://github.com/bytetriper/RAE
"""
from typing import *
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image

from ....utils import dist_utils


class NoisyImageConditionedMixin:
    """
    Mixin for image-conditioned models with RAE-style noise injection.
    
    During training, Gaussian noise is added to the encoded image features
    to improve robustness. The noise scale can be:
    - Fixed: constant noise_std
    - Uniform: sampled from [noise_std_min, noise_std_max]
    - LogNormal: sampled from a log-normal distribution
    
    Args:
        image_cond_model: The image conditioning model (e.g., 'dinov2_vitl14_reg').
        cond_noise_std: Standard deviation of noise to add to conditioning features.
        cond_noise_std_min: Minimum noise std when using uniform sampling.
        cond_noise_std_max: Maximum noise std when using uniform sampling.
        cond_noise_schedule: Type of noise schedule ('fixed', 'uniform', 'lognormal').
        cond_noise_enabled: Whether to enable noise injection during training.
        cond_noise_at_inference: Whether to also add noise during inference/sampling.
    """
    def __init__(
        self, 
        *args, 
        image_cond_model: str = 'dinov2_vitl14_reg',
        cond_noise_std: float = 0.1,
        cond_noise_std_min: float = 0.0,
        cond_noise_std_max: float = 0.2,
        cond_noise_schedule: str = 'fixed',  # 'fixed', 'uniform', 'lognormal'
        cond_noise_enabled: bool = True,
        cond_noise_at_inference: bool = False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.image_cond_model_name = image_cond_model
        self.image_cond_model = None  # the model is init lazily
        
        # Noise configuration
        self.cond_noise_std = cond_noise_std
        self.cond_noise_std_min = cond_noise_std_min
        self.cond_noise_std_max = cond_noise_std_max
        self.cond_noise_schedule = cond_noise_schedule
        self.cond_noise_enabled = cond_noise_enabled
        self.cond_noise_at_inference = cond_noise_at_inference
        
    @staticmethod
    def prepare_for_training(image_cond_model: str, **kwargs):
        """
        Prepare for training.
        """
        if hasattr(super(NoisyImageConditionedMixin, NoisyImageConditionedMixin), 'prepare_for_training'):
            super(NoisyImageConditionedMixin, NoisyImageConditionedMixin).prepare_for_training(**kwargs)
        # download the model
        torch.hub.load('facebookresearch/dinov2', image_cond_model, pretrained=True)
        
    def _init_image_cond_model(self):
        """
        Initialize the image conditioning model.
        """
        with dist_utils.local_master_first():
            dinov2_model = torch.hub.load('facebookresearch/dinov2', self.image_cond_model_name, pretrained=True)
        dinov2_model.eval().cuda()
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model = {
            'model': dinov2_model,
            'transform': transform,
        }
    
    def _sample_noise_std(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample noise standard deviation based on the schedule.
        
        Returns:
            Tensor of shape [batch_size, 1, 1] for broadcasting.
        """
        if self.cond_noise_schedule == 'fixed':
            noise_std = torch.full((batch_size,), self.cond_noise_std, device=device)
        elif self.cond_noise_schedule == 'uniform':
            noise_std = torch.rand(batch_size, device=device) * (self.cond_noise_std_max - self.cond_noise_std_min) + self.cond_noise_std_min
        elif self.cond_noise_schedule == 'lognormal':
            # Log-normal distribution centered around cond_noise_std
            log_std = torch.randn(batch_size, device=device) * 0.5 + np.log(self.cond_noise_std)
            noise_std = torch.exp(log_std)
            noise_std = torch.clamp(noise_std, self.cond_noise_std_min, self.cond_noise_std_max)
        else:
            raise ValueError(f"Unknown noise schedule: {self.cond_noise_schedule}")
        
        return noise_std.view(-1, 1, 1)  # Shape for broadcasting with [B, N, C] features
    
    def _add_noise_to_features(self, features: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Add Gaussian noise to features during training.
        
        Args:
            features: Tensor of shape [B, N, C] where B is batch size, 
                     N is number of tokens, C is feature dimension.
            training: Whether in training mode.
            
        Returns:
            Noisy features if training and noise enabled, otherwise original features.
        """
        if not training or not self.cond_noise_enabled:
            return features
        
        batch_size = features.shape[0]
        noise_std = self._sample_noise_std(batch_size, features.device)
        
        # Add Gaussian noise scaled by noise_std
        noise = torch.randn_like(features) * noise_std
        noisy_features = features + noise
        
        return noisy_features
    
    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, List[Image.Image]], add_noise: bool = False) -> torch.Tensor:
        """
        Encode the image.
        
        Args:
            image: Input images as tensor or list of PIL images.
            add_noise: Whether to add noise to the encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).cuda()
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        if self.image_cond_model is None:
            self._init_image_cond_model()
        image = self.image_cond_model['transform'](image).cuda()
        features = self.image_cond_model['model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        
        # Add noise if requested (during training)
        if add_noise:
            patchtokens = self._add_noise_to_features(patchtokens, training=True)
        
        return patchtokens
        
    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data with noise injection during training.
        """
        # Add noise during training
        cond = self.encode_image(cond, add_noise=self.cond_noise_enabled)
        kwargs['neg_cond'] = torch.zeros_like(cond)
        cond = super().get_cond(cond, **kwargs)
        return cond
    
    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        Noise is added if cond_noise_at_inference is True.
        """
        # Add noise during inference if enabled
        add_noise = self.cond_noise_enabled and self.cond_noise_at_inference
        cond = self.encode_image(cond, add_noise=add_noise)
        kwargs['neg_cond'] = torch.zeros_like(cond)
        cond = super().get_inference_cond(cond, **kwargs)
        return cond

    def vis_cond(self, cond, **kwargs):
        """
        Visualize the conditioning data.
        """
        return {'image': {'value': cond, 'type': 'image'}}


