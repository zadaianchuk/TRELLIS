#!/usr/bin/env python3
"""
Evaluation script for the ABO noisy conditioning experiment.

This script evaluates trained models on the validation set and generates
visualizations comparing ground truth sparse structures with generated ones.

Usage:
    python evaluate.py --config CONFIG --checkpoint CHECKPOINT --data_dir DATA_DIR --output_dir OUTPUT_DIR
"""
import os
import sys
import json
import argparse
from typing import *
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import utils3d

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from trellis import models, datasets
from trellis.pipelines import samplers
from trellis.representations.octree import DfsOctree as Octree
from trellis.renderers import OctreeRenderer


class Evaluator:
    """Evaluator for the noisy conditioning experiment."""
    
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        data_dir: str,
        output_dir: str,
        device: str = 'cuda',
        image_cond_model: str = 'dinov2_vitl14_reg',
    ):
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Build model
        print("Building model...")
        self.model = getattr(models, self.config['models']['denoiser']['name'])(
            **self.config['models']['denoiser']['args']
        ).to(device)
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.model.load_state_dict(ckpt)
        self.model.eval()
        
        # Build dataset
        print("Building dataset...")
        dataset_args = self.config['dataset']['args'].copy()
        self.dataset = getattr(datasets, self.config['dataset']['name'])(
            data_dir,
            **dataset_args
        )
        print(f"Dataset: {self.dataset}")
        
        # Initialize image conditioning model
        print(f"Loading image conditioning model: {image_cond_model}...")
        self.image_cond_model = torch.hub.load('facebookresearch/dinov2', image_cond_model, pretrained=True)
        self.image_cond_model.eval().to(device)
        
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Initialize sampler
        sigma_min = self.config['trainer']['args'].get('sigma_min', 1e-5)
        self.sampler = samplers.FlowEulerCfgSampler(sigma_min)
        
        # Load sparse structure decoder for visualization
        print("Loading sparse structure decoder...")
        ss_dec_path = dataset_args.get('pretrained_ss_dec', 'pretrained_models/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16')
        self.ss_dec = models.from_pretrained(ss_dec_path).to(device).eval()
        
    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image using DINOv2."""
        image = self.transform(image).to(self.device)
        features = self.image_cond_model(image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
    
    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to sparse structure."""
        return self.ss_dec(z)
    
    @torch.no_grad()
    def generate(
        self,
        cond: torch.Tensor,
        noise: torch.Tensor,
        steps: int = 50,
        cfg_strength: float = 3.0,
    ) -> torch.Tensor:
        """Generate sparse structure from conditioning."""
        cond = self.encode_image(cond)
        neg_cond = torch.zeros_like(cond)
        
        result = self.sampler.sample(
            self.model,
            noise=noise,
            cond=cond,
            neg_cond=neg_cond,
            steps=steps,
            cfg_strength=cfg_strength,
            verbose=False,
        )
        return result.samples
    
    @torch.no_grad()
    def visualize_sparse_structure(self, ss: torch.Tensor, resolution: int = 512) -> torch.Tensor:
        """Render sparse structure from multiple views."""
        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = resolution
        renderer.rendering_options.near = 0.8
        renderer.rendering_options.far = 1.6
        renderer.rendering_options.bg_color = (0, 0, 0)
        renderer.rendering_options.ssaa = 4
        renderer.pipe.primitive = 'voxel'
        
        # Camera setup
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        pitch = np.pi / 6  # Fixed pitch for consistent visualization
        
        images = []
        for i in range(ss.shape[0]):
            # Build representation
            representation = Octree(
                depth=10,
                aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
                device=self.device,
                primitive='voxel',
                sh_degree=0,
                primitive_config={'solid': True},
            )
            coords = torch.nonzero(ss[i, 0] > 0, as_tuple=False)
            res = ss.shape[-1]
            representation.position = coords.float() / res
            representation.depth = torch.full(
                (representation.position.shape[0], 1), 
                int(np.log2(res)), 
                dtype=torch.uint8, 
                device=self.device
            )
            
            # Render from multiple views
            view_images = []
            for yaw in yaws:
                orig = torch.tensor([
                    np.sin(yaw) * np.cos(pitch),
                    np.cos(yaw) * np.cos(pitch),
                    np.sin(pitch),
                ]).float().to(self.device) * 2
                fov = torch.deg2rad(torch.tensor(30)).to(self.device)
                extrinsics = utils3d.torch.extrinsics_look_at(
                    orig, 
                    torch.tensor([0, 0, 0]).float().to(self.device), 
                    torch.tensor([0, 0, 1]).float().to(self.device)
                )
                intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
                
                res = renderer.render(representation, extrinsics, intrinsics, colors_overwrite=representation.position)
                view_images.append(res['color'])
            
            # Combine views into a 2x2 grid
            combined = torch.zeros(3, resolution * 2, resolution * 2, device=self.device)
            combined[:, :resolution, :resolution] = view_images[0]
            combined[:, :resolution, resolution:] = view_images[1]
            combined[:, resolution:, :resolution] = view_images[2]
            combined[:, resolution:, resolution:] = view_images[3]
            images.append(combined)
        
        return torch.stack(images)
    
    @torch.no_grad()
    def evaluate(
        self,
        num_samples: int = 50,
        batch_size: int = 4,
        steps: int = 50,
        cfg_strength: float = 3.0,
        save_individual: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate the model on the validation set.
        
        Returns:
            Dictionary with evaluation metrics and paths to visualizations.
        """
        print(f"\nEvaluating on {num_samples} samples...")
        
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )
        
        results = {
            'num_samples': num_samples,
            'steps': steps,
            'cfg_strength': cfg_strength,
            'samples': [],
        }
        
        samples_processed = 0
        sample_idx = 0
        
        for batch in tqdm(dataloader, desc="Generating samples"):
            if samples_processed >= num_samples:
                break
            
            batch_size_actual = min(batch_size, num_samples - samples_processed)
            
            # Get data
            x_0 = batch['x_0'][:batch_size_actual].to(self.device)
            cond = batch['cond'][:batch_size_actual].to(self.device)
            
            # Generate
            noise = x_0.clone()
            noise.feats = torch.randn_like(noise.feats)
            
            generated = self.generate(cond, noise, steps=steps, cfg_strength=cfg_strength)
            
            # Decode to sparse structure
            ss_gt = self.decode_latent(x_0.feats.reshape(x_0.shape[0], 8, 16, 16, 16))
            ss_gen = self.decode_latent(generated.feats.reshape(generated.shape[0], 8, 16, 16, 16))
            
            # Visualize
            vis_gt = self.visualize_sparse_structure(ss_gt)
            vis_gen = self.visualize_sparse_structure(ss_gen)
            
            # Save individual samples
            if save_individual:
                for i in range(batch_size_actual):
                    sample_dir = os.path.join(self.output_dir, f'sample_{sample_idx:04d}')
                    os.makedirs(sample_dir, exist_ok=True)
                    
                    # Save conditioning image
                    cond_img = (cond[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(cond_img).save(os.path.join(sample_dir, 'cond.png'))
                    
                    # Save GT visualization
                    gt_img = (vis_gt[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(gt_img).save(os.path.join(sample_dir, 'gt.png'))
                    
                    # Save generated visualization
                    gen_img = (vis_gen[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(gen_img).save(os.path.join(sample_dir, 'gen.png'))
                    
                    # Create comparison image
                    comparison = np.concatenate([
                        np.pad(cond_img, ((256, 256), (256, 256), (0, 0))),  # Center condition in larger canvas
                        gt_img,
                        gen_img
                    ], axis=1)
                    Image.fromarray(comparison).save(os.path.join(sample_dir, 'comparison.png'))
                    
                    results['samples'].append({
                        'idx': sample_idx,
                        'path': sample_dir,
                    })
                    
                    sample_idx += 1
            
            samples_processed += batch_size_actual
        
        # Save results summary
        results_path = os.path.join(self.output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {self.output_dir}")
        return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate ABO noisy conditioning experiment')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config JSON file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to validation data directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of samples to evaluate')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for evaluation')
    parser.add_argument('--steps', type=int, default=50,
                        help='Number of sampling steps')
    parser.add_argument('--cfg_strength', type=float, default=3.0,
                        help='Classifier-free guidance strength')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    args = parser.parse_args()
    
    evaluator = Evaluator(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
    )
    
    results = evaluator.evaluate(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        steps=args.steps,
        cfg_strength=args.cfg_strength,
    )
    
    print("\nEvaluation complete!")
    print(f"Generated {len(results['samples'])} samples")


if __name__ == '__main__':
    main()


