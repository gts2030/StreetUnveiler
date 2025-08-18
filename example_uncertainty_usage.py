#!/usr/bin/env python3
"""
Example script demonstrating how to use the improved uncertainty computation
for wandb logging and general uncertainty analysis.
"""

import torch
from utils.dyn_uncertainty.uncertainty_model import get_viewpoint_uncertainty_no_grad
from utils.general_utils import colormap
import numpy as np

def example_uncertainty_computation(scene, viewpoint_cam, feature_extractor, feature_cfg):
    """
    Example function showing how to compute uncertainty using the new improved function.
    
    Args:
        scene: Scene object containing uncertainty_mlp
        viewpoint_cam: Camera viewpoint object
        feature_extractor: DINO feature extractor
        feature_cfg: Feature extractor configuration
    """
    
    # Method 1: Using the standalone function
    print("=== Method 1: Using standalone function ===")
    
    # First, extract DINO features if not already available
    if not hasattr(viewpoint_cam, 'features') or viewpoint_cam.features is None:
        gt_image = viewpoint_cam.original_image.cuda()
        if gt_image.dim() == 3:
            gt_image_input = gt_image.unsqueeze(0)
        else:
            gt_image_input = gt_image
            
        from utils.mono_priors.img_feature_extractors import predict_img_features
        uncertainty_features = predict_img_features(
            feature_extractor,
            0,  # frame_id
            gt_image_input,
            feature_cfg,
            "cuda",
            save_feat=False
        )
        # Store features in viewpoint
        viewpoint_cam.features = uncertainty_features
    
    # Get uncertainty configuration
    from utils.config_utils import get_default_config
    uncertainty_cfg = get_default_config()['uncertainty']
    
    # Compute uncertainty using the improved function
    uncertainty_map = get_viewpoint_uncertainty_no_grad(
        scene.uncertainty_mlp,
        viewpoint_cam,
        uncer_params=uncertainty_cfg,
        device="cuda"
    )
    
    print(f"Uncertainty map shape: {uncertainty_map.shape}")
    print(f"Uncertainty stats - Min: {uncertainty_map.min().item():.6f}, Max: {uncertainty_map.max().item():.6f}, Mean: {uncertainty_map.mean().item():.6f}")
    
    # Method 2: Using the Camera class method (more convenient)
    print("\n=== Method 2: Using Camera class method ===")
    
    uncertainty_map_2 = viewpoint_cam.get_uncertainty_no_grad(
        scene.uncertainty_mlp,
        uncer_params=uncertainty_cfg
    )
    
    print(f"Uncertainty map 2 shape: {uncertainty_map_2.shape}")
    print(f"Uncertainty 2 stats - Min: {uncertainty_map_2.min().item():.6f}, Max: {uncertainty_map_2.max().item():.6f}, Mean: {uncertainty_map_2.mean().item():.6f}")
    
    # Verify both methods give the same result
    assert torch.allclose(uncertainty_map, uncertainty_map_2, atol=1e-6), "Both methods should give identical results"
    print("✓ Both methods produce identical results")
    
    return uncertainty_map

def visualize_uncertainty_for_wandb(uncertainty_map, image_name, iteration=0):
    """
    Example function showing how to properly visualize uncertainty for wandb logging.
    
    Args:
        uncertainty_map: Uncertainty tensor from get_viewpoint_uncertainty_no_grad
        image_name: Name for the image
        iteration: Training iteration number
    """
    
    print(f"\n=== Visualizing uncertainty for {image_name} ===")
    
    # Apply jet colormap directly (colormap function handles normalization automatically)
    # No manual normalization needed since colormap uses matplotlib's imshow which auto-normalizes
    uncertainty_colored_tensor = colormap(uncertainty_map.cpu().numpy(), cmap='jet')
    
    # Log to wandb (if available)
    try:
        from utils.wandb_utils import log_image, is_wandb_available
        if is_wandb_available():
            log_image(f"uncertainty/{image_name}", uncertainty_colored_tensor, step=iteration, caption=f"Uncertainty map {image_name}")
            print("✓ Logged uncertainty image to wandb")
        else:
            print("⚠ Wandb not available, skipping wandb logging")
    except ImportError:
        print("⚠ Wandb utils not available, skipping wandb logging")
    
    return uncertainty_colored_tensor

def compare_old_vs_new_uncertainty(scene, viewpoint_cam, feature_extractor, feature_cfg):
    """
    Compare the old uncertainty computation method vs the new improved method.
    """
    
    print("\n=== Comparing Old vs New Uncertainty Methods ===")
    
    # Extract features
    gt_image = viewpoint_cam.original_image.cuda()
    if gt_image.dim() == 3:
        gt_image_input = gt_image.unsqueeze(0)
    else:
        gt_image_input = gt_image
        
    from utils.mono_priors.img_feature_extractors import predict_img_features
    uncertainty_features = predict_img_features(
        feature_extractor,
        0,
        gt_image_input,
        feature_cfg,
        "cuda",
        save_feat=False
    )
    
    # Old method (direct network call with basic resizing)
    target_size = gt_image.shape[-2:]  # (H, W)
    old_uncertainty = scene.uncertainty_mlp(uncertainty_features, target_size=target_size)
    
    # New method (with proper processing pipeline)
    viewpoint_cam.features = uncertainty_features
    from utils.config_utils import get_default_config
    uncertainty_cfg = get_default_config()['uncertainty']
    
    new_uncertainty = get_viewpoint_uncertainty_no_grad(
        scene.uncertainty_mlp,
        viewpoint_cam,
        uncer_params=uncertainty_cfg,
        device="cuda"
    )
    
    print(f"Old uncertainty stats - Min: {old_uncertainty.min().item():.6f}, Max: {old_uncertainty.max().item():.6f}, Mean: {old_uncertainty.mean().item():.6f}")
    print(f"New uncertainty stats - Min: {new_uncertainty.min().item():.6f}, Max: {new_uncertainty.max().item():.6f}, Mean: {new_uncertainty.mean().item():.6f}")
    
    # Check difference
    diff = torch.abs(new_uncertainty - old_uncertainty)
    print(f"Difference stats - Min: {diff.min().item():.6f}, Max: {diff.max().item():.6f}, Mean: {diff.mean().item():.6f}")
    
    return old_uncertainty, new_uncertainty

if __name__ == "__main__":
    print("This is an example script showing how to use the improved uncertainty computation.")
    print("Import the functions above in your training script to use them.")
    print("\nKey improvements:")
    print("1. Follows exact same processing pipeline as mapping loss computation")
    print("2. Applies proper clipping: torch.clip(uncertainty, min=0.1) + 1e-3")
    print("3. Uses resample_tensor_to_shape for consistent resizing")
    print("4. Applies data rate adjustment: (uncertainty_resized - 0.1) * data_rate + 0.1")
    print("5. Returns uncertainty^2 as final result")
    print("6. Provides both standalone function and Camera class method") 