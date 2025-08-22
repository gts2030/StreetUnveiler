"""
W&B (Weights & Biases) utilities for StreetUnveiler training.

This module handles W&B initialization, API key management, and logging utilities.
"""

import os
import torch
import numpy as np
import uuid
from datetime import datetime
from argparse import ArgumentParser, Namespace
from pathlib import Path

# Check if wandb is available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def prepare_output_and_wandb(args):
    """
    Prepare output directories and initialize W&B logging.
    
    Args:
        args: Dataset/model arguments containing model_path
        config: Training configuration dictionary
    
    Returns:
        bool: True if W&B was successfully initialized, False otherwise
    """
    # Set up output folder
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    
    # Initialize W&B logging
    wandb_enabled = False
    if WANDB_AVAILABLE:
        try:
            # Create experiment name with date and time
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = getattr(args, 'model_path', 'experiment').split('/')[-1]
            experiment_name = f"{base_name}_{current_time}"
            
            wandb_enabled = init_wandb(
                project_name="StreetUnveiler",
                experiment_name=experiment_name,
                model_path=args.model_path
            )
            
            if wandb_enabled:
                print(f"✅ W&B logging initialized: {experiment_name}")
            else:
                print("⚠️  W&B logging failed to initialize")
                
        except Exception as e:
            print(f"⚠️  W&B initialization failed: {e}")
            wandb_enabled = False
    else:
        print("⚠️  W&B not available. Install with: pip install wandb")
    
    return wandb_enabled


def load_wandb_api_key():
    """
    Load W&B API key from various sources in order of priority.
    
    Returns:
        str: W&B API key or None if not found
    """
    # Try different locations for API key
    key_locations = [
        "secrets/wandb_api_key.txt",
        "wandb_api_key.txt", 
        os.path.expanduser("~/.wandb_api_key"),
    ]
    
    # First check environment variable
    api_key = os.getenv('WANDB_API_KEY')
    if api_key:
        return api_key
    
    # Then check files
    for key_file in key_locations:
        if os.path.exists(key_file):
            try:
                with open(key_file, 'r') as f:
                    lines = f.readlines()
                    # Check each line for valid API key
                    for line in lines:
                        content = line.strip()
                        # Skip empty lines, comment lines, and example content
                        if (content and 
                            not content.startswith('#') and 
                            content != 'your_wandb_api_key_here' and
                            len(content) > 10):  # API keys are typically longer than 10 chars
                            return content
            except Exception as e:
                print(f"Warning: Could not read API key from {key_file}: {e}")
                continue
    
    return None


def init_wandb(project_name="StreetUnveiler", experiment_name=None, model_path=None):
    """
    Initialize W&B for training logging.
    
    Args:
        project_name (str): W&B project name
        experiment_name (str): Experiment name (optional)
        model_path (str): Model output path
        
    Returns:
        bool: True if W&B was successfully initialized, False otherwise
    """
    try:
        # Load API key
        api_key = load_wandb_api_key()
        if not api_key:
            print("⚠️  W&B API key not found. Logging will be disabled.")
            print("   Please create secrets/wandb_api_key.txt with your API key")
            print("   or set WANDB_API_KEY environment variable")
            return False
        
        # Login to W&B
        wandb.login(key=api_key)
        
        # Generate experiment name if not provided
        if not experiment_name:
            experiment_name = f"streetunveiler_{wandb.util.generate_id()}"
        
        # Initialize run
        wandb.init(
            project=project_name,
            name=experiment_name,
            dir=model_path if model_path else ".",
            reinit=True
        )
        
        print(f"✅ W&B initialized: {wandb.run.url}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize W&B: {e}")
        print("   Continuing without W&B logging...")
        return False


def log_scalar(name, value, step=None):
    """
    Log scalar value to W&B.
    
    Args:
        name (str): Metric name
        value (float): Metric value
        step (int): Step number (optional)
    """
    if wandb.run is None:
        return
        
    try:
        if step is not None:
            wandb.log({name: value}, step=step)
        else:
            wandb.log({name: value})
    except Exception as e:
        print(f"Warning: Failed to log scalar {name}: {e}")


def log_image(name, image, step=None, caption=None):
    """
    Log image to W&B.
    
    Args:
        name (str): Image name
        image (torch.Tensor or np.ndarray): Image data
        step (int): Step number (optional)
        caption (str): Image caption (optional)
    """
    if wandb.run is None:
        return
        
    try:
        # Convert tensor to numpy if needed
        if isinstance(image, torch.Tensor):
            image_np = image.detach().cpu().numpy()
        else:
            image_np = image
        
        # Handle different image formats
        if image_np.ndim == 4:  # Batch dimension
            image_np = image_np[0]  # Take first image
        
        if image_np.ndim == 3:
            # Check if channels first (C, H, W) -> convert to (H, W, C)
            if image_np.shape[0] <= 4:  # Assume channels first if first dim is small
                image_np = np.transpose(image_np, (1, 2, 0))
        elif image_np.ndim == 2:
            # Handle grayscale images (2D) - no conversion needed, wandb handles 2D arrays
            pass
        else:
            print(f"Warning: Unexpected image dimensions: {image_np.shape}")
            return
        
        # Ensure image values are in valid range [0, 1] for wandb
        if image_np.dtype != np.uint8:  # If not already uint8
            image_np = np.clip(image_np, 0, 1)  # Clamp to [0, 1] for float images
        
        # Create W&B image
        wandb_image = wandb.Image(image_np, caption=caption)
        
        if step is not None:
            wandb.log({name: wandb_image}, step=step)
        else:
            wandb.log({name: wandb_image})
            
    except Exception as e:
        print(f"Warning: Failed to log image {name}: {e}")


def log_histogram(name, data, step=None):
    """
    Log histogram to W&B.
    
    Args:
        name (str): Histogram name
        data (torch.Tensor or np.ndarray): Data for histogram
        step (int): Step number (optional)
    """
    if wandb.run is None:
        return
        
    try:
        # Convert tensor to numpy if needed
        if isinstance(data, torch.Tensor):
            data_np = data.detach().cpu().numpy().flatten()
        else:
            data_np = data.flatten()
        
        wandb_hist = wandb.Histogram(data_np)
        
        if step is not None:
            wandb.log({name: wandb_hist}, step=step)
        else:
            wandb.log({name: wandb_hist})
            
    except Exception as e:
        print(f"Warning: Failed to log histogram {name}: {e}")


def log_metrics(metrics_dict, step=None):
    """
    Log multiple metrics at once.
    
    Args:
        metrics_dict (dict): Dictionary of metric names and values
        step (int): Step number (optional)
    """
    if wandb.run is None:
        return
        
    try:
        if step is not None:
            wandb.log(metrics_dict, step=step)
        else:
            wandb.log(metrics_dict)
    except Exception as e:
        print(f"Warning: Failed to log metrics: {e}")


def finish_wandb():
    """Finish W&B run."""
    if wandb.run is not None:
        try:
            wandb.finish()
            print("✅ W&B run finished")
        except Exception as e:
            print(f"Warning: Error finishing W&B run: {e}")


def is_wandb_available():
    """Check if W&B is available and initialized."""
    return wandb.run is not None 