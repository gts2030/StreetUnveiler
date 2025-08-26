import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from torch import Tensor
from utils.loss_utils import ssim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from threading import Lock

class MLPNetwork(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 64, output_dim: int = 1, 
                 net_depth: int = 2, net_activation=F.relu, weight_init: str = 'he_uniform',
                 upscale_factor: int = 1):  # Keep for compatibility but default to 1
        super(MLPNetwork, self).__init__()
        
        self.output_layer_input_dim = hidden_dim
        
        # Initialize MLP layers
        self.layers = nn.ModuleList()
        for i in range(net_depth):
            dense_layer = nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
            
            # Apply weight initialization
            if weight_init == 'he_uniform':
                nn.init.kaiming_uniform_(dense_layer.weight, nonlinearity='relu')
            elif weight_init == 'xavier_uniform':
                nn.init.xavier_uniform_(dense_layer.weight)
            else:
                raise NotImplementedError(f"Unknown Weight initialization method {weight_init}")

            self.layers.append(dense_layer)
        
        # Initialize output layer
        self.output_layer = nn.Linear(self.output_layer_input_dim, output_dim)
        nn.init.kaiming_uniform_(self.output_layer.weight, nonlinearity='relu')
        
        # Set activation function
        self.net_activation = net_activation
        self.softplus = nn.Softplus()
        self.optimizer = None  # Will be initialized when setup_training is called

    def forward(self, x: torch.Tensor, target_size: tuple = None) -> torch.Tensor:
        """
        Forward pass with optional target size for interpolation.
        
        Args:
            x: Input features (H, W, C) or (B, H, W, C)
            target_size: Optional (H, W) for output interpolation
        
        Returns:
            Uncertainty map, optionally resized to target_size
        """
        # Get input dimensions
        H, W, C = x.shape[-3:]
        input_with_batch_dim = True
        
        # Add batch dimension if not present
        if len(x.shape) == 3:
            input_with_batch_dim = False
            x = x.unsqueeze(0)
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # Flatten input for MLP
        x = x.view(-1, x.size()[-1])
        
        # Pass through MLP layers
        for layer in self.layers:
            x = layer(x)
            x = self.net_activation(x)
            x = F.dropout(x, p=0.2, training=self.training)

        # Pass through output layer and apply softplus activation
        x = self.output_layer(x)
        x = self.softplus(x)

        # Reshape output to original feature dimensions
        if input_with_batch_dim:
            x = x.view(batch_size, H, W)
        else:
            x = x.view(H, W)

        # Interpolate to target size if specified
        if target_size is not None and (H, W) != target_size:
            if not input_with_batch_dim:
                x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
                x = x.squeeze(0).squeeze(0)  # Remove batch and channel dims
            else:
                x = x.unsqueeze(1)  # Add channel dim
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
                x = x.squeeze(1)  # Remove channel dim

        return x

    def setup_training(self, lr: float = 0.0004, weight_decay: float = 0.00001, optimizer_type: str = 'adam'):
        """Initialize optimizer for uncertainty MLP training"""
        if optimizer_type.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        print(f"Uncertainty MLP optimizer ({optimizer_type}) initialized with lr={lr}, weight_decay={weight_decay}")

    def step_optimizer(self):
        """Step the optimizer and zero gradients"""
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        else:
            print("Warning: Optimizer not initialized. Call setup_training() first.")

    def save_checkpoint(self, checkpoint_path: str, iteration: int):
        """Save uncertainty MLP checkpoint"""
        import os
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        checkpoint_data = {
            'model_state_dict': self.state_dict(),
            'iteration': iteration
        }
        
        # Add optimizer state if available
        if self.optimizer is not None:
            checkpoint_data['optimizer_state_dict'] = self.optimizer.state_dict()
        
        torch.save(checkpoint_data, checkpoint_path)
        print(f"Uncertainty MLP checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str, setup_optimizer: bool = True, lr: float = 0.0001):
        """Load uncertainty MLP checkpoint"""
        import os
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Uncertainty MLP checkpoint not found at {checkpoint_path}")
            return False
        
        checkpoint_data = torch.load(checkpoint_path, weights_only=False)
        
        # Load model state
        self.load_state_dict(checkpoint_data['model_state_dict'])
        
        # Setup and load optimizer if requested
        if setup_optimizer:
            if self.optimizer is None:
                self.setup_training(lr=lr)
            
            if 'optimizer_state_dict' in checkpoint_data:
                self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        iteration = checkpoint_data.get('iteration', -1)
        print(f"Uncertainty MLP checkpoint loaded: {checkpoint_path} (iteration {iteration})")
        return True

def generate_uncertainty_mlp(n_features: int, lr: float = 0.0004, weight_decay: float = 0.00001, 
                            setup_training: bool = True, hidden_dim: int = 64, net_depth: int = 2) -> MLPNetwork:
    """
    Create uncertainty MLP for per-pixel uncertainty prediction.
    
    Args:
        n_features: Number of input features from feature extractor
        lr: Learning rate for optimizer
        weight_decay: Weight decay for optimizer regularization
        setup_training: Whether to initialize optimizer
        hidden_dim: Hidden layer dimension
        net_depth: Number of hidden layers
    
    Returns:
        MLPNetwork instance on CUDA
    """
    network = MLPNetwork(
        input_dim=n_features,
        hidden_dim=hidden_dim,
        output_dim=1,
        net_depth=net_depth,
        net_activation=F.relu,
        weight_init='he_uniform'
    ).cuda()
    
    if setup_training:
        network.setup_training(lr=lr, weight_decay=weight_decay)
    
    return network

def compute_uncertainty_weighted_loss(
    uncertainty: Tensor,
    l1_loss: Tensor,
    ssim_map: Tensor,
    depth_rendered: Optional[Tensor] = None,
    depth_gt: Optional[Tensor] = None,
    lambda1: float = 1.0,
    lambda2: float = 0.01,
    lambda3: float = 0.01,
    eps: float = 1e-6
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute uncertainty loss following the formula:
    L_uncer = (L_SSIM + λ1 * L_uncer_D) / β_i^2 + λ2 * L_reg_V + λ3 * L_reg_U
    
    Args:
        uncertainty: Per-pixel uncertainty estimates β_i (H, W)
        l1_loss: L1 reconstruction loss (scalar or H, W)
        ssim_map: SSIM map (H, W)
        depth_rendered: Rendered depth (optional, for depth uncertainty)
        depth_gt: Ground truth depth (optional, for depth uncertainty)
        lambda1: Weight for depth uncertainty term
        lambda2: Weight for variance regularization
        lambda3: Weight for uncertainty regularization
        eps: Small value to prevent division by zero
        
    Returns:
        uncertainty_weighted_loss: Main uncertainty-weighted loss
        reg_v: Variance regularization term
        reg_u: Uncertainty regularization term
    """
    # Handle multi-channel ssim_map (e.g., RGB channels)
    if ssim_map.dim() == 3 and ssim_map.shape[0] == 3:
        # Average across RGB channels to get 2D map
        ssim_map = ssim_map.mean(dim=0)
    
    # Ensure uncertainty has minimum value to prevent division by zero
    uncertainty_safe = torch.clamp(uncertainty, min=eps)
    
    # Resize uncertainty to match ssim_map if shapes don't match
    if uncertainty_safe.shape != ssim_map.shape:
        print(f"[DEBUG] Resizing uncertainty from {uncertainty_safe.shape} to {ssim_map.shape}")
        if uncertainty_safe.dim() == 2:
            uncertainty_safe = uncertainty_safe.unsqueeze(0).unsqueeze(0)
        elif uncertainty_safe.dim() == 3:
            uncertainty_safe = uncertainty_safe.unsqueeze(0)
        
        uncertainty_safe = F.interpolate(
            uncertainty_safe,
            size=ssim_map.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        uncertainty_safe = uncertainty_safe.squeeze()
    
    # Main uncertainty-weighted loss: (L_SSIM + λ1 * L_uncer_D) / β_i^2
    uncertainty_weighted_main = 0.001*(ssim_map / (uncertainty_safe**2)).mean()
    
    # Add depth uncertainty term if depth is provided
    if depth_rendered is not None and depth_gt is not None:
        # Ensure both depths have the same shape
        depth_rendered_2d = depth_rendered.squeeze() if depth_rendered.dim() > 2 else depth_rendered
        depth_gt_2d = depth_gt.squeeze() if depth_gt.dim() > 2 else depth_gt
        
        # Resize depth_gt to match depth_rendered if shapes are different
        if depth_rendered_2d.shape != depth_gt_2d.shape:
            depth_gt_2d = F.interpolate(
                depth_gt_2d.unsqueeze(0).unsqueeze(0),
                size=depth_rendered_2d.shape,
                mode='bilinear',
                align_corners=False
            ).squeeze()
        
        depth_loss = F.l1_loss(depth_rendered_2d, depth_gt_2d, reduction='none')
        
        # Weight depth loss by uncertainty
        depth_uncertainty_loss = 10*(depth_loss / (uncertainty_safe**2)).mean()
        uncertainty_weighted_main += lambda1 * depth_uncertainty_loss
    
    # Regularization terms
    # L_reg_V: Variance regularization (encourage spatial smoothness)
    reg_v = torch.var(uncertainty)
    
    # L_reg_U: Uncertainty regularization (prevent extreme values)
    reg_u = torch.log(uncertainty_safe).mean()
    
    # Total uncertainty loss
    total_uncertainty_loss = uncertainty_weighted_main + lambda2 * reg_v + lambda3 * reg_u
    
    return total_uncertainty_loss, reg_v, reg_u

def get_uncertainty_and_loss(
    features: Tensor,
    uncertainty_network: MLPNetwork,
    l1_loss: Tensor,
    ssim_loss: Tensor,
    depth_rendered: Optional[Tensor] = None,
    depth_gt: Optional[Tensor] = None,
    lambda1: float = 1.0,
    lambda2: float = 0.01,
    lambda3: float = 0.01
) -> Tuple[Tensor, Tensor]:
    """
    Predict uncertainty and compute uncertainty loss.
    
    Args:
        features: Image features for uncertainty prediction (H, W, C)
        uncertainty_network: MLP for uncertainty prediction
        l1_loss: L1 reconstruction loss (scalar or H, W)
        ssim_loss: SSIM loss (scalar)
        depth_rendered: Rendered depth (optional)
        depth_gt: Ground truth depth (optional)
        lambda1: Weight for depth uncertainty term
        lambda2: Weight for variance regularization  
        lambda3: Weight for uncertainty regularization
        
    Returns:
        uncertainty: Per-pixel uncertainty estimates (H, W)
        uncertainty_loss: Total uncertainty loss (scalar)
    """
    # Predict uncertainty from features
    uncertainty = uncertainty_network(features)  # (H, W)
    
    # Compute uncertainty loss
    uncertainty_loss, reg_v, reg_u = compute_uncertainty_weighted_loss(
        uncertainty, 
        l1_loss,
        ssim_loss,
        depth_rendered,
        depth_gt,
        lambda1,
        lambda2,
        lambda3
    )
    
    return uncertainty, uncertainty_loss

# Keep the old complex function for backward compatibility but mark as deprecated
def get_loss_mapping_uncertainty(
    config: Dict,
    rendered_img: Tensor,
    rendered_depth: Tensor,
    viewpoint, # from src.utils.camera_utils import Camera, to avoid loop import
    opacity: Tensor,
    uncertainty_network: MLPNetwork,
    train_frac: float,
    ssim_frac: float,
    initialization: bool = False,
    freeze_uncertainty_loss: bool = False,  # Renamed parameter
) -> Tuple[Tensor, Tensor]:
    """
    [DEPRECATED] Original complex uncertainty function for backward compatibility.
    Use get_uncertainty_and_loss instead.
    """
    print("Warning: get_loss_mapping_uncertainty is deprecated. Use get_uncertainty_and_loss instead.")
    
    # Simple fallback implementation
    if hasattr(viewpoint, 'features'):
        uncertainty = uncertainty_network(viewpoint.features)
        dummy_loss = torch.tensor(0.0, device=rendered_img.device)
        return uncertainty, dummy_loss
    else:
        # Return dummy values if no features available
        H, W = rendered_img.shape[-2:]
        dummy_uncertainty = torch.ones(H, W, device=rendered_img.device)
        dummy_loss = torch.tensor(0.0, device=rendered_img.device)
        return dummy_uncertainty, dummy_loss

def get_viewpoint_uncertainty_no_grad(uncertainty_network: 'MLPNetwork', viewpoint, uncer_params: dict = None, device: str = "cuda") -> torch.Tensor:
    """
    Compute the uncertainty for a given viewpoint without gradient computation.
    This function follows the exact same processing pipeline as used in mapping loss computation.
    
    Args:
        uncertainty_network: The uncertainty MLP network
        viewpoint: Camera viewpoint with features attribute (containing DINO features)
        uncer_params: Dictionary containing uncertainty parameters (especially train_frac_fix)
        device: Device to run computation on
        
    Returns:
        uncertainty_adjusted: Processed uncertainty values (uncertainty^2)
    """
    if not hasattr(viewpoint, 'features') or viewpoint.features is None:
        # Return default uncertainty if no features available
        H, W = viewpoint.image_height, viewpoint.image_width
        return torch.ones((H, W), device=device) * 0.01
    
    features = viewpoint.features.to(device)
    
    with torch.no_grad():
        # Use Lock for thread safety if needed (from original function signature)
        with Lock():
            uncertainty = uncertainty_network(features)

        # Process uncertainty values - same as in compute_mapping_loss_components
        uncertainty = torch.clip(uncertainty, min=0.1) + 1e-3
        target_shape = (viewpoint.image_height, viewpoint.image_width)
        
        # Use resample_tensor_to_shape from mapping_utils
        from utils.dyn_uncertainty.mapping_utils import resample_tensor_to_shape, compute_bias_factor
        uncertainty_resized = resample_tensor_to_shape(
            uncertainty, target_shape
        )

        # Apply data rate adjustment, the same as how we calculate the loss function
        if uncer_params is not None:
            train_frac = uncer_params.get("train_frac_fix", 0.5)
        else:
            train_frac = getattr(uncertainty_network, 'uncer_params', {}).get("train_frac_fix", 0.5)
        
        data_rate = 1 + compute_bias_factor(train_frac, 0.8)
        uncertainty_adjusted = (uncertainty_resized - 0.1) * data_rate + 0.1

        return uncertainty_adjusted ** 2
