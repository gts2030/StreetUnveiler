import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from torch import Tensor
from utils.loss_utils import ssim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class MLPNetwork(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 64, output_dim: int = 1, 
                 net_depth: int = 2, net_activation=F.relu, weight_init: str = 'he_uniform',
                 upscale_factor: int = 14):
        super(MLPNetwork, self).__init__()
        
        self.upscale_factor = upscale_factor
        
        # 기본 MLP
        self.layers = nn.ModuleList()
        for i in range(net_depth):
            dense_layer = nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
            if weight_init == 'he_uniform':
                nn.init.kaiming_uniform_(dense_layer.weight, nonlinearity='relu')
            self.layers.append(dense_layer)
        
        # MLP 출력을 sub-pixel convolution을 위해 확장
        self.output_layer = nn.Linear(hidden_dim, upscale_factor ** 2)
        nn.init.kaiming_uniform_(self.output_layer.weight, nonlinearity='relu')
        
        # Sub-pixel convolution (PixelShuffle)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
        # 후처리 네트워크
        self.post_process = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Softplus()
        )
        
        self.net_activation = net_activation
        self.optimizer = None  # Will be initialized when setup_training is called

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input processing
        H_feat, W_feat, C = x.shape[-3:]
        
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            batch_size = 1
            remove_batch = True
        else:
            batch_size = x.shape[0]
            remove_batch = False

        # MLP processing
        x_flat = x.view(-1, C)
        
        for layer in self.layers:
            x_flat = layer(x_flat)
            x_flat = self.net_activation(x_flat)
            x_flat = F.dropout(x_flat, p=0.2, training=self.training)

        # Output layer - produces upscale_factor^2 channels per pixel
        x_flat = self.output_layer(x_flat)  # (N, upscale_factor^2)
        
        # Reshape for pixel shuffle: (B, upscale_factor^2, H_feat, W_feat)
        x = x_flat.view(batch_size, self.upscale_factor**2, H_feat, W_feat)
        
        # Pixel shuffle: (B, 1, H_feat*upscale_factor, W_feat*upscale_factor)
        x = self.pixel_shuffle(x)
        
        # Post-processing
        x = self.post_process(x)
        
        # Remove dimensions
        if remove_batch:
            x = x.squeeze(0)
        x = x.squeeze(-3) if x.dim() > 2 else x.squeeze()
        
        return x

    def setup_training(self, lr: float = 0.0001, optimizer_type: str = 'adam'):
        """Initialize optimizer for uncertainty MLP training"""
        if optimizer_type.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        print(f"Uncertainty MLP optimizer ({optimizer_type}) initialized with lr={lr}")

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

def generate_uncertainty_mlp(n_features: int, upscale_factor: int = 1, lr: float = 0.0001, setup_training: bool = True) -> MLPNetwork:
    """
    Create uncertainty MLP that outputs at rendered image resolution.
    
    Args:
        n_features: Number of input features
        upscale_factor: Upsampling factor (1 means no upsampling, use interpolation instead)
        lr: Learning rate for optimizer
        setup_training: Whether to initialize optimizer
    """
    network = MLPNetwork(input_dim=n_features, upscale_factor=upscale_factor).cuda()
    
    if setup_training:
        network.setup_training(lr=lr)
    
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
    uncertainty_weighted_main = 0.01 * (ssim_map / (uncertainty_safe**2)).mean()
    
    # Add depth uncertainty term if depth is provided
    if depth_rendered is not None and depth_gt is not None:
        depth_loss = F.l1_loss(depth_rendered, depth_gt, reduction='none')
        if depth_loss.dim() > 2:
            depth_loss = depth_loss.squeeze()
        
        # Weight depth loss by uncertainty
        depth_uncertainty_loss = (depth_loss / (uncertainty_safe**2)).mean()
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

def visualize_uncertainty_heatmap(
    uncertainty: Tensor,
    save_path: str = None,
    cmap: str = 'jet',
    show_colorbar: bool = True,
    title: str = "Uncertainty Heat Map",
    figsize: tuple = (10, 8),
    dpi: int = 150
) -> np.ndarray:
    """
    Visualize uncertainty as heat map.
    
    Args:
        uncertainty: Uncertainty tensor (H, W) or (1, H, W)
        save_path: Optional path to save the figure
        cmap: Colormap name ('jet', 'viridis', 'plasma', 'hot', 'cool', etc.)
        show_colorbar: Whether to show colorbar
        title: Title for the plot
        figsize: Figure size (width, height)
        dpi: DPI for the figure
        
    Returns:
        np.ndarray: RGB image array of the heat map
    """
    # Convert to numpy and ensure 2D
    if isinstance(uncertainty, torch.Tensor):
        uncertainty_np = uncertainty.detach().cpu().numpy()
    else:
        uncertainty_np = uncertainty
        
    if uncertainty_np.ndim == 3 and uncertainty_np.shape[0] == 1:
        uncertainty_np = uncertainty_np.squeeze(0)
    elif uncertainty_np.ndim > 2:
        raise ValueError(f"Uncertainty must be 2D or 3D with shape (1, H, W), got {uncertainty_np.shape}")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    
    # Create heat map
    im = ax.imshow(uncertainty_np, cmap=cmap, aspect='auto')
    
    # Add colorbar if requested
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Uncertainty Value', rotation=270, labelpad=20)
    
    # Set title and remove axes
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Uncertainty heat map saved to: {save_path}")
    
    # Convert to RGB array
    fig.canvas.draw()
    rgb_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    rgb_array = rgb_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # Close figure to free memory
    plt.close(fig)
    
    return rgb_array

def uncertainty_to_colormap(
    uncertainty: Tensor,
    cmap: str = 'jet',
    normalize: bool = True
) -> Tensor:
    """
    Convert uncertainty tensor to colormap RGB tensor.
    
    Args:
        uncertainty: Uncertainty tensor (H, W) or (1, H, W)
        cmap: Colormap name
        normalize: Whether to normalize uncertainty to [0, 1] range
        
    Returns:
        torch.Tensor: RGB tensor (3, H, W) with values in [0, 1]
    """
    # Convert to numpy and ensure 2D
    if isinstance(uncertainty, torch.Tensor):
        uncertainty_np = uncertainty.detach().cpu().numpy()
        device = uncertainty.device
    else:
        uncertainty_np = uncertainty
        device = torch.device('cpu')
        
    if uncertainty_np.ndim == 3 and uncertainty_np.shape[0] == 1:
        uncertainty_np = uncertainty_np.squeeze(0)
    elif uncertainty_np.ndim > 2:
        raise ValueError(f"Uncertainty must be 2D or 3D with shape (1, H, W), got {uncertainty_np.shape}")
    
    # Normalize if requested
    if normalize:
        u_min, u_max = uncertainty_np.min(), uncertainty_np.max()
        if u_max > u_min:
            uncertainty_np = (uncertainty_np - u_min) / (u_max - u_min)
        else:
            uncertainty_np = np.zeros_like(uncertainty_np)
    
    # Apply colormap
    colormap = cm.get_cmap(cmap)
    rgb_array = colormap(uncertainty_np)[:, :, :3]  # Remove alpha channel
    
    # Convert back to tensor (3, H, W)
    rgb_tensor = torch.from_numpy(rgb_array).float().permute(2, 0, 1).to(device)
    
    return rgb_tensor

def save_uncertainty_analysis(
    uncertainty: Tensor,
    save_dir: str,
    iteration: int,
    frame_id: int = None,
    additional_info: dict = None
):
    """
    Save comprehensive uncertainty analysis including heat map and statistics.
    
    Args:
        uncertainty: Uncertainty tensor (H, W)
        save_dir: Directory to save analysis files
        iteration: Training iteration number
        frame_id: Optional frame ID
        additional_info: Optional dictionary with additional information to save
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy
    uncertainty_np = uncertainty.detach().cpu().numpy()
    
    # Generate filename suffix
    suffix = f"iter_{iteration}"
    if frame_id is not None:
        suffix += f"_frame_{frame_id}"
    
    # Save heat map with different colormaps
    colormaps = ['jet', 'viridis', 'plasma', 'hot']
    for cmap in colormaps:
        save_path = os.path.join(save_dir, f"uncertainty_heatmap_{cmap}_{suffix}.png")
        visualize_uncertainty_heatmap(
            uncertainty,
            save_path=save_path,
            cmap=cmap,
            title=f"Uncertainty Heat Map ({cmap}) - Iteration {iteration}"
        )
    
    # Save uncertainty statistics
    stats = {
        'iteration': iteration,
        'frame_id': frame_id,
        'mean': float(uncertainty_np.mean()),
        'std': float(uncertainty_np.std()),
        'min': float(uncertainty_np.min()),
        'max': float(uncertainty_np.max()),
        'median': float(np.median(uncertainty_np.flatten())),
        'percentile_95': float(np.percentile(uncertainty_np.flatten(), 95)),
        'percentile_99': float(np.percentile(uncertainty_np.flatten(), 99)),
        'shape': uncertainty_np.shape
    }
    
    if additional_info is not None:
        stats.update(additional_info)
    
    # Save statistics as text file
    stats_path = os.path.join(save_dir, f"uncertainty_stats_{suffix}.txt")
    with open(stats_path, 'w') as f:
        f.write(f"Uncertainty Analysis - Iteration {iteration}\n")
        f.write("=" * 50 + "\n\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    # Save raw uncertainty as numpy array
    uncertainty_path = os.path.join(save_dir, f"uncertainty_raw_{suffix}.npy")
    np.save(uncertainty_path, uncertainty_np)
    
    print(f"Uncertainty analysis saved to: {save_dir}")
    print(f"  - Mean uncertainty: {stats['mean']:.6f}")
    print(f"  - Std uncertainty: {stats['std']:.6f}")
    print(f"  - Min/Max uncertainty: {stats['min']:.6f} / {stats['max']:.6f}")