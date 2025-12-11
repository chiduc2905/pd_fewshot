"""Professional feature map visualization for few-shot learning encoders.

This module provides hook-based feature extraction and visualization tools
for analyzing intermediate representations in few-shot learning models.

Features:
- Hook-based extraction from any layer
- Multi-channel activation grids
- Heatmap overlay on original images
- Support for all encoder types (Conv64F, ResNet12, ResNet18)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union


class FeatureVisualizer:
    """
    Professional feature map visualizer using PyTorch hooks.
    
    Example usage:
        >>> from net.matchingnet import MatchingNet
        >>> model = MatchingNet(backbone='resnet18')
        >>> visualizer = FeatureVisualizer(model.encoder)
        >>> image = load_image('scalogram.png')
        >>> features = visualizer.extract_all_layers(image)
        >>> visualizer.plot_feature_grid(features['layer4'], save_path='output.png')
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize visualizer with a model.
        
        Args:
            model: PyTorch model (encoder) to visualize
        """
        self.model = model
        self.model.eval()
        self.features: Dict[str, torch.Tensor] = {}
        self.hooks: List = []
        
    def _get_hook(self, name: str):
        """Create a hook function that saves output to self.features."""
        def hook(module, input, output):
            # Handle both single tensor and tuple outputs
            if isinstance(output, tuple):
                self.features[name] = output[0].detach().cpu()
            else:
                self.features[name] = output.detach().cpu()
        return hook
    
    def register_hooks(self, layer_names: Optional[List[str]] = None):
        """
        Register hooks on specified layers or all conv/batchnorm layers.
        
        Args:
            layer_names: List of layer names to hook. If None, auto-detect conv layers.
        """
        self.clear_hooks()
        
        if layer_names is None:
            # Auto-detect: register hooks on all Conv2d and key Sequential modules
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Sequential)):
                    if isinstance(module, nn.Conv2d):
                        hook = module.register_forward_hook(self._get_hook(name))
                        self.hooks.append(hook)
                    elif name and 'layer' in name.lower():
                        # Hook on layer-level modules (layer1, layer2, etc.)
                        hook = module.register_forward_hook(self._get_hook(name))
                        self.hooks.append(hook)
        else:
            # Register on specific named layers
            name_to_module = dict(self.model.named_modules())
            for name in layer_names:
                if name in name_to_module:
                    hook = name_to_module[name].register_forward_hook(self._get_hook(name))
                    self.hooks.append(hook)
                else:
                    print(f"Warning: Layer '{name}' not found in model.")
    
    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.features = {}
    
    def extract_features(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run forward pass and collect features from hooked layers.
        
        Args:
            image: Input tensor of shape (B, C, H, W) or (C, H, W)
        
        Returns:
            Dictionary mapping layer names to feature tensors
        """
        self.features = {}
        
        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        with torch.no_grad():
            _ = self.model(image)
        
        return self.features.copy()
    
    def extract_all_layers(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from all detected conv/layer modules.
        
        Args:
            image: Input tensor
            
        Returns:
            Dictionary of layer_name -> feature_tensor
        """
        self.register_hooks(layer_names=None)
        features = self.extract_features(image)
        self.clear_hooks()
        return features
    
    def __del__(self):
        """Cleanup hooks on deletion."""
        self.clear_hooks()


def extract_features(model: nn.Module, image: torch.Tensor, 
                     layer_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
    """
    Convenience function to extract features without creating a visualizer instance.
    
    Args:
        model: PyTorch model (encoder)
        image: Input tensor (B, C, H, W) or (C, H, W)
        layer_names: Specific layers to extract from (None = auto-detect)
    
    Returns:
        Dictionary mapping layer names to feature tensors
    """
    visualizer = FeatureVisualizer(model)
    visualizer.register_hooks(layer_names)
    features = visualizer.extract_features(image)
    visualizer.clear_hooks()
    return features


def create_activation_grid(
    feature_map: torch.Tensor,
    num_channels: int = 16,
    nrow: Optional[int] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Create a grid of activation maps from a feature tensor.
    
    Args:
        feature_map: Feature tensor of shape (B, C, H, W) or (C, H, W)
        num_channels: Number of channels to display (first N channels)
        nrow: Number of images per row (auto-calculated if None)
        normalize: Whether to normalize each channel to [0, 1]
    
    Returns:
        Grid image as numpy array (H_grid, W_grid)
    """
    # Handle batch dimension
    if feature_map.dim() == 4:
        feature_map = feature_map[0]  # Take first batch element
    
    C, H, W = feature_map.shape
    num_channels = min(num_channels, C)
    
    # Auto-calculate grid size
    if nrow is None:
        nrow = int(np.ceil(np.sqrt(num_channels)))
    ncol = int(np.ceil(num_channels / nrow))
    
    # Create grid
    grid_h = nrow * H + (nrow - 1) * 2  # 2px padding between cells
    grid_w = ncol * W + (ncol - 1) * 2
    grid = np.zeros((grid_h, grid_w))
    
    for idx in range(num_channels):
        row = idx // ncol
        col = idx % ncol
        
        activation = feature_map[idx].numpy()
        
        if normalize:
            # Normalize to [0, 1]
            a_min, a_max = activation.min(), activation.max()
            if a_max > a_min:
                activation = (activation - a_min) / (a_max - a_min)
            else:
                activation = np.zeros_like(activation)
        
        y_start = row * (H + 2)
        x_start = col * (W + 2)
        grid[y_start:y_start + H, x_start:x_start + W] = activation
    
    return grid


def visualize_feature_maps(
    feature_map: torch.Tensor,
    original_image: Optional[Union[torch.Tensor, np.ndarray]] = None,
    num_channels: int = 16,
    figsize: Tuple[int, int] = (12, 12),
    cmap: str = 'viridis',
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    show_colorbar: bool = True,
) -> plt.Figure:
    """
    Create a professional visualization of feature maps.
    
    Args:
        feature_map: Feature tensor (B, C, H, W) or (C, H, W)
        original_image: Optional original image to show alongside (for reference)
        num_channels: Number of channels to display
        figsize: Figure size
        cmap: Colormap for feature maps
        save_path: Path to save figure (None = don't save)
        title: Figure title
        show_colorbar: Whether to show colorbar
    
    Returns:
        matplotlib Figure object
    """
    # Create grid of activations
    grid = create_activation_grid(feature_map, num_channels=num_channels)
    
    # Create figure
    if original_image is not None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        ax_orig, ax_feat = axes
        
        # Show original image
        if isinstance(original_image, torch.Tensor):
            if original_image.dim() == 4:
                original_image = original_image[0]
            # CHW -> HWC
            img_np = original_image.permute(1, 2, 0).numpy()
            # Handle normalization
            if img_np.min() < 0:
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        else:
            img_np = original_image
        
        ax_orig.imshow(img_np)
        ax_orig.set_title('Original Image')
        ax_orig.axis('off')
    else:
        fig, ax_feat = plt.subplots(1, 1, figsize=(figsize[0] // 2, figsize[1]))
    
    # Show feature map grid
    im = ax_feat.imshow(grid, cmap=cmap)
    ax_feat.set_title(f'Feature Maps ({num_channels} channels)')
    ax_feat.axis('off')
    
    if show_colorbar:
        plt.colorbar(im, ax=ax_feat, fraction=0.046, pad=0.04)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved feature visualization to: {save_path}")
    
    return fig


def create_heatmap_overlay(
    feature_map: torch.Tensor,
    original_image: Union[torch.Tensor, np.ndarray],
    alpha: float = 0.5,
    colormap: str = 'jet',
) -> np.ndarray:
    """
    Create a heatmap overlay of feature activations on the original image.
    
    Args:
        feature_map: Feature tensor (B, C, H, W) or (C, H, W)
        original_image: Original image tensor or numpy array
        alpha: Transparency of heatmap overlay
        colormap: Colormap for heatmap
    
    Returns:
        Overlay image as numpy array (H, W, 3)
    """
    # Get mean activation across channels
    if feature_map.dim() == 4:
        feature_map = feature_map[0]
    
    # Average across channels
    mean_activation = feature_map.mean(dim=0).numpy()  # (H, W)
    
    # Normalize to [0, 1]
    a_min, a_max = mean_activation.min(), mean_activation.max()
    if a_max > a_min:
        mean_activation = (mean_activation - a_min) / (a_max - a_min)
    
    # Convert original image
    if isinstance(original_image, torch.Tensor):
        if original_image.dim() == 4:
            original_image = original_image[0]
        img_np = original_image.permute(1, 2, 0).numpy()
        if img_np.min() < 0:
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    else:
        img_np = original_image.copy()
    
    # Resize activation to match image size
    img_h, img_w = img_np.shape[:2]
    activation_resized = np.array(
        Image.fromarray((mean_activation * 255).astype(np.uint8)).resize((img_w, img_h))
    ) / 255.0
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    heatmap = cmap(activation_resized)[:, :, :3]  # Remove alpha channel
    
    # Blend with original image
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    elif img_np.shape[-1] == 1:
        img_np = np.repeat(img_np, 3, axis=-1)
    
    overlay = (1 - alpha) * img_np + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    
    return overlay


def visualize_layer_progression(
    model: nn.Module,
    image: torch.Tensor,
    save_dir: str = 'results/feature_maps',
    figsize: Tuple[int, int] = (15, 5),
) -> List[str]:
    """
    Visualize feature progression through all layers.
    
    Args:
        model: Encoder model
        image: Input image tensor
        save_dir: Directory to save visualizations
        figsize: Figure size per layer
    
    Returns:
        List of saved file paths
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract features from all layers
    visualizer = FeatureVisualizer(model)
    features = visualizer.extract_all_layers(image)
    
    saved_paths = []
    
    for i, (layer_name, feat) in enumerate(features.items()):
        if feat.dim() >= 4 or (feat.dim() == 3 and feat.shape[0] > 3):
            # Only visualize spatial feature maps
            safe_name = layer_name.replace('.', '_')
            save_path = os.path.join(save_dir, f'{i:02d}_{safe_name}.png')
            
            fig = visualize_feature_maps(
                feat,
                original_image=image if i == 0 else None,
                num_channels=min(16, feat.shape[1] if feat.dim() == 4 else feat.shape[0]),
                title=f'Layer: {layer_name} | Shape: {list(feat.shape)}',
                save_path=save_path,
                figsize=figsize,
            )
            plt.close(fig)
            saved_paths.append(save_path)
    
    print(f"Saved {len(saved_paths)} feature visualizations to {save_dir}/")
    return saved_paths
