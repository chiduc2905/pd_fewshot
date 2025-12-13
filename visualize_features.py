#!/usr/bin/env python
"""Feature Map Visualization CLI for Few-Shot Learning Models.

This script provides a command-line interface to visualize feature maps
from trained few-shot learning encoders.

Usage Examples:
    # Visualize features from MatchingNet with resnet18 backbone
    python visualize_features.py --model matchingnet --backbone resnet18 \
        --image scalogram/corona/sample.png

    # Visualize all layers with heatmap overlay
    python visualize_features.py --model protonet --image image.png \
        --output results/feature_maps --overlay
    
    # List available layers in a model
    python visualize_features.py --model matchingnet --backbone resnet12 --list-layers
"""
import os
import sys
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize feature maps from few-shot learning encoders',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model selection
    parser.add_argument('--model', type=str, default='matchingnet',
                       choices=['matchingnet', 'protonet', 'relationnet', 'covamnet', 'cosine'],
                       help='Model type (default: matchingnet)')
    parser.add_argument('--backbone', type=str, default='conv64f',
                       choices=['conv64f', 'resnet12', 'resnet18'],
                       help='Backbone for MatchingNet (default: conv64f)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (optional)')
    
    # Input
    parser.add_argument('--image', type=str, required=False,
                       help='Path to input image (required unless --list-layers)')
    parser.add_argument('--image-size', type=int, default=64,
                       help='Input image size (default: 64)')
    
    # Output
    parser.add_argument('--output', type=str, default='results/feature_maps',
                       help='Output directory for visualizations')
    parser.add_argument('--num-channels', type=int, default=16,
                       help='Number of channels to visualize (default: 16)')
    
    # Visualization options
    parser.add_argument('--overlay', action='store_true',
                       help='Create heatmap overlay on original image')
    parser.add_argument('--all-layers', action='store_true',
                       help='Visualize all layers (progression view)')
    parser.add_argument('--layer', type=str, default=None,
                       help='Specific layer to visualize (e.g., "layer4")')
    parser.add_argument('--cmap', type=str, default='viridis',
                       help='Colormap for feature maps (default: viridis)')
    
    # Utility
    parser.add_argument('--list-layers', action='store_true',
                       help='List available layers in the model and exit')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use (default: auto)')
    
    return parser.parse_args()


def load_model(args):
    """Load the specified model."""
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.model == 'matchingnet':
        from net.matchingnet import MatchingNet
        model = MatchingNet(backbone=args.backbone, device=device)
        encoder = model.encoder
    elif args.model == 'protonet':
        from net.protonet import ProtoNet
        model = ProtoNet(device=device)
        encoder = model.encoder
    elif args.model == 'relationnet':
        from net.relationnet import RelationNet
        model = RelationNet(device=device)
        encoder = model.encoder
    elif args.model == 'covamnet':
        from net.covamnet import CovaMNet
        model = CovaMNet(device=device)
        encoder = model.encoder
    elif args.model == 'cosine':
        from net.cosine import CosineNet
        model = CosineNet(device=device)
        encoder = model.encoder
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    encoder.eval()
    
    return encoder, device


def load_image(image_path: str, image_size: int, device: str) -> torch.Tensor:
    """Load and preprocess an image."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    
    return tensor


def list_model_layers(encoder):
    """Print all named modules in the encoder."""
    print("\n" + "=" * 60)
    print("Available Layers in Encoder")
    print("=" * 60)
    
    for name, module in encoder.named_modules():
        if name:  # Skip the root module
            module_type = module.__class__.__name__
            # Get shape info if possible
            shape_info = ""
            if hasattr(module, 'weight') and module.weight is not None:
                shape_info = f" | weight: {list(module.weight.shape)}"
            print(f"  {name:<30} ({module_type}){shape_info}")
    
    print("=" * 60 + "\n")


def main():
    args = get_args()
    
    # Load model
    print(f"\nLoading model: {args.model}", end="")
    if args.model == 'matchingnet':
        print(f" (backbone: {args.backbone})")
    else:
        print()
    
    encoder, device = load_model(args)
    
    # List layers mode
    if args.list_layers:
        list_model_layers(encoder)
        return
    
    # Require image for visualization
    if not args.image:
        print("Error: --image is required for visualization")
        sys.exit(1)
    
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    # Import visualization module
    from visualization.feature_visualizer import (
        FeatureVisualizer,
        visualize_feature_maps,
        visualize_layer_progression,
        create_heatmap_overlay,
    )
    import matplotlib.pyplot as plt
    
    # Load image
    print(f"Loading image: {args.image}")
    image_tensor = load_image(args.image, args.image_size, device)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Visualization
    if args.all_layers:
        # Visualize all layers
        print(f"\nVisualizing all layers...")
        saved_paths = visualize_layer_progression(
            encoder.cpu(), image_tensor.cpu(), save_dir=args.output
        )
        print(f"Created {len(saved_paths)} visualizations")
        
    else:
        # Single layer or default visualization
        visualizer = FeatureVisualizer(encoder.cpu())
        
        if args.layer:
            # Specific layer
            visualizer.register_hooks([args.layer])
        else:
            # Auto-detect (will use all conv layers)
            visualizer.register_hooks()
        
        features = visualizer.extract_features(image_tensor.cpu())
        
        if not features:
            print("No features extracted. Try --list-layers to see available layers.")
            return
        
        # Visualize each extracted layer
        for layer_name, feat in features.items():
            print(f"Visualizing layer: {layer_name} | Shape: {list(feat.shape)}")
            
            if feat.dim() < 3:
                print(f"  Skipping (not a spatial feature map)")
                continue
            
            safe_name = layer_name.replace('.', '_')
            save_path = os.path.join(args.output, f'{safe_name}.png')
            
            # Feature map grid
            fig = visualize_feature_maps(
                feat,
                original_image=image_tensor.cpu() if args.overlay else None,
                num_channels=args.num_channels,
                cmap=args.cmap,
                title=f'{args.model.upper()} - {layer_name}',
                save_path=save_path,
            )
            plt.close(fig)
            
            # Heatmap overlay
            if args.overlay and feat.dim() >= 3:
                overlay = create_heatmap_overlay(feat, image_tensor.cpu())
                overlay_path = os.path.join(args.output, f'{safe_name}_overlay.png')
                plt.imsave(overlay_path, overlay)
                print(f"  Saved overlay: {overlay_path}")
        
        visualizer.clear_hooks()
    
    print(f"\n✓ Visualizations saved to: {args.output}/")


if __name__ == '__main__':
    main()
