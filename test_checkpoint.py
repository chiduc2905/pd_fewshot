"""
Test script using saved checkpoints.

Usage:
    # Test a single checkpoint
    python test_checkpoint.py --dataset original --model covamnet --samples 18 --shot 1
    
    # Test all checkpoints in a folder
    python test_checkpoint.py --all
    
    # Test with a specific checkpoint file
    python test_checkpoint.py --checkpoint checkpoints/original_covamnet_18samples_1shot_best.pth --model covamnet --shot 1
"""

import os
import sys
import argparse
import glob
import re
import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import load_dataset
from dataloader.dataloader import FewshotDataset
from function.function import seed_func
from main import get_model, test_final
import wandb


def get_args():
    parser = argparse.ArgumentParser(description='Test with saved checkpoints')
    
    # Checkpoint specification
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to specific checkpoint file')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints/',
                        help='Directory containing checkpoints')
    
    # Model specification (used to infer checkpoint if not specified)
    parser.add_argument('--model', type=str, default=None,
                        choices=['cosine', 'baseline', 'protonet', 'covamnet', 'matchingnet', 
                                 'relationnet', 'siamese', 'dn4', 'feat', 'deepemd'])
    parser.add_argument('--samples', type=int, default=None,
                        help='Training samples (18, 60, or None for all)')
    parser.add_argument('--shot', type=int, default=1, choices=[1, 5])
    parser.add_argument('--dataset', type=str, default='scalogram',
                        help='Dataset name for new checkpoint format (e.g., original, augmented)')
    parser.add_argument('--backbone', type=str, default='conv64f',
                        choices=['conv64f', 'resnet12', 'resnet18'])
    
    # Test all checkpoints
    parser.add_argument('--all', action='store_true',
                        help='Test all checkpoints in the directory')
    
    # Dataset
    parser.add_argument('--dataset_path', type=str, default='/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/scalogram_v2_split')
    parser.add_argument('--image_size', type=int, default=64, choices=[64, 84])
    parser.add_argument('--way_num', type=int, default=3)
    parser.add_argument('--query_num', type=int, default=1)
    parser.add_argument('--test_episodes', type=int, default=200)
    
    # Other
    parser.add_argument('--path_results', type=str, default='results/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--project', type=str, default='prpd-test',
                        help='WandB project name for test runs')
    
    return parser.parse_args()


def parse_checkpoint_name(filename):
    """
    Parse checkpoint filename to extract dataset, model, samples, and shot.
    
    Expected format: {dataset}_{model}_{samples}_{shot}shot_best.pth
    Examples:
        - original_covamnet_18samples_1shot_best.pth
        - augmented_matchingnet_all_5shot_best.pth
    
    Also supports legacy format: {model}_{samples}_{shot}shot_best.pth
    """
    basename = os.path.basename(filename)
    
    # New pattern: dataset_model_samples_shotshot_best.pth
    pattern_new = r'^(\w+)_(\w+)_(\d+samples|all)_(\d+)shot_best\.pth$'
    match = re.match(pattern_new, basename)
    
    if match:
        dataset = match.group(1)
        model = match.group(2)
        samples_str = match.group(3)
        shot = int(match.group(4))
        
        if samples_str == 'all':
            samples = None
        else:
            samples = int(samples_str.replace('samples', ''))
        
        return {'dataset': dataset, 'model': model, 'samples': samples, 'shot': shot}
    
    # Legacy pattern: model_samples_shotshot_best.pth
    pattern_legacy = r'^(\w+)_(\d+samples|all)_(\d+)shot_best\.pth$'
    match = re.match(pattern_legacy, basename)
    
    if match:
        model = match.group(1)
        samples_str = match.group(2)
        shot = int(match.group(3))
        
        if samples_str == 'all':
            samples = None
        else:
            samples = int(samples_str.replace('samples', ''))
        
        return {'dataset': None, 'model': model, 'samples': samples, 'shot': shot}
    
    return None


def test_single_checkpoint(checkpoint_path, args):
    """Test a single checkpoint."""
    print(f"\n{'='*60}")
    print(f"Testing checkpoint: {checkpoint_path}")
    print(f"{'='*60}")
    
    # Parse checkpoint name or use args
    parsed = parse_checkpoint_name(checkpoint_path)
    if parsed:
        dataset_name = parsed.get('dataset') or args.dataset
        model_name = parsed['model']
        samples = parsed['samples']
        shot_num = parsed['shot']
        print(f"Parsed: dataset={dataset_name}, model={model_name}, samples={samples}, shot={shot_num}")
    else:
        # Fall back to command line args
        if args.model is None:
            print(f"Error: Could not parse checkpoint name and --model not specified")
            return
        dataset_name = args.dataset
        model_name = args.model
        samples = args.samples
        shot_num = args.shot
    
    # Determine image size based on backbone
    if args.backbone in ['resnet12', 'resnet18']:
        image_size = 84
    else:
        image_size = 64
    
    # Create args for model
    class ModelArgs:
        pass
    
    model_args = ModelArgs()
    model_args.model = model_name
    model_args.backbone = args.backbone
    model_args.use_base_encoder = False
    model_args.way_num = args.way_num
    model_args.shot_num = shot_num
    model_args.query_num = args.query_num
    model_args.image_size = image_size
    model_args.path_results = args.path_results
    model_args.training_samples = samples
    model_args.loss = 'contrastive'
    model_args.lambda_center = 0
    model_args.dataset_name = dataset_name
    model_args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize WandB
    samples_str = f"{samples}samples" if samples else "all"
    run_name = f"test_{model_name}_{samples_str}_{shot_num}shot"
    
    wandb.init(
        project=args.project,
        name=run_name,
        config={
            'dataset_name': dataset_name,
            'model': model_name,
            'shot_num': shot_num,
            'training_samples': samples,
            'backbone': args.backbone,
            'checkpoint': checkpoint_path,
            'mode': 'test'
        },
        reinit=True
    )
    
    # Set seed
    seed_func(args.seed)
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    dataset = load_dataset(args.dataset_path, image_size=image_size)
    
    test_X = torch.from_numpy(dataset.X_test.astype(np.float32))
    test_y = torch.from_numpy(dataset.y_test).long()
    
    # Create test loader
    test_ds = FewshotDataset(test_X, test_y, args.test_episodes,
                              args.way_num, shot_num, args.query_num, args.seed)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    # Initialize model
    print(f"Initializing model: {model_name}")
    net = get_model(model_args)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=model_args.device)
    net.load_state_dict(state_dict)
    
    # Run test
    test_final(net, test_loader, model_args)
    
    wandb.finish()
    print(f"Test complete for {checkpoint_path}")


def main():
    args = get_args()
    
    os.makedirs(args.path_results, exist_ok=True)
    
    if args.all:
        # Test all checkpoints
        pattern = os.path.join(args.checkpoints_dir, '*_best.pth')
        checkpoints = sorted(glob.glob(pattern))
        
        if not checkpoints:
            print(f"No checkpoints found in {args.checkpoints_dir}")
            return
        
        print(f"Found {len(checkpoints)} checkpoints to test")
        
        for i, ckpt in enumerate(checkpoints, 1):
            print(f"\n[{i}/{len(checkpoints)}] {os.path.basename(ckpt)}")
            try:
                test_single_checkpoint(ckpt, args)
            except Exception as e:
                print(f"Error testing {ckpt}: {e}")
                continue
    
    elif args.checkpoint:
        # Test specific checkpoint
        if not os.path.exists(args.checkpoint):
            print(f"Checkpoint not found: {args.checkpoint}")
            return
        
        test_single_checkpoint(args.checkpoint, args)
    
    elif args.model:
        # Build checkpoint path from model/samples/shot/dataset
        samples_str = f"{args.samples}samples" if args.samples else "all"
        checkpoint_name = f"{args.dataset}_{args.model}_{samples_str}_{args.shot}shot_best.pth"
        checkpoint_path = os.path.join(args.checkpoints_dir, checkpoint_name)
        
        if not os.path.exists(checkpoint_path):
            # Try legacy format without dataset
            checkpoint_name_legacy = f"{args.model}_{samples_str}_{args.shot}shot_best.pth"
            checkpoint_path_legacy = os.path.join(args.checkpoints_dir, checkpoint_name_legacy)
            if os.path.exists(checkpoint_path_legacy):
                checkpoint_path = checkpoint_path_legacy
                print(f"Using legacy checkpoint format: {checkpoint_name_legacy}")
            else:
                print(f"Checkpoint not found: {checkpoint_path}")
                print(f"Also tried legacy: {checkpoint_path_legacy}")
                print(f"Available checkpoints in {args.checkpoints_dir}:")
                for f in os.listdir(args.checkpoints_dir):
                    if f.endswith('.pth'):
                        print(f"  - {f}")
                return
        
        test_single_checkpoint(checkpoint_path, args)
    
    else:
        print("Please specify one of:")
        print("  --checkpoint <path>          Test a specific checkpoint")
        print("  --model <name> [--samples N] [--shot N]   Build checkpoint path from args")
        print("  --all                        Test all checkpoints in directory")


if __name__ == '__main__':
    main()
