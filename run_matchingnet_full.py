"""Run MatchingNet experiments with all backbones, image sizes, and sample configurations."""
import subprocess
import sys
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Run all MatchingNet experiments')
    parser.add_argument('--project', type=str, default='prpd', help='WandB project name')
    return parser.parse_args()

args = get_args()

# Configuration
backbones = ['conv64f', 'resnet12', 'resnet18']
image_sizes = [64, 84]
samples_list = [18, 60, None]  # None = all
shots = [1, 5]

total = len(backbones) * len(image_sizes) * len(samples_list) * len(shots)
current = 0

print(f"=" * 70)
print(f"MatchingNet Full Experiment Suite")
print(f"  Project: {args.project}")
print(f"  Backbones: {backbones}")
print(f"  Image Sizes: {image_sizes}")
print(f"  Samples: {samples_list}")
print(f"  Shots: {shots}")
print(f"  Total: {len(backbones)} × {len(image_sizes)} × {len(samples_list)} × {len(shots)} = {total} experiments")
print(f"=" * 70)

for backbone in backbones:
    # conv64f baseline only works with 64x64 (uses flatten, not GAP)
    # resnet12/resnet18 work with any size (use Global Average Pooling)
    if backbone == 'conv64f':
        valid_sizes = [64]
    else:
        valid_sizes = image_sizes  # [64, 84]
    
    for image_size in valid_sizes:
        for samples in samples_list:
            for shot in shots:
                current += 1
                samples_str = str(samples) if samples else "All"
                print(f"\n[{current}/{total}] Backbone={backbone}, Size={image_size}, Samples={samples_str}, Shot={shot}")
                
                cmd = [sys.executable, 'main.py',
                       '--model', 'matchingnet',
                       '--backbone', backbone,
                       '--image_size', str(image_size),
                       '--shot_num', str(shot),
                       '--loss', 'contrastive',
                       '--lambda_center', '0',
                       '--mode', 'train',
                       '--project', args.project]
                
                if samples is not None:
                    cmd.extend(['--training_samples', str(samples)])
                
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error: {e}")
                    continue

print(f"\n{'=' * 70}")
print(f"All {total} MatchingNet experiments completed!")
print(f"{'=' * 70}")

print(f"{'=' * 70}")
