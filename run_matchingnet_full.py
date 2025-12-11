"""Run MatchingNet experiments with all backbones, image sizes, and sample configurations."""
import subprocess
import sys

# Configuration
backbones = ['conv64f', 'resnet12', 'resnet18']
image_sizes = [64, 84]
samples_list = [18, 60, None]  # None = all
shots = [1, 5]

total = len(backbones) * len(image_sizes) * len(samples_list) * len(shots)
current = 0

print(f"=" * 70)
print(f"MatchingNet Full Experiment Suite")
print(f"  Backbones: {backbones}")
print(f"  Image Sizes: {image_sizes}")
print(f"  Samples: {samples_list}")
print(f"  Shots: {shots}")
print(f"  Total: {len(backbones)} × {len(image_sizes)} × {len(samples_list)} × {len(shots)} = {total} experiments")
print(f"=" * 70)

for backbone in backbones:
    for image_size in image_sizes:
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
                       '--mode', 'train']
                
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
