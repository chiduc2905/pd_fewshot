import subprocess
import sys
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Run all experiments')
    parser.add_argument('--project', type=str, default='prpd', help='WandB project name')
    return parser.parse_args()

# Configuration
# Models that ONLY use 64x64 (paper standard)
base_models = ['covamnet', 'protonet', 'cosine', 'relationnet']

# MatchingNet variants (support multiple image sizes)
matchingnet_variants = ['matchingnet', 'matchingnet_resnet12', 'matchingnet_resnet18']

shots = [1, 5]
samples_list = [18, 60, None]  # None means all samples
lambda_center = 0

args = get_args()

# Calculate total experiments
# Base models: always 64x64
base_count = len(base_models) * len(shots) * len(samples_list)
# MatchingNet variants: 1 size each (conv64f: 64, resnet12/18: 84)
matchingnet_count = len(matchingnet_variants) * len(shots) * len(samples_list)
total_experiments = base_count + matchingnet_count
current_experiment = 0

print(f"=" * 80)
print(f"Configuration:")
print(f"  Base Models (64x64 only): {base_models}")
print(f"  MatchingNet variants: {matchingnet_variants}")
print(f"  MatchingNet image sizes: conv64f=64, resnet12/18=84")
print(f"  Shots: {shots}")
print(f"  Samples: {samples_list}")
print(f"  Total: {base_count} (base) + {matchingnet_count} (matchingnet) = {total_experiments}")
print(f"=" * 80)

def run_experiment(model, shot, samples, image_size, backbone=None):
    global current_experiment
    current_experiment += 1
    print(f"\n[{current_experiment}/{total_experiments}] Model={model}, Shot={shot}, Samples={samples if samples else 'All'}, ImageSize={image_size}")
    
    cmd = [sys.executable, 'main.py', 
           '--model', model, 
           '--shot_num', str(shot), 
           '--image_size', str(image_size),
           '--loss', 'contrastive',
           '--lambda_center', str(lambda_center),
           '--mode', 'train',
           '--project', args.project]
    
    if backbone is not None:
        cmd.extend(['--backbone', backbone])
    
    if samples is not None:
        cmd.extend(['--training_samples', str(samples)])
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

# Run base models (64x64 ONLY)
print("\n" + "=" * 40)
print("Running Base Models (64x64 only)")
print("=" * 40)
for model in base_models:
    for shot in shots:
        for samples in samples_list:
            run_experiment(model, shot, samples, image_size=64)

# Run MatchingNet variants (with appropriate image sizes)
print("\n" + "=" * 40)
print("Running MatchingNet Variants (conv64f: 64x64, resnet: 84x84)")
print("=" * 40)
for variant in matchingnet_variants:
    # Parse backbone from variant name
    if variant == 'matchingnet_resnet12':
        model, backbone = 'matchingnet', 'resnet12'
        valid_sizes = [84]  # resnet12 uses 84x84 only
    elif variant == 'matchingnet_resnet18':
        model, backbone = 'matchingnet', 'resnet18'
        valid_sizes = [84]  # resnet18 uses 84x84 only
    else:
        model, backbone = 'matchingnet', 'conv64f'
        valid_sizes = [64]  # conv64f uses 64x64 only
    
    for image_size in valid_sizes:
        for shot in shots:
            for samples in samples_list:
                run_experiment(model, shot, samples, image_size, backbone)

print("\nAll experiments completed.")

