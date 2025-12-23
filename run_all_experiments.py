import subprocess
import sys
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Run all experiments')
    parser.add_argument('--project', type=str, default='prpd', help='WandB project name')
    return parser.parse_args()

# Configuration
# Models that ONLY use 64x64 (paper standard)
base_models = ['covamnet', 'protonet', 'cosine', 'baseline', 'relationnet', 
               'siamese', 'dn4', 'feat', 'deepemd']  # Added baseline (Baseline++)

# MatchingNet variants (support multiple image sizes)
matchingnet_variants = ['matchingnet', 'matchingnet_resnet12', 'matchingnet_resnet18']

shots = [1, 5]
lambda_center = 0

args = get_args()

# =============================================================================
# Dataset configuration
# =============================================================================
# Server paths for two datasets
DATASETS = {
    'original': {
        'path': '/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/scalogram_original',
        'samples_list': [18, 30],  # Original: only 18 and 30 samples
    },
    'augmented': {
        'path': '/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/scalogram_augmented',
        'samples_list': [18, 60, None],  # Augmented: 18, 60, and all samples
    }
}

# Calculate total experiments
total_experiments = 0
for dataset_name, config in DATASETS.items():
    samples_list = config['samples_list']
    # Base models: always 64x64
    base_count = len(base_models) * len(shots) * len(samples_list)
    # MatchingNet variants: 1 size each (conv64f: 64, resnet12/18: 84)
    matchingnet_count = len(matchingnet_variants) * len(shots) * len(samples_list)
    total_experiments += base_count + matchingnet_count

current_experiment = 0

print(f"=" * 80)
print(f"Configuration:")
print(f"  Base Models (64x64 only): {base_models}")
print(f"  MatchingNet variants: {matchingnet_variants}")
print(f"  MatchingNet image sizes: conv64f=64, resnet12/18=84")
print(f"  Shots: {shots}")
print(f"  Datasets:")
for name, config in DATASETS.items():
    print(f"    - {name}: {config['path']}")
    print(f"      Samples: {config['samples_list']}")
print(f"  Total experiments: {total_experiments}")
print(f"=" * 80)

def run_experiment(model, shot, samples, image_size, dataset_path, dataset_name, backbone=None):
    global current_experiment
    current_experiment += 1
    print(f"\n[{current_experiment}/{total_experiments}] Dataset={dataset_name}, Model={model}, Shot={shot}, Samples={samples if samples else 'All'}, ImageSize={image_size}")
    
    cmd = [sys.executable, 'main.py', 
           '--model', model, 
           '--shot_num', str(shot), 
           '--image_size', str(image_size),
           '--loss', 'contrastive',
           '--lambda_center', str(lambda_center),
           '--mode', 'train',
           '--project', args.project,
           '--dataset_path', dataset_path,
           '--dataset_name', dataset_name]
    
    if backbone is not None:
        cmd.extend(['--backbone', backbone])
    
    if samples is not None:
        cmd.extend(['--training_samples', str(samples)])
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

# =============================================================================
# Run experiments for each dataset
# =============================================================================
for dataset_name, config in DATASETS.items():
    dataset_path = config['path']
    samples_list = config['samples_list']
    
    print("\n" + "=" * 80)
    print(f"DATASET: {dataset_name.upper()}")
    print(f"Path: {dataset_path}")
    print(f"Samples: {samples_list}")
    print("=" * 80)
    
    # Run base models (64x64 ONLY)
    print("\n" + "=" * 40)
    print(f"Running Base Models (64x64 only) - {dataset_name}")
    print("=" * 40)
    for model in base_models:
        for shot in shots:
            for samples in samples_list:
                run_experiment(model, shot, samples, image_size=64, 
                              dataset_path=dataset_path, dataset_name=dataset_name)

    # Run MatchingNet variants (with appropriate image sizes)
    print("\n" + "=" * 40)
    print(f"Running MatchingNet Variants - {dataset_name}")
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
                    run_experiment(model, shot, samples, image_size, 
                                  dataset_path=dataset_path, dataset_name=dataset_name, 
                                  backbone=backbone)

print("\nAll experiments completed.")

# Generate model comparison bar charts after all experiments are done
print("\n" + "=" * 40)
print("Generating Model Comparison Bar Charts")
print("=" * 40)

import os
import re
from function.function import plot_model_comparison_bar

results_dir = 'results/'

# Model name mapping for display
model_display_names = {
    'cosine': 'Cosine Classifier',
    'baseline': 'Baseline++',
    'protonet': 'ProtoNet',
    'covamnet': 'CovaMNet',
    'matchingnet': 'MatchingNet',
    'relationnet': 'RelationNet',
    'siamese': 'SiameseNet',
    'dn4': 'DN4',
    'feat': 'FEAT',
    'deepemd': 'DeepEMD'
}

all_models = ['cosine', 'baseline', 'protonet', 'covamnet', 'matchingnet', 'relationnet',
              'siamese', 'dn4', 'feat', 'deepemd']

# Generate bar charts for each dataset
for dataset_name, config in DATASETS.items():
    samples_list = config['samples_list']
    
    print(f"\n--- Bar Charts for {dataset_name.upper()} ---")
    
    for samples in samples_list:
        samples_str = f"{samples}samples" if samples else "allsamples"
        
        # Collect results for this sample count
        model_results = {}
        
        for model in all_models:
            display_name = model_display_names.get(model, model)
            model_results[display_name] = {'1shot': None, '5shot': None}
            
            for shot in [1, 5]:
                # New naming format: results_dataset_model_samples_shot.txt
                result_file = os.path.join(results_dir, 
                    f"results_{dataset_name}_{model}_{samples_str}_{shot}shot.txt")
                
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        content = f.read()
                        # Updated regex to match new format: "Accuracy : 0.9700 ± 0.0234"
                        match = re.search(r'Accuracy\s*:\s*([\d.]+)\s*±', content)
                        if match:
                            acc = float(match.group(1))
                            model_results[display_name][f'{shot}shot'] = acc
        
        # Remove models with missing data
        model_results = {k: v for k, v in model_results.items() 
                         if v['1shot'] is not None and v['5shot'] is not None}
        
        if len(model_results) > 0:
            training_samples = samples if samples else 'All'
            # Include dataset name in the chart filename
            save_path = os.path.join(results_dir, f"model_comparison_{dataset_name}_{samples_str}.png")
            plot_model_comparison_bar(model_results, training_samples, save_path)
            print(f"  Bar chart saved: {save_path}")
        else:
            print(f"  No complete results found for {dataset_name}/{samples_str}")

print("\n" + "=" * 40)
print("All bar charts generated!")
print("=" * 40)

