"""Run all benchmark experiments for 1-shot, 5-shot with various training sample sizes."""
import subprocess
import sys
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description='Run all benchmark experiments')
    parser.add_argument('--project', type=str, default='prpd', help='WandB project name')
    parser.add_argument('--dataset_path', type=str, 
                        default='/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/scalogram_official',
                        help='Path to dataset')
    parser.add_argument('--dataset_name', type=str, default='minh', help='Dataset name for logging')
    return parser.parse_args()


# Configuration
SHOTS = [1, 5]

# Training samples: [min, small, medium, all] - matches mamba_glscnet
SAMPLES_LIST = [30, 60, 150, None]

# Query samples (same for train/val/test) - matches mamba_glscnet
QUERY_NUM = 5

# Base models (64x64)
BASE_MODELS = ['covamnet', 'protonet', 'cosine', 'baseline', 'relationnet', 
               'siamese', 'dn4', 'feat', 'deepemd']

# MatchingNet variants
MATCHINGNET_VARIANTS = ['matchingnet', 'matchingnet_resnet12', 'matchingnet_resnet18']


def run_experiment(model, shot, samples, image_size, dataset_path, dataset_name, project, backbone=None):
    """Run a single benchmark experiment."""
    print(f"\n{'='*60}")
    print(f"Model={model}, Shot={shot}, Samples={samples if samples else 'All'}, ImageSize={image_size}")
    print('='*60)
    
    cmd = [
        sys.executable, 'main.py',
        '--model', model,
        '--shot_num', str(shot),
        '--way_num', '3',
        '--query_num', str(QUERY_NUM),
        '--image_size', str(image_size),
        '--mode', 'train',
        '--project', project,
        '--dataset_path', dataset_path,
        '--dataset_name', dataset_name,
        '--num_epochs', '100',
        '--lr', '1e-3',
        '--eta_min', '1e-5',
        '--weight_decay', '1e-4',
        '--episode_num_train', '100',
        '--episode_num_val', '150',
        '--episode_num_test', '150',
    ]
    
    if backbone is not None:
        cmd.extend(['--backbone', backbone])
    
    if samples is not None:
        cmd.extend(['--training_samples', str(samples)])
    
    # Set GPU device to 1
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '1'
    
    try:
        subprocess.run(cmd, check=True, env=env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False


def main():
    args = get_args()
    
    # Count total experiments
    # Base models: 9 models * 2 shots * 4 samples = 72
    # MatchingNet variants: 3 * 2 * 4 = 24
    total_experiments = len(BASE_MODELS) * len(SHOTS) * len(SAMPLES_LIST) + \
                        len(MATCHINGNET_VARIANTS) * len(SHOTS) * len(SAMPLES_LIST)
    current = 0
    
    print("="*60)
    print("Benchmark Models - Full Experiment Suite")
    print("="*60)
    print(f"Base Models: {BASE_MODELS}")
    print(f"MatchingNet Variants: {MATCHINGNET_VARIANTS}")
    print(f"Shots: {SHOTS}")
    print(f"Training samples: {SAMPLES_LIST}")
    print(f"Dataset: {args.dataset_path} ({args.dataset_name})")
    print(f"GPU Device: 1")
    print(f"Total experiments: {total_experiments}")
    print("="*60)
    
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    success_count = 0
    failed_experiments = []
    
    # Run base models (64x64 only)
    print("\n" + "="*40)
    print("Running Base Models (64x64)")
    print("="*40)
    for model in BASE_MODELS:
        for shot in SHOTS:
            for samples in SAMPLES_LIST:
                current += 1
                print(f"\n[{current}/{total_experiments}]", end=" ")
                
                success = run_experiment(
                    model=model,
                    shot=shot,
                    samples=samples,
                    image_size=64,
                    dataset_path=args.dataset_path,
                    dataset_name=args.dataset_name,
                    project=args.project
                )
                
                if success:
                    success_count += 1
                else:
                    failed_experiments.append(f"{model}_{shot}shot_{samples}samples")
    
    # Run MatchingNet variants
    print("\n" + "="*40)
    print("Running MatchingNet Variants")
    print("="*40)
    for variant in MATCHINGNET_VARIANTS:
        if variant == 'matchingnet_resnet12':
            model, backbone, img_size = 'matchingnet', 'resnet12', 84
        elif variant == 'matchingnet_resnet18':
            model, backbone, img_size = 'matchingnet', 'resnet18', 84
        else:
            model, backbone, img_size = 'matchingnet', 'conv64f', 64
        
        for shot in SHOTS:
            for samples in SAMPLES_LIST:
                current += 1
                print(f"\n[{current}/{total_experiments}]", end=" ")
                
                success = run_experiment(
                    model=model,
                    shot=shot,
                    samples=samples,
                    image_size=img_size,
                    dataset_path=args.dataset_path,
                    dataset_name=args.dataset_name,
                    project=args.project,
                    backbone=backbone
                )
                
                if success:
                    success_count += 1
                else:
                    failed_experiments.append(f"{variant}_{shot}shot_{samples}samples")
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Total: {total_experiments}")
    print(f"Success: {success_count}")
    print(f"Failed: {len(failed_experiments)}")
    
    if failed_experiments:
        print("\nFailed experiments:")
        for exp in failed_experiments:
            print(f"  - {exp}")
    
    print("\n" + "="*60)
    print("Generating comparison charts...")
    print("="*60)
    
    # Generate comparison after all experiments
    generate_comparison_charts(args.dataset_name)
    
    print("\nAll experiments completed!")


def generate_comparison_charts(dataset_name):
    """Generate comparison bar charts from results."""
    import re
    try:
        from function.function import plot_model_comparison_bar
    except ImportError:
        print("Warning: Could not import plot function, skipping charts")
        return
    
    results_dir = 'results/'
    
    # Model display names
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
    
    all_models = ['cosine', 'baseline', 'protonet', 'covamnet', 'matchingnet', 
                  'relationnet', 'siamese', 'dn4', 'feat', 'deepemd']
    
    for samples in SAMPLES_LIST:
        samples_str = f"{samples}samples"
        
        model_results = {}
        
        for model in all_models:
            display_name = model_display_names.get(model, model)
            model_results[display_name] = {}
            
            for shot in SHOTS:
                result_file = os.path.join(
                    results_dir,
                    f"results_{dataset_name}_{model}_{samples_str}_{shot}shot.txt"
                )
                
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        content = f.read()
                        # Parse accuracy
                        match = re.search(r'Accuracy\s*:\s*([\d.]+)\s*±', content)
                        if match:
                            acc = float(match.group(1))
                            model_results[display_name][f'{shot}shot'] = acc
        
        # Remove incomplete results
        complete_results = {}
        for model, shots_dict in model_results.items():
            if all(f'{shot}shot' in shots_dict for shot in SHOTS):
                complete_results[model] = shots_dict
        
        if complete_results:
            save_path = os.path.join(results_dir, f"model_comparison_{dataset_name}_{samples_str}.png")
            try:
                plot_model_comparison_bar(complete_results, samples, save_path)
                print(f"  Chart saved: {save_path}")
            except Exception as e:
                print(f"  Error generating chart: {e}")
        else:
            print(f"  No complete results for {samples_str}")


if __name__ == '__main__':
    main()
