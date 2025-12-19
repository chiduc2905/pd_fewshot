"""Run experiments for new few-shot models: Siamese, DN4, FEAT, DeepEMD."""
import subprocess
import sys
import argparse


def run_experiment(model, shot, loss, samples, project):
    """Run a single experiment."""
    cmd = [
        sys.executable, "main.py",
        "--model", model,
        "--shot_num", str(shot),
        "--loss", loss,
        "--lambda_center", "0",
        "--mode", "train",
        "--project", project
    ]
    if samples is not None:
        cmd.extend(["--training_samples", str(samples)])
    
    print(f"\n{'='*60}")
    print(f"Running: {model} | {shot}-shot | {loss} | {samples if samples else 'all'} samples")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Run new few-shot model experiments')
    parser.add_argument('--project', type=str, default='prpd', help='WandB project name')
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['siamese', 'dn4', 'feat', 'deepemd'],
                        choices=['siamese', 'dn4', 'feat', 'deepemd'],
                        help='Models to run')
    parser.add_argument('--shots', type=int, nargs='+', default=[1, 5],
                        help='Shot numbers to test')
    parser.add_argument('--samples', type=int, nargs='+', default=[18, 60, None],
                        help='Training sample sizes (None for all)')
    args = parser.parse_args()
    
    # Fixed loss for now
    loss = 'contrastive'
    
    for model in args.models:
        for shot in args.shots:
            for samples in args.samples:
                run_experiment(model, shot, loss, samples, args.project)


if __name__ == "__main__":
    main()
