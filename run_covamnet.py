import subprocess
import sys
import argparse

def run_experiment(model, shot, loss, samples, lambda_center, project):
    cmd = [
        sys.executable, "main.py",
        "--model", model,
        "--shot_num", str(shot),
        "--loss", loss,
        "--lambda_center", str(lambda_center),
        "--mode", "train",
        "--project", project
    ]
    if samples is not None:
        cmd.extend(["--training_samples", str(samples)])
    
    print(f"\n{'='*50}")
    print(f"Running: {model} | {shot}-shot | {loss} | Lambda={lambda_center} | {samples if samples else 'all'} samples")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*50}\n")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Run CovaMNet experiments')
    parser.add_argument('--project', type=str, default='prpd', help='WandB project name')
    args = parser.parse_args()

    model = 'covamnet'
    shots = [1, 5]
    losses = ['contrastive', 'triplet']
    lambda_center = 0.1
    sample_sizes = [12, 60, None]
    
    for shot in shots:
        for loss in losses:
            for samples in sample_sizes:
                run_experiment(model, shot, loss, samples, lambda_center, args.project)

if __name__ == "__main__":
    main()
