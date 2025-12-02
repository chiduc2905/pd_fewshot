import subprocess
import sys
import argparse

def run_experiment(model, shot, loss, samples, device):
    cmd = [
        sys.executable, "main.py",
        "--model", model,
        "--shot_num", str(shot),
        "--loss", loss,
        "--mode", "train",
        "--device", device
    ]
    if samples is not None:
        cmd.extend(["--training_samples", str(samples)])
    
    print(f"\n{'='*50}")
    print(f"Running: {model} | {shot}-shot | {loss} | {samples if samples else 'all'} samples | Device: {device}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*50}\n")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Run CovaMNet experiments')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (e.g., cuda:0, cuda:1)')
    args = parser.parse_args()

    model = 'covamnet'
    shots = [1, 5]
    losses = ['contrastive', 'supcon', 'triplet']
    sample_sizes = [30, 60, 90, None]
    
    for shot in shots:
        for loss in losses:
            for samples in sample_sizes:
                run_experiment(model, shot, loss, samples, args.device)

if __name__ == "__main__":
    main()
