import subprocess
import sys

def run_experiment(model, shot, loss, samples):
    cmd = [
        sys.executable, "main.py",
        "--model", model,
        "--shot_num", str(shot),
        "--loss", loss,
        "--mode", "train"
    ]
    if samples is not None:
        cmd.extend(["--training_samples", str(samples)])
    
    print(f"\n{'='*50}")
    print(f"Running: {model} | {shot}-shot | {loss} | {samples if samples else 'all'} samples")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*50}\n")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

def main():
    model = 'protonet'
    shots = [1, 5]
    losses = ['contrastive', 'supcon', 'triplet']
    sample_sizes = [30, 60, 90, None]
    
    for shot in shots:
        for loss in losses:
            for samples in sample_sizes:
                run_experiment(model, shot, loss, samples)

if __name__ == "__main__":
    main()
