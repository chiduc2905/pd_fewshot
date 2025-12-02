import subprocess
import sys

# Configuration
models = ['covamnet', 'protonet', 'cosine']
shots = [1, 5]
samples_list = [30, 60, 90, None]  # None means all samples
losses = ['contrastive', 'supcon', 'triplet']

total_experiments = len(models) * len(shots) * len(samples_list) * len(losses)
current_experiment = 0

print(f"Starting {total_experiments} experiments...")

for model in models:
    for shot in shots:
        for loss in losses:
            for samples in samples_list:
                current_experiment += 1
                print(f"\n[{current_experiment}/{total_experiments}] Running: Model={model}, Shot={shot}, Loss={loss}, Samples={samples if samples else 'All'}")
                
                cmd = [sys.executable, 'main.py', 
                       '--model', model, 
                       '--shot_num', str(shot), 
                       '--loss', loss, 
                       '--mode', 'train']
                
                if samples is not None:
                    cmd.extend(['--training_samples', str(samples)])
                
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error running experiment: {e}")
                    # Decide whether to continue or stop. Continuing is usually safer for long runs.
                    continue

print("\nAll experiments completed.")
