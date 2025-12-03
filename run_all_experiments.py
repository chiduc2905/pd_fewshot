import subprocess
import sys
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Run all experiments')
    parser.add_argument('--project', type=str, default='prpd', help='WandB project name')
    return parser.parse_args()

# Configuration
models = ['covamnet', 'protonet', 'cosine']
shots = [1, 5]
samples_list = [12, 60, None]  # None means all samples
losses = ['contrastive', 'triplet']
lambda_center = 0.1

total_experiments = len(models) * len(shots) * len(samples_list) * len(losses)
current_experiment = 0

args = get_args()
print(f"Starting {total_experiments} experiments on project '{args.project}'...")

for model in models:
    for shot in shots:
        for loss in losses:
            for samples in samples_list:
                current_experiment += 1
                print(f"\n[{current_experiment}/{total_experiments}] Running: Model={model}, Shot={shot}, Loss={loss}, Lambda={lambda_center}, Samples={samples if samples else 'All'}")
                
                cmd = [sys.executable, 'main.py', 
                       '--model', model, 
                       '--shot_num', str(shot), 
                       '--loss', loss, 
                       '--lambda_center', str(lambda_center),
                       '--mode', 'train',
                       '--project', args.project]
                
                if samples is not None:
                    cmd.extend(['--training_samples', str(samples)])
                
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error running experiment: {e}")
                    # Decide whether to continue or stop. Continuing is usually safer for long runs.
                    continue

print("\nAll experiments completed.")
