import subprocess
import sys

# Matching Networks experiments
def run_matchingnet_experiments(shots=[1, 5], samples_list=[18, 60, None]):
    """Run MatchingNet experiments with different configurations."""
    
    model = 'matchingnet'
    lambda_center = 0
    
    total = len(shots) * len(samples_list)
    current = 0
    
    print(f"=" * 80)
    print(f"Running MatchingNet Experiments")
    print(f"  Shots: {shots}")
    print(f"  Samples: {samples_list}")
    print(f"  Loss: CrossEntropy (paper-standard)")
    print(f"  Total: {total} experiments")
    print(f"=" * 80)
    
    for shot in shots:
        for samples in samples_list:
            current += 1
            print(f"\n[{current}/{total}] MatchingNet: Shot={shot}, Samples={samples if samples else 'All'}")
            
            cmd = [sys.executable, 'main.py',
                   '--model', model,
                   '--shot_num', str(shot),
                   '--loss', 'contrastive',
                   '--lambda_center', str(lambda_center),
                   '--mode', 'train']
            
            if samples is not None:
                cmd.extend(['--training_samples', str(samples)])
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error: {e}")
                continue
    
    print("\nMatchingNet experiments completed!")

if __name__ == '__main__':
    run_matchingnet_experiments()
