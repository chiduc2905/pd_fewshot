import os
import subprocess
import argparse
import sys

def get_args():
    parser = argparse.ArgumentParser(description='Run all experiments with custom arguments')
    
    # Define arguments that we want to allow overriding
    # We use parse_known_args so that any argument not defined here 
    # but valid for main.py can be passed through.
    
    # Models and Shots to iterate over (specific to this runner)
    parser.add_argument('--models', nargs='+', default=['cosine', 'protonet', 'covamnet'], 
                        help='List of models to run (default: cosine protonet covamnet)')
    parser.add_argument('--shots', nargs='+', type=int, default=[1, 5], 
                        help='List of shot numbers to run (default: 1 5)')
    
    return parser.parse_known_args()

def run_experiment(model, shot, unknown_args):
    # Base command
    cmd = [
        sys.executable, "main.py",
        "--model", model,
        "--shot_num", str(shot),
        "--mode", "train" # Default to train, can be overridden if passed in unknown_args but might duplicate
    ]
    
    # Check if user explicitly provided a mode in unknown_args, if so, it will override the one above 
    # because argparse typically takes the last occurrence if provided multiple times, 
    # OR we can just append unknown_args.
    
    # Append all other arguments passed to this script
    cmd.extend(unknown_args)
    
    print(f"\n{'='*50}")
    print(f"Running Experiment: {model} {shot}-shot")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*50}\n")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {model} {shot}-shot: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
        sys.exit(1)

def main():
    args, unknown_args = get_args()
    
    print(f"Models to run: {args.models}")
    print(f"Shots to run: {args.shots}")
    print(f"Additional args passing to main.py: {unknown_args}")
    
    for model in args.models:
        for shot in args.shots:
            run_experiment(model, shot, unknown_args)

if __name__ == "__main__":
    main()
