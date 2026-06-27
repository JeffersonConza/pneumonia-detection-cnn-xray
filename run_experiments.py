import os
import json
import subprocess
import sys
import time
import argparse

# Add root folder to sys.path to enable src imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.config import BASE_DIR, DEVICE

# The three models defined in our main.py
models_to_run = [
    'cnn',      # Simple CNN (Baseline)
    'resnet',   # ResNet50 (Transfer Learning)
    'chexds'    # CheX-DS (DenseNet + Swin Ensemble)
]


def get_final_metrics(model_name):
    history_path = os.path.join(BASE_DIR, 'results', f'history_{model_name}.json')
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
            if history:
                # Return metrics from the final epoch
                last_epoch = history[-1]
                return {
                    'train_loss': last_epoch.get('train_loss'),
                    'train_acc': last_epoch.get('train_acc'),
                    'val_loss': last_epoch.get('val_loss'),
                    'val_acc': last_epoch.get('val_acc')
                }
        except Exception as e:
            print(f"Error reading history for {model_name}: {e}")
    return None


def run_experiment(model_name):
    print(f"\n{'=' * 60}")
    print(f"STARTING EXPERIMENT: {model_name.upper()}")
    print(f"{'=' * 60}\n")

    start_time = time.time()
    success = False

    try:
        # We use subprocess to launch a fresh Python instance for each model.
        # This ensures GPU/RAM is completely freed after each run.
        subprocess.run(
            [sys.executable, 'main.py', '--model', model_name],
            check=True
        )
        success = True
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: {model_name.upper()} crashed with exit code {e.returncode}")
    except KeyboardInterrupt:
        print(f"\nINTERRUPTED: User stopped the training.")

    duration = (time.time() - start_time) / 60
    status_str = "FINISHED" if success else "FAILED"
    print(f"\n{status_str}: {model_name.upper()} in {duration:.2f} minutes.")
    
    metrics = get_final_metrics(model_name) if success else None
    
    return {
        'model': model_name,
        'status': 'SUCCESS' if success else 'FAILED',
        'duration_minutes': round(duration, 2),
        'final_metrics': metrics
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full Benchmark Suite")
    parser.add_argument('--force', action='store_true',
                        help="Force running experiments on CPU despite safety check warnings.")
    args = parser.parse_args()

    # CPU Safety Check
    if DEVICE.type == 'cpu':
        print("\n" + "!"*60)
        print("⚠️  WARNING: NO GPU RUNTIME DETECTED (RUNNING ON CPU)")
        print("Training all three models sequentially on CPU will take ~12 HOURS!")
        print("This is highly likely to cause system lockup or overheating on local PCs.")
        print("Please train models on Google Colab GPU runtime instead.")
        print("!"*60 + "\n")
        
        if not args.force:
            print("Execution aborted to protect system resources.")
            print("To bypass this safety check and run anyway, use the '--force' flag:")
            print("  python run_experiments.py --force\n")
            sys.exit(0)
        else:
            print("⚠️ '--force' flag detected. Proceeding with CPU training...\n")

    print("--- Starting Full Benchmark Protocol ---")

    experiments_log = []
    all_success = True

    for model in models_to_run:
        result = run_experiment(model)
        experiments_log.append(result)
        if result['status'] == 'FAILED':
            all_success = False
            print("Stopping sequence due to error.")
            break

    # Save summary report to results/ directory
    summary_path = os.path.join(BASE_DIR, 'results', 'experiments_summary.json')
    try:
        with open(summary_path, 'w') as f:
            json.dump(experiments_log, f, indent=4)
        print(f"\nExperiment summaries saved to: {summary_path}")
    except Exception as e:
        print(f"Error saving experiment summaries: {e}")

    if all_success:
        print("\nAll models built successfully.")
    else:
        print("\nSome models failed.")
