import subprocess
import sys
import time

# The three models defined in our main.py
models_to_run = [
    'cnn',      # Simple CNN (Baseline)
    'resnet',   # ResNet50 (Transfer Learning)
    'chexds'    # CheX-DS (DenseNet + Swin Ensemble)
]


def run_experiment(model_name):
    print(f"\n{'=' * 60}")
    print(f"STARTING EXPERIMENT: {model_name.upper()}")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    try:
        # We use subprocess to launch a fresh Python instance for each model.
        # This ensures GPU/RAM is completely freed after each run.
        subprocess.run(
            [sys.executable, 'main.py', '--model', model_name],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: {model_name.upper()} crashed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\nINTERRUPTED: User stopped the training.")
        return False

    duration = (time.time() - start_time) / 60
    print(f"\nFINISHED: {model_name.upper()} in {duration:.2f} minutes.")
    return True


if __name__ == "__main__":
    print("--- Starting Full Benchmark Protocol ---")

    all_success = True

    for model in models_to_run:
        success = run_experiment(model)
        if not success:
            all_success = False
            print("Stopping sequence due to error.")
            break

    if all_success:
        print("\nModels built successfully.")
    else:
        print("\nSome models failed.")
