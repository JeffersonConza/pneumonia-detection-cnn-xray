import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import get_data_loaders
from src.model import SimpleCNN, TransferResNet, CheXDS  # <--- Added CheXDS
from src.train import train_model
from src.evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Pneumonia Detection Training")
    # Updated choices
    parser.add_argument('--model', type=str, choices=['cnn', 'resnet', 'chexds'], default='chexds',
                        help="Choose model: 'cnn', 'resnet', or 'chexds' (DenseNet+Swin Ensemble)")
    args = parser.parse_args()

    print(f"Initializing Data Loaders...")
    try:
        train_loader, val_loader, test_loader = get_data_loaders()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Building {args.model.upper()} Model...")
    if args.model == 'cnn':
        model = SimpleCNN()
    elif args.model == 'resnet':
        model = TransferResNet()
    elif args.model == 'chexds':
        # CheX-DS: DenseNet + Swin Ensemble
        model = CheXDS(num_classes=2, fine_tune=False)

    # Train
    model = train_model(model, train_loader, val_loader, model_type=args.model)

    # Evaluate
    evaluate_model(model, test_loader)


if __name__ == "__main__":
    main()