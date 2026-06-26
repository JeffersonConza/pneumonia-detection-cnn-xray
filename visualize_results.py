import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np
import argparse
import os

# Import our modules
from src.data_loader import get_data_loaders
from src.model import SimpleCNN, TransferResNet, CheXDS
from src.config import DEVICE, MODEL_SAVE_PATH, BASE_DIR

def plot_confusion_matrix(y_true, y_pred, classes, model_name='chexds'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    # Heatmap visualization
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix ({model_name.upper()})')
    
    # Save the model-specific plot to results folder
    save_path = os.path.join(BASE_DIR, 'results', f'confusion_matrix_{model_name.lower()}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved confusion matrix plot to: {save_path}")
    
    # Also overwrite default Confusion Matrix for default path reference
    default_path = os.path.join(BASE_DIR, 'results', 'confusion_matrix.png')
    plt.savefig(default_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_roc_curve(y_true, y_probs, model_name='chexds'):
    # ROC Curve helps us see the trade-off between Sensitivity and Specificity
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (False Alarms)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title(f'Receiver Operating Characteristic (ROC) - {model_name.upper()}')
    plt.legend(loc="lower right")
    
    # Save the model-specific plot to results folder
    save_path = os.path.join(BASE_DIR, 'results', f'roc_curve_{model_name.lower()}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved ROC curve plot to: {save_path}")
    
    # Also overwrite default ROC Curve
    default_path = os.path.join(BASE_DIR, 'results', 'roc_curve.png')
    plt.savefig(default_path, bbox_inches='tight', dpi=300)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Evaluate and Visualize Model Performance")
    parser.add_argument('--model', type=str, choices=['cnn', 'resnet', 'chexds'], default='chexds',
                        help="Choose model type to visualize: 'cnn', 'resnet', or 'chexds'")
    args = parser.parse_args()

    print(f"Loading Test Data for Evaluation...")
    # We only need the test_loader
    _, _, test_loader = get_data_loaders()
    
    # 1. Initialize Model
    if args.model == 'cnn':
        model = SimpleCNN(num_classes=2)
    elif args.model == 'resnet':
        model = TransferResNet(num_classes=2)
    else:
        model = CheXDS(num_classes=2)
        
    # Check if a model-specific file exists, otherwise fallback to default MODEL_SAVE_PATH
    specific_path = os.path.join(BASE_DIR, 'models', f'pneumonia_model_{args.model}.pth')
    if os.path.exists(specific_path):
        load_path = specific_path
    elif os.path.exists(MODEL_SAVE_PATH):
        print(f"Warning: Model-specific weights not found at {specific_path}. Falling back to default path {MODEL_SAVE_PATH}.")
        load_path = MODEL_SAVE_PATH
    else:
        print(f"Error: No weight file found for {args.model.upper()}. Checked paths:\n - {specific_path}\n - {MODEL_SAVE_PATH}")
        return

    print(f"Loading {args.model.upper()} weights from {load_path}...")
    try:
        state_dict = torch.load(load_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error: Could not load state_dict for {args.model.upper()} from {load_path}: {e}")
        print("This typically happens if the file contains weights for a different architecture.")
        print("Please train the model first using: python main.py --model <model_type>")
        return
        
    model.to(DEVICE)
    model.eval()
    
    y_true = []
    y_pred = []
    y_probs = []
    
    print("Running Inference on Test Set (this may take a moment)...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get predicted class (0 or 1)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            # Probability of 'PNEUMONIA' (Class 1) is needed for ROC
            y_probs.extend(probs[:, 1].cpu().numpy())

    # 2. Print Stats
    classes = ['NORMAL', 'PNEUMONIA']
    print("\n" + "="*60)
    print(f"FINAL CLASSIFICATION REPORT ({args.model.upper()})")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=classes))

    # 3. Plotting
    plot_confusion_matrix(y_true, y_pred, classes, model_name=args.model)
    plot_roc_curve(y_true, y_probs, model_name=args.model)

if __name__ == "__main__":
    main()