import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Import our modules
from src.data_loader import get_data_loaders
from src.model import CheXDS
from src.config import DEVICE, MODEL_SAVE_PATH

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    # Heatmap visualization
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (CheX-DS)')
    plt.show()

def plot_roc_curve(y_true, y_probs):
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
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

def main():
    print(f"Loading Test Data for Evaluation...")
    # We only need the test_loader
    _, _, test_loader = get_data_loaders()
    
    # 1. Load the Trained Model
    print(f"Loading CheX-DS weights from {MODEL_SAVE_PATH}...")
    model = CheXDS(num_classes=2)
    
    # Load weights (map_location ensures it loads on CPU if GPU is missing)
    state_dict = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    
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
    print("FINAL CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=classes))

    # 3. Plotting
    plot_confusion_matrix(y_true, y_pred, classes)
    plot_roc_curve(y_true, y_probs)

if __name__ == "__main__":
    main()