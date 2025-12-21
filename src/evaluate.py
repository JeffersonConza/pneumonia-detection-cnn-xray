import torch
import torch.nn as nn
from src.config import *


def evaluate_model(model, test_loader):
    """
    Evaluates the trained model on the test dataset.
    """
    model = model.to(DEVICE)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    running_corrects = 0
    total = 0

    print("\n--- Evaluating on Test Set ---")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += (preds == labels).sum().item()
            total += labels.size(0)

    final_loss = running_loss / total
    final_acc = running_corrects / total

    print(f"Test Loss: {final_loss:.4f}")
    print(f"Test Accuracy: {final_acc * 100:.2f}%")
    return final_loss, final_acc