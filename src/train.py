import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.config import *
from src.utils import EarlyStopping


# --- CheX-DS Specific Loss Function ---
class CheXDSLoss(nn.Module):
    """
    Combines Weighted BCE and Asymmetric Loss.
    Eq (3) from paper.
    """

    def __init__(self, class_ratios, gamma_pos=1, gamma_neg=4, m=0.05):
        super(CheXDSLoss, self).__init__()
        # rho: ratio of positive samples
        self.rho = torch.tensor(class_ratios).to(DEVICE)
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.m = m
        self.epsilon = 1e-7

    def forward(self, logits, targets):
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=1)  # Use softmax for multi-class

        # Convert scalar targets to One-Hot
        targets = torch.nn.functional.one_hot(targets, num_classes=logits.size(1)).float()

        # 1. Calculate Weights w_i
        # w_i = y_i * e^(1-rho) + (1-y_i) * e^rho
        w = targets * torch.exp(1 - self.rho) + (1 - targets) * torch.exp(self.rho)

        # 2. Asymmetric Terms
        # p_m = max(p - m, 0)
        p_m = torch.clamp(probs - self.m, min=0.0)

        # Positive term
        pos_term = torch.pow(1 - probs, self.gamma_pos) * targets * torch.log(probs + self.epsilon)

        # Negative term
        neg_term = torch.pow(p_m, self.gamma_neg) * (1 - targets) * torch.log(1 - p_m + self.epsilon)

        # 3. Final Summation
        loss = -torch.sum(w * (pos_term + neg_term)) / logits.size(0)
        return loss


def calculate_class_ratios(loader):
    """Calculates rho (ratio of positive samples)."""
    print("Calculating class ratios for CheX-DS Loss...")
    total = 0
    # Access underlying dataset safely
    if hasattr(loader.dataset, 'dataset'):
        targets = [s[1] for s in loader.dataset.dataset.samples]
        indices = loader.dataset.indices
        subset_targets = torch.tensor(targets)[indices]
    else:
        # Fallback for simple datasets
        subset_targets = torch.tensor([s[1] for s in loader.dataset])

    class_counts = torch.zeros(2)  # Binary: 0=Normal, 1=Pneumonia
    for i in range(2):
        class_counts[i] = (subset_targets == i).sum()

    total = len(subset_targets)
    ratios = class_counts / total
    print(f"Class Ratios (rho): {ratios.tolist()}")
    return ratios.tolist()


# --- Standard Loops ---
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    pbar = tqdm(loader, desc="Training", unit="batch")

    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += (preds == labels).sum().item()
        total_samples += labels.size(0)
        pbar.set_postfix({'loss': running_loss / total_samples})

    return running_loss / total_samples, running_corrects / total_samples


def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += (preds == labels).sum().item()
            total_samples += labels.size(0)
    return running_loss / total_samples, running_corrects / total_samples


def train_model(model, train_loader, val_loader, model_type='cnn'):
    model = model.to(DEVICE)

    # Select Loss Function
    if model_type == 'chexds':
        class_ratios = calculate_class_ratios(train_loader)
        # Using hyperparameters
        criterion = CheXDSLoss(class_ratios, gamma_pos=1, gamma_neg=4, m=0.05)
    else:
        criterion = nn.CrossEntropyLoss()

    # Hyperparameters
    if model_type in ['resnet', 'chexds']:
        lr = LR_RESNET
        epochs = EPOCHS_RESNET
    else:
        lr = LR_CNN
        epochs = EPOCHS_CNN

    # Optimizer (AdamW is used in paper )
    if model_type == 'chexds':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    early_stopping = EarlyStopping(patience=3, verbose=True, path=MODEL_SAVE_PATH)

    print(f"\n--- Starting Training: {model_type.upper()} on {DEVICE} ---")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    model = early_stopping.load_best_weights(model)

    # Fine-Tuning Logic (Applied to Transfer Learning models)
    if model_type in ['resnet', 'chexds']:
        print(f"\n--- Starting Fine-Tuning ({model_type.upper()}) ---")
        if hasattr(model, 'unfreeze_last_layers'):
            model.unfreeze_last_layers()

        model = model.to(DEVICE)
        # Reduced LR for fine-tuning
        if model_type == 'chexds':
            optimizer_ft = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-2)
        else:
            optimizer_ft = optim.Adam(model.parameters(), lr=0.0001)

        early_stopping = EarlyStopping(patience=3, verbose=True, path=MODEL_SAVE_PATH)

        for epoch in range(5):
            print(f"Fine-tune Epoch {epoch + 1}/5")
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_ft)
            val_loss, val_acc = validate(model, val_loader, criterion)
            print(f"FT Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

            early_stopping(val_loss, model)
            if early_stopping.early_stop: break

        model = early_stopping.load_best_weights(model)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    return model
