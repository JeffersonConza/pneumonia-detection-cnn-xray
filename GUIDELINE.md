# 🫁 Guideline for Remaking & Enhancing Pneumonia Detection AI on Google Colab

This document provides a step-by-step, phase-by-phase engineering guideline to reconstruct, optimize, and deploy the modular Deep Learning pipeline for pneumonia detection from Chest X-Rays. 

Under this setup, **the entire project lifecycle (data retrieval, code generation, GPU training, metric plotting, and Gradio web interface hosting) is executed exclusively inside Google Colab**.

---

## 📂 Targeted Workspace Directory Structure in Colab

```text
/content/                               # Colab Local VM NVMe Root
├── data/                               # Local dataset directory
│   └── chest_xray/                     # Extracted train/test subsets
├── results/                            # Generated performance plots (Confusion Matrix, ROC)
├── src/                                # Core codebase package
│   ├── __init__.py
│   ├── config.py                       # Paths, GPU switches, hyperparameters
│   ├── data_loader.py                  # Custom loaders, augmentation, splits
│   ├── evaluate.py                     # Evaluation loops
│   ├── model.py                        # CNN, ResNet50, CheX-DS models
│   ├── train.py                        # Training loops, custom loss, AMP
│   └── utils.py                        # EarlyStopping callback
├── requirements.txt                    # Pip dependencies
├── main.py                             # CLI entrance script
├── visualize_results.py                # Plots confusion matrix & ROC curve
└── gradio_app.py                       # GUI Web Interface (Colab-native deployment)
```

---

## 🗺️ Project Remake & Optimization Phases

### Phase 1: Colab Workspace & Environment Initialization
**Objective**: Mount persistent Google Drive storage, verify GPU accelerators, download dependencies, and fetch the dataset to the local VM.

#### Cell 1: Mount Drive and Set Workspace
```python
from google.colab import drive
import os
drive.mount('/content/drive')

# Establish path to the project directory inside Google Drive
PROJECT_PATH = '/content/drive/MyDrive/pneumonia-detection-cnn-xray'
os.makedirs(PROJECT_PATH, exist_ok=True)
os.chdir(PROJECT_PATH)
print("Current Working Directory:", os.getcwd())
```

#### Cell 2: Check GPU Hardware
*Verify that the runtime is set to GPU (Runtime -> Change runtime type -> T4 GPU).*
```python
import torch
print("GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
```

#### Cell 3: Generate Dependencies list
```python
%%writefile requirements.txt
torch==2.9.1
torchvision==0.24.1
matplotlib==3.10.8
seaborn==0.13.2
scikit-learn==1.8.0
tqdm==4.67.1
gradio==4.44.0
```

#### Cell 4: Install Packages
```bash
!pip install -r requirements.txt
```

#### Cell 5: Fast Dataset Retrieval (SSD Caching Optimization)
> [!IMPORTANT]
> **Performance Tip**: Training directly on network-mounted Google Drive causes slow I/O training cycles. We cache the dataset on Colab's local NVMe SSD (`/content`) for faster training loops.
```python
!mkdir -p /content/data
print("Downloading 1.2GB dataset archive...")
!curl -L -o /content/xray_dataset.tar.gz "https://dsserver-prod-resources-1.s3.amazonaws.com/cnn/xray_dataset.tar.gz"
print("Decompressing tarball to local SSD...")
!tar -xzf /content/xray_dataset.tar.gz -C /content/
print("Dataset ready locally at: /content/chest_xray")
```

---

### Phase 2: Setup Core Configuration & Support Modules
**Objective**: Build configuration constants and helper callbacks.

#### Cell 6: Create config.py
```python
%%writefile src/config.py
import os
import torch

# GPU accelerator check
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths (Dynamic location resolution)
IN_COLAB = os.path.exists('/content')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = '/content/chest_xray' if IN_COLAB else os.path.join(BASE_DIR, 'data', 'chest_xray')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Model weights are written to Drive to persist across Colab runs
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'pneumonia_model.pth')

# Hyperparameters
IMG_SIZE = 256
BATCH_SIZE = 64 if torch.cuda.is_available() else 32
SEED = 417
VAL_SPLIT = 0.20

EPOCHS_CNN = 15
EPOCHS_RESNET = 10
LR_CNN = 0.001
LR_RESNET = 0.01

# Enable AMP if GPU is available
USE_AMP = torch.cuda.is_available()
```

#### Cell 7: Create utils.py (Early Stopping Helper)
```python
%%writefile src/utils.py
import torch
import copy
from src.config import MODEL_SAVE_PATH

class EarlyStopping:
    def __init__(self, patience=3, verbose=False, path=MODEL_SAVE_PATH):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_wts = None
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.counter = 0

    def load_best_weights(self, model):
        if self.best_model_wts:
            model.load_state_dict(self.best_model_wts)
            print("Restored best model weights.")
        return model
```

---

### Phase 3: Deep Learning Pipeline Construction
**Objective**: Build preprocessing loaders, model architectures, and training utilities.

#### Cell 8: Create data_loader.py (Data Loaders & Augmentation)
```python
%%writefile src/data_loader.py
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from src.config import *

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(degrees=20),
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
        transforms.ToTensor(),
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    return train_transform, val_test_transform

def get_data_loaders():
    train_tf, val_tf = get_transforms()
    full_train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_tf)

    total_size = len(full_train_dataset)
    val_size = int(total_size * VAL_SPLIT)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=generator)
    val_dataset.dataset.transform = val_tf

    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=val_tf)

    num_workers = 2 if torch.cuda.is_available() else 0
    pin_memory = True if torch.cuda.is_available() else False

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader
```

#### Cell 9: Create model.py (Neural Architectures)
```python
%%writefile src/model.py
import torch
import torch.nn as nn
from torchvision import models

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 128, 3, 2), nn.ReLU(), nn.MaxPool2d(3, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2), nn.ReLU(), nn.MaxPool2d(3, 2)
        )
        self.flatten_dim = 256 * 15 * 15
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)

class TransferResNet(nn.Module):
    def __init__(self, num_classes=2, fine_tune=False):
        super(TransferResNet, self).__init__()
        weights = models.ResNet50_Weights.DEFAULT
        self.base_model = models.resnet50(weights=weights)
        if not fine_tune:
            for param in self.base_model.parameters():
                param.requires_grad = False
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.base_model(x))

    def unfreeze_last_layers(self):
        for param in self.base_model.layer4.parameters():
            param.requires_grad = True

class CheXDS(nn.Module):
    def __init__(self, num_classes=2, fine_tune=False):
        super(CheXDS, self).__init__()
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        num_ftrs_dense = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_ftrs_dense, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, num_classes)
        )

        self.swin = models.swin_b(weights=models.Swin_B_Weights.DEFAULT)
        num_ftrs_swin = self.swin.head.in_features
        self.swin.head = nn.Sequential(
            nn.Linear(num_ftrs_swin, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, num_classes)
        )

        self.ensemble_weights = nn.Parameter(torch.tensor([0.5, 0.5]))

        if not fine_tune:
            for p in self.densenet.parameters(): p.requires_grad = False
            for p in self.densenet.classifier.parameters(): p.requires_grad = True
            for p in self.swin.parameters(): p.requires_grad = False
            for p in self.swin.head.parameters(): p.requires_grad = True

    def forward(self, x):
        out_dense = self.densenet(x)
        out_swin = self.swin(x)
        w = torch.nn.functional.softmax(self.ensemble_weights, dim=0)
        return w[0] * out_dense + w[1] * out_swin

    def unfreeze_last_layers(self):
        for p in self.densenet.features.denseblock4.parameters(): p.requires_grad = True
        for p in self.densenet.features.norm5.parameters(): p.requires_grad = True
        for p in self.swin.features[-2:].parameters(): p.requires_grad = True
```

#### Cell 10: Create train.py (Custom Loss, AMP, Loops)
```python
%%writefile src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.config import *
from src.utils import EarlyStopping
import os
from torch.cuda.amp import autocast, GradScaler

class CheXDSLoss(nn.Module):
    def __init__(self, class_ratios, gamma_pos=1, gamma_neg=4, m=0.05):
        super(CheXDSLoss, self).__init__()
        self.rho = torch.tensor(class_ratios).to(DEVICE)
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.m = m
        self.epsilon = 1e-7

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets = torch.nn.functional.one_hot(targets, num_classes=logits.size(1)).float()
        w = targets * torch.exp(1 - self.rho) + (1 - targets) * torch.exp(self.rho)
        p_m = torch.clamp(probs - self.m, min=0.0)
        pos_term = torch.pow(1 - probs, self.gamma_pos) * targets * torch.log(probs + self.epsilon)
        neg_term = torch.pow(p_m, self.gamma_neg) * (1 - targets) * torch.log(1 - p_m + self.epsilon)
        loss = -torch.sum(w * (pos_term + neg_term)) / logits.size(0)
        return loss

def calculate_class_ratios(loader):
    if hasattr(loader.dataset, 'dataset'):
        targets = [s[1] for s in loader.dataset.dataset.samples]
        indices = loader.dataset.indices
        subset_targets = torch.tensor(targets)[indices]
    else:
        subset_targets = torch.tensor([s[1] for s in loader.dataset])
    class_counts = torch.zeros(2)
    for i in range(2):
        class_counts[i] = (subset_targets == i).sum()
    return (class_counts / len(subset_targets)).tolist()

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    scaler = GradScaler(enabled=USE_AMP)
    running_loss, running_corrects, total_samples = 0.0, 0, 0
    pbar = tqdm(loader, desc="Training", unit="batch")

    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with autocast(enabled=USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += (preds == labels).sum().item()
        total_samples += labels.size(0)
        pbar.set_postfix({'loss': running_loss / total_samples})
    return running_loss / total_samples, running_corrects / total_samples

def validate(model, loader, criterion):
    model.eval()
    running_loss, running_corrects, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with autocast(enabled=USE_AMP):
                outputs = model(images)
                loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += (preds == labels).sum().item()
            total_samples += labels.size(0)
    return running_loss / total_samples, running_corrects / total_samples

def train_model(model, train_loader, val_loader, model_type='cnn'):
    model = model.to(DEVICE)
    if model_type == 'chexds':
        class_ratios = calculate_class_ratios(train_loader)
        criterion = CheXDSLoss(class_ratios)
        optimizer = optim.AdamW(model.parameters(), lr=LR_RESNET, weight_decay=1e-2)
        epochs = EPOCHS_RESNET
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR_CNN if model_type=='cnn' else LR_RESNET)
        epochs = EPOCHS_CNN if model_type=='cnn' else EPOCHS_RESNET

    early_stopping = EarlyStopping(patience=3, verbose=True, path=MODEL_SAVE_PATH)

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        early_stopping(val_loss, model)
        if early_stopping.early_stop: break

    model = early_stopping.load_best_weights(model)

    if model_type in ['resnet', 'chexds']:
        print("\n--- Starting Fine-Tuning ---")
        model.unfreeze_last_layers()
        optimizer_ft = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-2) if model_type=='chexds' else optim.Adam(model.parameters(), lr=0.0001)
        early_stopping = EarlyStopping(patience=3, verbose=True, path=MODEL_SAVE_PATH)
        for epoch in range(5):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_ft)
            val_loss, val_acc = validate(model, val_loader, criterion)
            print(f"FT Epoch {epoch+1}/5 | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
            early_stopping(val_loss, model)
            if early_stopping.early_stop: break
        model = early_stopping.load_best_weights(model)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Weights successfully saved to {MODEL_SAVE_PATH}")
    return model
```

---

### Phase 4: CLI & Orchestration Execution
**Objective**: Build the entry script and execute GPU-accelerated model training inside Colab.

#### Cell 11: Create main.py
```python
%%writefile main.py
import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.data_loader import get_data_loaders
from src.model import SimpleCNN, TransferResNet, CheXDS
from src.train import train_model

def main():
    parser = argparse.ArgumentParser(description="Pneumonia Detection Training")
    parser.add_argument('--model', type=str, choices=['cnn', 'resnet', 'chexds'], default='chexds')
    args = parser.parse_args()

    print("Initializing Data Loaders...")
    train_loader, val_loader, test_loader = get_data_loaders()

    print(f"Building {args.model.upper()} Model...")
    if args.model == 'cnn':
        model = SimpleCNN()
    elif args.model == 'resnet':
        model = TransferResNet()
    elif args.model == 'chexds':
        model = CheXDS()

    train_model(model, train_loader, val_loader, model_type=args.model)

if __name__ == "__main__":
    main()
```

#### Cell 12: Launch GPU Training
```bash
# Execute training (AMP + scaled batch size will save weights to Drive automatically)
!python main.py --model chexds
```

---

### Phase 5: Metrics & Performance Visualization
**Objective**: Run test dataset inference, print metrics, generate analytical plots, and display them inside the notebook.

#### Cell 13: Create visualize_results.py
```python
%%writefile visualize_results.py
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.data_loader import get_data_loaders
from src.model import CheXDS
from src.config import DEVICE, MODEL_SAVE_PATH

def main():
    _, _, test_loader = get_data_loaders()
    model = CheXDS()
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    y_true, y_pred, y_probs = [], [], []

    print("Running Inference across test data...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs[:, 1].cpu().numpy())

    classes = ['NORMAL', 'PNEUMONIA']
    print("\n" + "="*60)
    print("FINAL TEST CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=classes))

    os.makedirs('results', exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (CheX-DS)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('results/confusion_matrix.png', bbox_inches='tight')
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig('results/roc_curve.png', bbox_inches='tight')
    plt.close()
    print("Analytics charts successfully exported to results/ directory.")

if __name__ == "__main__":
    main()
```

#### Cell 14: Run Visualization
```bash
!python visualize_results.py
```

#### Cell 15: View Plots Inside Colab Notebook
```python
from IPython.display import Image, display

print("--- Confusion Matrix ---")
display(Image('results/confusion_matrix.png'))

print("\n--- ROC Curve ---")
display(Image('results/roc_curve.png'))
```

---

### Phase 6: GUI Deployment (Gradio Interface)
**Objective**: Build and launch the Gradio web dashboard server live from the GPU, outputting a secure public URL.

#### Cell 16: Create gradio_app.py
```python
%%writefile gradio_app.py
import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.model import CheXDS
from src.config import MODEL_SAVE_PATH, DEVICE

model = CheXDS()
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()

def predict(img):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_pil = Image.fromarray(img.astype('uint8'), 'RGB')
    tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    return {'NORMAL': float(probs[0]), 'PNEUMONIA': float(probs[1])}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Frontal X-Ray Scan"),
    outputs=gr.Label(num_top_classes=2, label="Classification Diagnosis"),
    title="Pneumonia Diagnosis Assistant",
    description="Dual branch DenseNet121 + Swin Transformer Ensemble running live on Colab GPU."
)

if __name__ == "__main__":
    demo.launch(share=True)
```

#### Cell 17: Expose Live UI Publicly
```python
# Starts the server live from Colab's GPU and outputs a secure public HF tunnel link
!python gradio_app.py
```
