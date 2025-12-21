import os
import torch

# --- Hardware Setup ---
# Automatically detect GPU (CUDA for NVIDIA, MPS for Mac) or fall back to CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")

# --- Paths ---
# We use abspath to ensure it works regardless of where you run the command from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'chest_xray')

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
# Note: 'val' comes from splitting 'train', so no separate folder needed yet.

MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'pneumonia_model.pth')

# --- Hyperparameters ---
# From Notebook [cite: 21, 26, 30]
IMG_SIZE = 256
BATCH_SIZE = 32         # Reduced from 128 to ensure stability on local PCs
SEED = 417
VAL_SPLIT = 0.20        # 20% of training data goes to validation

# Training settings [cite: 161, 165, 216]
EPOCHS_CNN = 15
EPOCHS_RESNET = 10
LR_CNN = 0.001
LR_RESNET = 0.01        # High initial LR for ResNet, then fine-tuning