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
# Detect if running in Google Colab to optimize file I/O speed
import sys
IS_COLAB = 'google.colab' in sys.modules or os.path.exists('/content')

if IS_COLAB:
    # Use fast local scratch disk for dataset reading (resolves Google Drive network latency bottleneck)
    DATA_DIR = '/content/data/chest_xray'
    
    # Try to use Google Drive for persistent storage of models and results
    drive_project_dir = '/content/drive/MyDrive/pneumonia-detection-cnn-xray'
    if os.path.exists('/content/drive/MyDrive'):
        BASE_DIR = drive_project_dir
    else:
        BASE_DIR = '/content'
else:
    # Local execution
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'chest_xray')

# Ensure output directories exist
os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
# Note: 'val' comes from splitting 'train', so no separate folder needed yet.

MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'pneumonia_model.pth')

# --- Hyperparameters ---
IMG_SIZE = 256
BATCH_SIZE = 32         # Reduced from 128 to ensure stability on local PCs
SEED = 417
VAL_SPLIT = 0.20        # 20% of training data goes to validation

# Training settings
EPOCHS_CNN = 15
EPOCHS_RESNET = 10
LR_CNN = 0.001
LR_RESNET = 0.01        # High initial LR for ResNet, then fine-tuning
