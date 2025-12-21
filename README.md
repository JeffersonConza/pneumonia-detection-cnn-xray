# Pneumonia Detection: CheX-DS Implementation (PyTorch)

This project implements a state-of-the-art modular Deep Learning pipeline for detecting Pneumonia from Chest X-Rays. 

It features the **CheX-DS** architecture (DenseNet121 + Swin Transformer Ensemble) and uses a **Weighted Asymmetric Loss** to handle class imbalance, achieving **99% Sensitivity (Recall)** on pneumonia cases.

## Architectures Implemented
1.  **Simple CNN**: Lightweight custom baseline.
2.  **ResNet50**: Transfer learning with ResNet50V2.
3.  **CheX-DS**: Ensemble of DenseNet121 and Swin Transformer Base with learnable ensemble weights.

## Project Structure
```text
pneumonia-detection-modular/
├── data/               # Dataset (Auto-downloaded)
├── models/             # Saved weights (excluded from git)
├── src/
│   ├── config.py       # Hyperparameters & Hardware settings
│   ├── model.py        # Model Architectures (CheXDS, ResNet, CNN)
│   ├── train.py        # Training loops & Custom Loss functions
│   ├── data_loader.py  # Data pipeline & Augmentation
│   └── evaluate.py     # Evaluation metrics
├── main.py             # CLI Entry point
├── run_experiments.py  # Automated benchmarking script
├── inference.py        # Single image prediction
└── visualize_results.py # Confusion Matrix & ROC generation
```

## 🚀 Quick Start

### 1. Installation
```bash
# Clone repo
git clone https://github.com/JeffersonConza/pneumonia-detection-cnn-xray.git
cd pneumonia-detection-modular

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Setup

Automatically download and extract the dataset (1.2 GB):
```bash
python download_data.py
```

### 3. Training

Train the CheX-DS model (Recommended):
```bash
python main.py --model chexds
```

Or run the full benchmark suite (CNN -> ResNet -> CheX-DS):
```bash
python run_experiments.py
```

## 📊 Benchmark Results

| Model | Accuracy | PNEUMONIA Recall (Sensitivity) | NORMAL Recall | Test Loss |
| --- | --- | --- | --- | --- |
| **Simple CNN** | 77.88% | 0.95 | 0.50 | 0.8516 |
| **ResNet50** | 83.33% | 0.96 | 0.61 | 1.0329 |
| **CheX-DS** | **85.26%** | **0.99** | **0.63** | **0.2856** |

**Key Insight:** The CheX-DS model minimizes False Negatives (missing sick patients) better than any other model, making it ideal for medical screening.

## Usage

To test the model on a random X-ray from the test set:
```bash
python inference.py
```

To generate Confusion Matrix and ROC Curves:
```bash
python visualize_results.py
```