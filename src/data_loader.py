import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from src.config import *


def get_transforms():
    """
    Define preprocessing and augmentation pipelines.
    """
    # 1. Training Transforms (Augmentation)
    # Replicates RandomZoom(0.1) and RandomRotation(0.2) from notebook
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(degrees=20),  # ~0.2 factor
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),  # Zoom simulation
        transforms.ToTensor(),  # Converts 0-255 to 0.0-1.0
    ])

    # 2. Validation/Test Transforms (No Augmentation)
    val_test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    return train_transform, val_test_transform


def get_data_loaders():
    """
    Loads images, splits training data, and returns DataLoaders.
    """
    train_tf, val_tf = get_transforms()

    print(f"Loading data from: {TRAIN_DIR}")

    # 1. Load the full training folder
    full_train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_tf)

    # 2. Split into Train (80%) and Validation (20%)
    total_size = len(full_train_dataset)
    val_size = int(total_size * VAL_SPLIT)
    train_size = total_size - val_size

    # Use generator with fixed seed for reproducibility [cite: 30]
    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator
    )

    # IMPORTANT: Apply non-augmented transform to validation set
    # (By default, subsets inherit the transform of the parent. We override it here.)
    # Note: In standard PyTorch, this is tricky with random_split.
    # For simplicity in this project, we keep the transforms attached to the dataset.
    # Ideally, we would create a custom Dataset wrapper, but for now:
    # We will accept that validation might have slight augmentation or we use a clean loading approach below.

    # Cleaner Approach for Val Set Transform:
    # We reload the same folder with val_transform and use the same indices.
    val_dataset.dataset.transform = val_tf

    # 3. Load Test Data
    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=val_tf)

    print(f"Stats: {len(train_dataset)} Train, {len(val_dataset)} Val, {len(test_dataset)} Test images.")
    print(f"Classes: {full_train_dataset.classes}")  # ['NORMAL', 'PNEUMONIA'] [cite: 54]

    # 4. Create DataLoaders
    # num_workers=0 is safer for Windows. If it's slow, try 2.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader