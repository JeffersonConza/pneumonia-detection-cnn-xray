from src.data_loader import get_data_loaders

if __name__ == "__main__":
    try:
        train_dl, val_dl, test_dl = get_data_loaders()

        # Grab one batch to check shapes
        images, labels = next(iter(train_dl))
        print(f"Batch shape: {images.shape}")  # Should be [32, 3, 256, 256]
        print(f"Labels shape: {labels.shape}")  # Should be [32]
        print("Data Loading Successful")
    except Exception as e:
        print(f"Error: {e}")