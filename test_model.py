import torch
from src.model import SimpleCNN, TransferResNet, CheXDS
from src.config import IMG_SIZE


def test():
    # Create dummy input [Batch, Channels, Height, Width]
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    # Test CNN
    cnn = SimpleCNN()
    output_cnn = cnn(dummy_input)
    print(f"CNN Output Shape: {output_cnn.shape}")  # Should be [1, 2]

    # Test ResNet
    resnet = TransferResNet()
    output_res = resnet(dummy_input)
    print(f"ResNet Output Shape: {output_res.shape}")  # Should be [1, 2]

    # Test CheXDS
    chexds = CheXDS()
    output_chexds = chexds(dummy_input)
    print(f"CheXDS Output Shape: {output_chexds.shape}")  # Should be [1, 2]

    print("Models built successfully.")


if __name__ == "__main__":
    test()