import torch
import torch.nn as nn
from torchvision import models
from src.config import IMG_SIZE

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
    """
    Implements CheX-DS: An ensemble of DenseNet121 and Swin Transformer Base.
    Paper: 'CheX-DS: Improving Chest X-ray Image Classification...'
    """

    def __init__(self, num_classes=2, fine_tune=False):
        super(CheXDS, self).__init__()

        # --- Branch 1: DenseNet121---
        weights_dense = models.DenseNet121_Weights.DEFAULT
        self.densenet = models.densenet121(weights=weights_dense)

        # Modify DenseNet Classifier to match project classes
        num_ftrs_dense = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_ftrs_dense, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

        # --- Branch 2: Swin Transformer Base---
        # Note: Swin-B is large. Ensure your GPU has enough VRAM.
        weights_swin = models.Swin_B_Weights.DEFAULT
        self.swin = models.swin_b(weights=weights_swin)

        # Modify Swin Classifier (Head)
        num_ftrs_swin = self.swin.head.in_features
        self.swin.head = nn.Sequential(
            nn.Linear(num_ftrs_swin, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

        # Ensemble Weights
        # We use a learnable parameter to approximate optimal weighting
        self.ensemble_weights = nn.Parameter(torch.tensor([0.5, 0.5]))

        # Freeze base layers initially
        if not fine_tune:
            for param in self.densenet.parameters():
                param.requires_grad = False
            for param in self.densenet.classifier.parameters():
                param.requires_grad = True

            for param in self.swin.parameters():
                param.requires_grad = False
            for param in self.swin.head.parameters():
                param.requires_grad = True

    def forward(self, x):
        # Get predictions from both models
        out_dense = self.densenet(x)
        out_swin = self.swin(x)

        # Normalize ensemble weights to ensure they sum to 1
        w = torch.nn.functional.softmax(self.ensemble_weights, dim=0)

        # Weighted Ensemble
        out = w[0] * out_dense + w[1] * out_swin
        return out

    def unfreeze_last_layers(self):
        """
        Unfreezes the last blocks of both models for fine-tuning.
        """
        # Unfreeze DenseNet last block
        for param in self.densenet.features.denseblock4.parameters():
            param.requires_grad = True
        for param in self.densenet.features.norm5.parameters():
            param.requires_grad = True

        # Unfreeze Swin last layers (approx last 2 stages)
        # Swin structure: features -> [0,1,2,3,4,5,6,7]
        for param in self.swin.features[-2:].parameters():
            param.requires_grad = True
