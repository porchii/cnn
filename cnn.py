"""Convolutional models for MNIST-to-FashionMNIST transfer experiments."""

from __future__ import annotations

import torch
from torch import nn


class CNNBackbone(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.features(images)


class CNNClassifier(nn.Module):
    """CNN with a replaceable classification head."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.backbone = CNNBackbone()
        self.head = nn.Linear(128, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(images))

    def freeze_backbone(self, frozen: bool = True) -> None:
        """Freeze or unfreeze feature-extractor parameters and BN statistics."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not frozen
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval() if frozen else module.train()

    def train(self, mode: bool = True) -> CNNClassifier:
        super().train(mode)
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d) and not any(
                parameter.requires_grad for parameter in module.parameters()
            ):
                module.eval()
        return self