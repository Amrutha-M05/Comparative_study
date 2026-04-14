import torch
import torch.nn as nn
from torchvision import models


class ViTClassifier(nn.Module):
    """
    Vision Transformer (ViT-B/16) for binary classification
    """

    def __init__(self, num_classes=1, pretrained=True):
        super(ViTClassifier, self).__init__()

        self.model = models.vit_b_16(weights="IMAGENET1K_V1" if pretrained else None)

        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def get_vit():
    return ViTClassifier(num_classes=1)