from __future__ import annotations
import torch.nn as nn
from torchvision import models


def build_model(model_name: str, num_classes: int = 1, pretrained: bool = True):
    model_name = model_name.lower()

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "vgg16":
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        model = models.vgg16(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    elif model_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    elif model_name == "inception_v3":
        weights = models.Inception_V3_Weights.DEFAULT if pretrained else None
        model = models.inception_v3(weights=weights, aux_logits=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        if model.AuxLogits is not None:
            aux_in = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(aux_in, num_classes)

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def get_target_layer(model, model_name: str):
    model_name = model_name.lower()

    if model_name == "resnet50":
        return model.layer4[-1]
    elif model_name == "vgg16":
        return model.features[-1]
    elif model_name == "densenet121":
        return model.features[-1]
    elif model_name == "mobilenet_v2":
        return model.features[-1]
    elif model_name == "inception_v3":
        return model.Mixed_7c
    elif model_name == "efficientnet_b0":
        return model.features[-1]
    else:
        raise ValueError(f"No target layer mapping for: {model_name}")