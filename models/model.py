import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes):
    model = models.resnet50(pretrained=True)

    for name, param in model.named_parameters():
        if "layer4" not in name:
            param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model