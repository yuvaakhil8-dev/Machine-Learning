from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V2_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    mobilenet_v2,
    mobilenet_v3_small,
    resnet18,
    resnet50,
)


def build_frame_encoder(name: str = "mobilenet_v2") -> tuple[nn.Module, object, int]:
    name = name.lower()
    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        encoder = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        return encoder.eval(), weights.transforms(), 512
    if name == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        encoder = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        return encoder.eval(), weights.transforms(), 2048
    if name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
        encoder = nn.Sequential(model.features, model.avgpool, nn.Flatten())
        return encoder.eval(), weights.transforms(), 576
    if name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weights)
        encoder = nn.Sequential(model.features, model.avgpool, nn.Flatten())
        return encoder.eval(), weights.transforms(), 1280

    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    encoder = nn.Sequential(model.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
    return encoder.eval(), weights.transforms(), 1280


@torch.inference_mode()
def encode_frame_batch(encoder: nn.Module, batch: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    encoder = encoder.to(device).eval()
    return encoder(batch.to(device)).detach().cpu()
