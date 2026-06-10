from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class MobileNetLSTMClassifier(nn.Module):
    """Small-dataset-friendly CNN-LSTM video classifier.

    MobileNetV2 acts as a frozen transfer-learning frame encoder, while the LSTM
    learns temporal activity patterns over sampled frames.
    """

    def __init__(self, hidden_size: int = 128, dropout: float = 0.35, freeze_cnn: bool = True) -> None:
        super().__init__()
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.cnn = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        if freeze_cnn:
            for parameter in self.cnn.parameters():
                parameter.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=1280,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = x.shape
        flat = x.view(batch * time, channels, height, width)
        features = self.pool(self.cnn(flat)).flatten(1)
        sequence = features.view(batch, time, -1)
        _, (hidden, _) = self.lstm(sequence)
        return self.classifier(hidden[-1])

