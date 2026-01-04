"""
pneumonia_head.py

Single-disease classification head for Pneumonia detection.

Medical & industrial guarantees:
- Binary or multi-class classification
- Produces raw logits (NO softmax)
- Compatible with CrossEntropyLoss
- Explainability-safe (Grad-CAM compatible)
- Deterministic and auditable

This head predicts a FINDING, not a diagnosis.
"""

import torch
import torch.nn as nn


class PneumoniaHead(nn.Module):
    """
    Classification head for Pneumonia detection.

    Expected usage:
        - Classes: Normal vs Pneumonia
        - Output: raw logits

    Output shape:
        (B, num_classes)
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        dropout: float = 0.2,
    ):
        """
        Args:
            in_features (int): Feature dimension from backbone
            num_classes (int): Number of output classes (usually 2)
            dropout (float): Dropout probability
        """
        super().__init__()

        if num_classes < 2:
            raise ValueError("num_classes must be >= 2 for PneumoniaHead")

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features (Tensor): Feature tensor (B, F)

        Returns:
            logits (Tensor): Raw class logits (B, num_classes)
        """
        return self.classifier(features)
