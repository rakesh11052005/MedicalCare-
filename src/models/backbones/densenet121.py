# File: MedicalCare+/src/models/backbones/densenet121.py

"""
densenet121.py

DenseNet-121 backbone for MedicalCare+.

Design goals (industrial & medical-grade):
- ImageNet-pretrained feature extractor
- Explicit last convolutional layer exposure (Grad-CAM compatible)
- Freeze / unfreeze control for transfer learning
- Deterministic forward behavior
- Disease-agnostic (supports multi-disease heads)

IMPORTANT:
This backbone NEVER performs classification.
It ONLY extracts visual features.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights

from src.common.logging_utils import setup_logger

# --------------------------------------------------
# Logger
# --------------------------------------------------
logger = setup_logger(__name__)


class DenseNet121Backbone(nn.Module):
    """
    DenseNet-121 backbone for chest X-ray feature extraction.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze: bool = True
    ):
        """
        Initialize DenseNet-121 backbone.

        Args:
            pretrained (bool):
                Use ImageNet pretrained weights (RECOMMENDED)
            freeze (bool):
                Freeze backbone weights (default: True)
        """
        super().__init__()

        logger.info("Initializing DenseNet-121 backbone")

        # --------------------------------------------------
        # Load DenseNet-121 safely (torchvision-compliant)
        # --------------------------------------------------
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        densenet = models.densenet121(weights=weights)

        # --------------------------------------------------
        # Feature extractor (convolutional trunk)
        # --------------------------------------------------
        self.features: nn.Sequential = densenet.features

        # --------------------------------------------------
        # Output feature dimension
        # --------------------------------------------------
        self.out_features: int = densenet.classifier.in_features

        # --------------------------------------------------
        # Identify last convolutional layer (Grad-CAM)
        # --------------------------------------------------
        self._last_conv_layer = self.features.denseblock4.denselayer16.conv2

        # --------------------------------------------------
        # Freeze parameters if requested
        # --------------------------------------------------
        if freeze:
            logger.info("Freezing DenseNet-121 backbone parameters")
            for param in self.features.parameters():
                param.requires_grad = False
        else:
            logger.info("Backbone parameters are trainable")

        logger.info(
            f"DenseNet-121 backbone ready | out_features={self.out_features}"
        )

    # ==================================================
    # FORWARD
    # ==================================================
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through convolutional backbone.

        Args:
            x (torch.Tensor):
                Input tensor of shape (B, 3, H, W)

        Returns:
            torch.Tensor:
                Feature map tensor
                Shape: (B, C, H', W')
        """

        if x.ndim != 4:
            raise ValueError(
                "Expected input tensor of shape (B, C, H, W)"
            )

        return self.features(x)

    # ==================================================
    # GRAD-CAM SUPPORT (CRITICAL)
    # ==================================================
    def get_last_conv_layer(self) -> nn.Module:
        """
        Returns the last convolutional layer.

        REQUIRED for:
        - Grad-CAM
        - Explainability audits
        - Regulatory transparency

        Returns:
            nn.Module
        """

        return self._last_conv_layer
