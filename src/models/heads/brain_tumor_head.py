"""
brain_tumor_head.py

Industrial-grade multi-class classification head for Brain Tumor MRI
analysis in MedicalCare+.

SUPPORTED CLASSES (FIXED CONTRACT):
    0 -> Normal
    1 -> Glioma
    2 -> Meningioma
    3 -> Pituitary

DESIGN GUARANTEES:
- Produces RAW LOGITS (no softmax inside the model)
- Compatible with CrossEntropyLoss
- Deterministic and auditable behavior
- Grad-CAM friendly (no architectural obfuscation)
- Safe for medical decision-support systems

IMPORTANT:
This head predicts IMAGE-LEVEL FINDINGS, not diagnoses.
Final medical interpretation must be performed by a qualified clinician.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.common.logging_utils import setup_logger

# --------------------------------------------------
# Logger
# --------------------------------------------------
logger = setup_logger(__name__)


class BrainTumorHead(nn.Module):
    """
    Multi-class Brain Tumor classification head.

    Input:
        Feature maps from CNN backbone
        Shape -> (B, C, H, W)

    Output:
        Raw logits per class
        Shape -> (B, num_classes)

    Training:
        Loss -> torch.nn.CrossEntropyLoss

    Inference:
        Apply softmax OUTSIDE the model
    """

    def __init__(self, in_features: int, num_classes: int = 4):
        """
        Initialize BrainTumorHead.

        Args:
            in_features (int): Number of channels from backbone output
            num_classes (int): Number of tumor classes (default = 4)
        """
        super().__init__()

        if num_classes <= 1:
            raise ValueError(
                "BrainTumorHead requires num_classes > 1 "
                "for multi-class classification"
            )

        self.in_features = in_features
        self.num_classes = num_classes

        logger.info(
            "Initializing BrainTumorHead | "
            f"in_features={in_features}, num_classes={num_classes}"
        )

        # --------------------------------------------------
        # Global pooling (medical imaging best practice)
        # --------------------------------------------------
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # --------------------------------------------------
        # Linear classifier
        # NOTE:
        # - No activation here
        # - CrossEntropyLoss expects raw logits
        # --------------------------------------------------
        self.classifier = nn.Linear(
            in_features=self.in_features,
            out_features=self.num_classes
        )

        self._initialize_weights()

        logger.info("BrainTumorHead initialized successfully")

    # --------------------------------------------------
    # Forward Pass
    # --------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor):
                Feature maps from backbone
                Shape -> (B, C, H, W)

        Returns:
            torch.Tensor:
                Raw logits
                Shape -> (B, num_classes)
        """

        if x.ndim != 4:
            raise ValueError(
                "BrainTumorHead expects input tensor of shape (B, C, H, W)"
            )

        # Global average pooling
        x = self.global_pool(x)

        # Flatten (B, C, 1, 1) -> (B, C)
        x = torch.flatten(x, start_dim=1)

        # Linear classification
        logits = self.classifier(x)

        return logits

    # --------------------------------------------------
    # Weight Initialization
    # --------------------------------------------------
    def _initialize_weights(self) -> None:
        """
        Initializes classifier weights.

        Uses conservative initialization suitable for
        transfer learning in medical imaging.
        """

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        logger.info("BrainTumorHead weights initialized (Xavier uniform)")
