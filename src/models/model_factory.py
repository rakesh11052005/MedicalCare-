"""
model_factory.py

Centralized model construction for MedicalCare+.

Industrial design principles:
- Single source of truth for model wiring
- Explicit separation of single-disease vs multi-disease models
- No training logic, no inference logic
- Deterministic, auditable architecture creation
- Safe for medical ML & regulatory review

This file ONLY builds models.
"""

from typing import Literal

import torch
import torch.nn as nn

from src.common.constants import (
    NUM_CLASSES,
    NUM_DISEASES,
    DISEASES,
    MULTILABEL,
)
from src.common.logging_utils import setup_logger
from src.models.backbones.densenet121 import DenseNet121Backbone
from src.models.heads.pneumonia_head import PneumoniaHead
from src.models.heads.multi_disease_head import MultiDiseaseHead

# --------------------------------------------------
# Logger
# --------------------------------------------------
logger = setup_logger(__name__)


# ==================================================
# BASE WRAPPER MODEL
# ==================================================
class MedicalCareModel(nn.Module):
    """
    Unified model wrapper.

    Responsibilities:
    - Holds backbone + task-specific head
    - Enables explainability (Grad-CAM)
    - Keeps architecture explicit and traceable
    - No hidden logic or side effects
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        task_type: Literal["single_disease", "multi_disease"],
    ):
        super().__init__()

        self.backbone = backbone
        self.head = head
        self.task_type = task_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): Input image batch (B, C, H, W)

        Returns:
            logits (Tensor):
                - Single disease: (B, C)
                - Multi disease:  (B, D)
        """

        features = self.backbone(x)
        logits = self.head(features)
        return logits


# ==================================================
# FACTORY FUNCTIONS
# ==================================================

def build_pneumonia_model(
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """
    Builds a single-disease Pneumonia model.

    Characteristics:
    - Mutually exclusive classes (Normal vs Pneumonia)
    - Softmax-based training
    - CrossEntropy-compatible output

    Returns:
        nn.Module: Pneumonia classification model
    """

    logger.info("Creating single-disease Pneumonia model")

    if MULTILABEL:
        logger.warning(
            "MULTILABEL=True detected, but single-disease model requested. "
            "Proceeding with single-label head."
        )

    backbone = DenseNet121Backbone(
        pretrained=pretrained,
        freeze=freeze_backbone,
    )

    head = PneumoniaHead(
        in_features=backbone.out_features,
        num_classes=NUM_CLASSES,
    )

    model = MedicalCareModel(
        backbone=backbone,
        head=head,
        task_type="single_disease",
    )

    logger.info(
        "Pneumonia model built successfully "
        f"(classes={NUM_CLASSES}, pretrained={pretrained})"
    )
    return model


def build_multidisease_model(
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """
    Builds a multi-disease Chest X-ray model.

    Characteristics:
    - Multi-label classification
    - Each finding is independent
    - Sigmoid-based training
    - BCEWithLogits-compatible output

    Findings handled:
        - Pneumonia
        - Tuberculosis
        - COVID-19
        - Lung Opacity

    Returns:
        nn.Module: Multi-disease model
    """

    logger.info("Creating multi-disease Chest X-ray model")
    logger.info(f"Configured findings: {DISEASES}")

    if not MULTILABEL:
        logger.warning(
            "MULTILABEL=False detected for a multi-disease model. "
            "Ensure this is intentional."
        )

    backbone = DenseNet121Backbone(
        pretrained=pretrained,
        freeze=freeze_backbone,
    )

    head = MultiDiseaseHead(
        in_features=backbone.out_features,
        num_diseases=NUM_DISEASES,
    )

    model = MedicalCareModel(
        backbone=backbone,
        head=head,
        task_type="multi_disease",
    )

    logger.info(
        "Multi-disease model built successfully "
        f"(findings={NUM_DISEASES}, pretrained={pretrained})"
    )
    return model
