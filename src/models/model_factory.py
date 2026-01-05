"""
model_factory.py

Centralized model construction for MedicalCare+.

CRITICAL FILE:
- Single source of truth for model wiring
- Explicit task separation (binary / multi-label / multi-class)
- No training or inference logic
- Deterministic, auditable, regulator-safe

THIS FILE DEFINES OUTPUT DIMENSIONS.
"""

from typing import Literal

import torch
import torch.nn as nn

from src.common.logging_utils import setup_logger
from src.common.constants import (
    FINDINGS,
    NUM_FINDINGS,
)
from src.models.backbones.densenet121 import DenseNet121Backbone
from src.models.heads.pneumonia_head import PneumoniaHead
from src.models.heads.multi_disease_head import MultiDiseaseHead
from src.models.heads.brain_tumor_head import BrainTumorHead

# --------------------------------------------------
# Logger
# --------------------------------------------------
logger = setup_logger(__name__)


# ==================================================
# BASE WRAPPER MODEL
# ==================================================
class MedicalCareModel(nn.Module):
    """
    Unified wrapper: backbone + task-specific head.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        task_type: Literal[
            "single_disease",
            "multi_disease",
            "multi_class",
        ],
    ):
        super().__init__()

        self.backbone = backbone
        self.head = head
        self.task_type = task_type

        logger.info(
            "MedicalCareModel initialized | "
            f"task_type={task_type}, "
            f"backbone={backbone.__class__.__name__}, "
            f"head={head.__class__.__name__}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    Binary pneumonia detection model.

    Output:
        logits → shape (B, 1)

    Loss:
        BCEWithLogitsLoss
    """

    logger.info("Building Pneumonia model (binary)")

    backbone = DenseNet121Backbone(
        pretrained=pretrained,
        freeze=freeze_backbone,
    )

    # CRITICAL: binary task → 1 logit
    head = PneumoniaHead(
        in_features=backbone.out_features,
        num_classes=1,
    )

    model = MedicalCareModel(
        backbone=backbone,
        head=head,
        task_type="single_disease",
    )

    logger.info("Pneumonia model built successfully")
    return model


def build_multidisease_model(
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """
    Multi-label Chest X-ray model.

    Output:
        logits → shape (B, NUM_FINDINGS)

    Loss:
        BCEWithLogitsLoss
    """

    logger.info("Building multi-disease Chest X-ray model")
    logger.info(f"Findings: {FINDINGS}")

    backbone = DenseNet121Backbone(
        pretrained=pretrained,
        freeze=freeze_backbone,
    )

    head = MultiDiseaseHead(
        in_features=backbone.out_features,
        num_diseases=NUM_FINDINGS,
    )

    model = MedicalCareModel(
        backbone=backbone,
        head=head,
        task_type="multi_disease",
    )

    logger.info(
        f"Multi-disease model built (num_findings={NUM_FINDINGS})"
    )
    return model


def build_brain_tumor_model(
    pretrained: bool = True,
    freeze_backbone: bool = True,
    num_classes: int = 4,
) -> nn.Module:
    """
    Multi-class Brain Tumor MRI model.

    Output:
        logits → shape (B, num_classes)

    Loss:
        CrossEntropyLoss
    """

    logger.info("Building Brain Tumor MRI model")

    if num_classes <= 1:
        raise ValueError(
            "Brain tumor model requires num_classes > 1"
        )

    backbone = DenseNet121Backbone(
        pretrained=pretrained,
        freeze=freeze_backbone,
    )

    head = BrainTumorHead(
        in_features=backbone.out_features,
        num_classes=num_classes,
    )

    model = MedicalCareModel(
        backbone=backbone,
        head=head,
        task_type="multi_class",
    )

    logger.info(
        f"Brain tumor model built (num_classes={num_classes})"
    )
    return model
