"""
losses.py

Centralized loss functions for MedicalCare+.

Industrial & medical design principles:
- Explicit separation of binary, multi-label, and multi-class losses
- Numerically stable (logits-based)
- No hidden thresholds or post-processing
- Deterministic and auditable
- Safe for clinical AI workflows
"""

import torch
import torch.nn as nn

from src.common.logging_utils import setup_logger

# --------------------------------------------------
# Logger
# --------------------------------------------------
logger = setup_logger(__name__)


# ==================================================
# SINGLE-DISEASE LOSS (PNEUMONIA)
# ==================================================
class PneumoniaLoss(nn.Module):
    """
    Binary classification loss for pneumonia detection.

    Uses:
    - BCEWithLogitsLoss (numerically stable)
    - Expects raw logits from the model
    """

    def __init__(self, pos_weight: float | None = None):
        super().__init__()

        if pos_weight is not None:
            logger.info(
                f"Using weighted BCE loss (pos_weight={pos_weight})"
            )
            self.loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(pos_weight)
            )
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits (Tensor): Shape (B, 1)
            targets (Tensor): Shape (B,) or (B, 1)

        Returns:
            Tensor: scalar loss
        """

        targets = targets.float().view_as(logits)
        return self.loss_fn(logits, targets)


# ==================================================
# MULTI-DISEASE LOSS (CHEST X-RAY)
# ==================================================
class MultiDiseaseLoss(nn.Module):
    """
    Multi-label loss for chest X-ray findings.

    Each disease is treated independently.
    No softmax. No mutual exclusivity.
    """

    def __init__(
        self,
        pos_weights: torch.Tensor | None = None
    ):
        super().__init__()

        if pos_weights is not None:
            logger.info("Using weighted multi-disease BCE loss")
            self.loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=pos_weights
            )
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits (Tensor): Shape (B, num_findings)
            targets (Tensor): Shape (B, num_findings)

        Returns:
            Tensor: scalar loss
        """

        targets = targets.float()
        return self.loss_fn(logits, targets)


# ==================================================
# MULTI-CLASS LOSS (BRAIN MRI)
# ==================================================
class BrainMRILoss(nn.Module):
    """
    Multi-class loss for Brain Tumor MRI classification.

    Characteristics:
    - Mutually exclusive classes
    - Uses CrossEntropyLoss
    - Expects raw logits (NO softmax)
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None
    ):
        """
        Args:
            class_weights (Tensor | None):
                Optional class weighting for imbalance.
                Shape: (num_classes,)
        """
        super().__init__()

        if class_weights is not None:
            logger.info("Using weighted CrossEntropy loss for Brain MRI")
            self.loss_fn = nn.CrossEntropyLoss(
                weight=class_weights
            )
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits (Tensor): Shape (B, num_classes)
            targets (Tensor): Shape (B,) with class indices

        Returns:
            Tensor: scalar loss
        """

        targets = targets.long()
        return self.loss_fn(logits, targets)


# ==================================================
# LOSS FACTORY (SINGLE SOURCE OF TRUTH)
# ==================================================
def build_loss(
    task: str,
    pos_weight: float | None = None,
    pos_weights: torch.Tensor | None = None,
    class_weights: torch.Tensor | None = None
) -> nn.Module:
    """
    Factory method for selecting the correct loss.

    Args:
        task (str):
            - "pneumonia"
            - "multidisease"
            - "brain_mri"
        pos_weight (float | None):
            Used only for pneumonia
        pos_weights (Tensor | None):
            Used only for multi-disease
        class_weights (Tensor | None):
            Used only for brain MRI

    Returns:
        nn.Module: loss function
    """

    task = task.lower()

    if task == "pneumonia":
        logger.info("Building PneumoniaLoss")
        return PneumoniaLoss(pos_weight=pos_weight)

    if task == "multidisease":
        logger.info("Building MultiDiseaseLoss")
        return MultiDiseaseLoss(pos_weights=pos_weights)

    if task == "brain_mri":
        logger.info("Building BrainMRILoss")
        return BrainMRILoss(class_weights=class_weights)

    raise ValueError(
        f"Unknown task '{task}'. "
        "Supported tasks: pneumonia, multidisease, brain_mri"
    )
