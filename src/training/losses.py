# File: MedicalCare+/src/training/losses.py

"""
losses.py

Centralized loss functions for MedicalCare+.

Industrial & medical design principles:
- Explicit separation of single-label vs multi-label losses
- Numerically stable (logits-based)
- Compatible with uncertainty-aware training
- No hidden thresholds or post-processing
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
        """
        Args:
            pos_weight (float | None):
                Optional positive class weighting to handle imbalance.
                If None, no weighting is applied.
        """
        super().__init__()

        if pos_weight is not None:
            logger.info(f"Using weighted BCE loss (pos_weight={pos_weight})")
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

        # Ensure shape consistency
        targets = targets.float().view_as(logits)

        loss = self.loss_fn(logits, targets)
        return loss


# ==================================================
# MULTI-DISEASE LOSS (FINDINGS-BASED)
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
        """
        Args:
            pos_weights (Tensor | None):
                Shape (num_findings,)
                Allows per-disease imbalance handling.
        """
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
        loss = self.loss_fn(logits, targets)
        return loss


# ==================================================
# LOSS FACTORY (SAFE & EXPLICIT)
# ==================================================
def build_loss(
    task: str = "pneumonia",
    pos_weight: float | None = None,
    pos_weights: torch.Tensor | None = None
) -> nn.Module:
    """
    Factory method for selecting the correct loss.

    Args:
        task (str):
            - "pneumonia"
            - "multidisease"
        pos_weight (float | None):
            Used only for pneumonia
        pos_weights (Tensor | None):
            Used only for multi-disease

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

    raise ValueError(
        f"Unknown task '{task}'. "
        f"Supported tasks: pneumonia, multidisease"
    )
