"""
metrics.py

Evaluation metrics for MedicalCare+.

Medical & industrial guarantees:
- Supports binary, multi-label, and multi-class tasks
- Uses probability-based metrics (AUROC, macro scores)
- Explicit abstention-aware reporting
- No hidden thresholds
- Deterministic and auditable outputs
"""

from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.common.logging_utils import setup_logger
from src.common.constants import (
    LOW_CONFIDENCE_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLD,
    FINDINGS
)

# --------------------------------------------------
# Logger
# --------------------------------------------------
logger = setup_logger(__name__)


# ==================================================
# UTILITY
# ==================================================
def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Safely convert tensor to NumPy."""
    return tensor.detach().cpu().numpy()


# ==================================================
# SINGLE-DISEASE METRICS (PNEUMONIA)
# ==================================================
def compute_pneumonia_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor
) -> Dict[str, float]:
    """
    Computes metrics for binary pneumonia classification.
    """

    probs = torch.sigmoid(logits).squeeze()
    y_prob = _to_numpy(probs)
    y_true = _to_numpy(targets)

    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "auroc": roc_auc_score(y_true, y_prob),
    }

    logger.info(f"Pneumonia metrics computed: {metrics}")
    return metrics


# ==================================================
# MULTI-DISEASE METRICS (CHEST X-RAY)
# ==================================================
def compute_multidisease_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    findings: List[str] = FINDINGS
) -> Dict[str, Dict[str, float]]:
    """
    Computes per-finding metrics for multi-label predictions.
    """

    probs = torch.sigmoid(logits)
    y_prob = _to_numpy(probs)
    y_true = _to_numpy(targets)

    results = {}

    for idx, finding in enumerate(findings):
        prob_f = y_prob[:, idx]
        true_f = y_true[:, idx]

        pred_f = (prob_f >= 0.5).astype(int)

        if np.unique(true_f).size < 2:
            auroc = float("nan")
        else:
            auroc = roc_auc_score(true_f, prob_f)

        results[finding] = {
            "accuracy": accuracy_score(true_f, pred_f),
            "precision": precision_score(true_f, pred_f, zero_division=0),
            "recall": recall_score(true_f, pred_f, zero_division=0),
            "auroc": auroc,
        }

    logger.info("Multi-disease metrics computed successfully")
    return results


# ==================================================
# MULTI-CLASS METRICS (BRAIN MRI)
# ==================================================
def compute_multiclass_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor
) -> Dict[str, float]:
    """
    Computes metrics for multi-class Brain MRI classification.

    Args:
        logits (Tensor): Shape (B, num_classes)
        targets (Tensor): Shape (B,) with class indices

    Returns:
        Dict[str, float]
    """

    probs = torch.softmax(logits, dim=1)
    y_prob = _to_numpy(probs)
    y_true = _to_numpy(targets)

    y_pred = y_prob.argmax(axis=1)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    }

    logger.info(f"Multi-class (Brain MRI) metrics computed: {metrics}")
    return metrics


# ==================================================
# UNCERTAINTY & ABSTENTION ANALYSIS
# ==================================================
def compute_abstention_stats(
    logits: torch.Tensor,
    task_type: str
) -> Dict[str, float]:
    """
    Computes abstention statistics.

    Args:
        logits (Tensor):
            - Binary / multi-label → sigmoid
            - Multi-class → softmax
        task_type (str):
            - "pneumonia"
            - "multidisease"
            - "brain_mri"

    Returns:
        Dict[str, float]
    """

    if task_type in ("pneumonia", "multidisease"):
        probs = torch.sigmoid(logits)
        y_prob = _to_numpy(probs)

        abstain_mask = (
            (y_prob > LOW_CONFIDENCE_THRESHOLD) &
            (y_prob < HIGH_CONFIDENCE_THRESHOLD)
        )

        total = y_prob.size
        abstained = abstain_mask.sum()

    elif task_type == "brain_mri":
        probs = torch.softmax(logits, dim=1)
        max_conf = probs.max(dim=1).values
        y_conf = _to_numpy(max_conf)

        abstain_mask = (
            (y_conf > LOW_CONFIDENCE_THRESHOLD) &
            (y_conf < HIGH_CONFIDENCE_THRESHOLD)
        )

        total = y_conf.size
        abstained = abstain_mask.sum()

    else:
        raise ValueError(
            "Invalid task_type for abstention stats"
        )

    stats = {
        "total_predictions": int(total),
        "abstained_predictions": int(abstained),
        "abstention_rate": float(abstained / total),
    }

    logger.info(f"Abstention stats: {stats}")
    return stats
