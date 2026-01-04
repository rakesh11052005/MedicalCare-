# File: MedicalCare+/src/training/metrics.py

"""
metrics.py

Evaluation metrics for MedicalCare+.

Medical & industrial guarantees:
- Supports single-disease and multi-disease tasks
- Uses probability-based metrics (AUROC, sensitivity)
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
    recall_score
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

    Args:
        logits (Tensor): Shape (B, 1)
        targets (Tensor): Shape (B,)

    Returns:
        Dict[str, float]
    """

    probs = torch.sigmoid(logits).squeeze()
    y_prob = _to_numpy(probs)
    y_true = _to_numpy(targets)

    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "auroc": roc_auc_score(y_true, y_prob)
    }

    logger.info(f"Pneumonia metrics computed: {metrics}")
    return metrics


# ==================================================
# MULTI-DISEASE METRICS (FINDINGS)
# ==================================================
def compute_multidisease_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    findings: List[str] = FINDINGS
) -> Dict[str, Dict[str, float]]:
    """
    Computes per-finding metrics for multi-label predictions.

    Args:
        logits (Tensor): Shape (B, num_findings)
        targets (Tensor): Shape (B, num_findings)
        findings (List[str]): Names of findings

    Returns:
        Dict[str, Dict[str, float]]:
            {
                "Pneumonia": {...},
                "COVID-19": {...},
                ...
            }
    """

    probs = torch.sigmoid(logits)
    y_prob = _to_numpy(probs)
    y_true = _to_numpy(targets)

    results = {}

    for idx, finding in enumerate(findings):
        prob_f = y_prob[:, idx]
        true_f = y_true[:, idx]

        # Binary predictions at 0.5 (evaluation only)
        pred_f = (prob_f >= 0.5).astype(int)

        # Handle edge cases safely
        if np.unique(true_f).size < 2:
            auroc = float("nan")
        else:
            auroc = roc_auc_score(true_f, prob_f)

        results[finding] = {
            "accuracy": accuracy_score(true_f, pred_f),
            "precision": precision_score(true_f, pred_f, zero_division=0),
            "recall": recall_score(true_f, pred_f, zero_division=0),
            "auroc": auroc
        }

    logger.info("Multi-disease metrics computed successfully")
    return results


# ==================================================
# UNCERTAINTY & ABSTENTION ANALYSIS
# ==================================================
def compute_abstention_stats(
    logits: torch.Tensor
) -> Dict[str, float]:
    """
    Computes abstention statistics based on confidence thresholds.

    Args:
        logits (Tensor): Shape (B, num_findings) or (B, 1)

    Returns:
        Dict[str, float]
    """

    probs = torch.sigmoid(logits)
    y_prob = _to_numpy(probs)

    abstain_mask = (
        (y_prob > LOW_CONFIDENCE_THRESHOLD) &
        (y_prob < HIGH_CONFIDENCE_THRESHOLD)
    )

    total = y_prob.size
    abstained = abstain_mask.sum()

    stats = {
        "total_predictions": int(total),
        "abstained_predictions": int(abstained),
        "abstention_rate": float(abstained / total)
    }

    logger.info(f"Abstention stats: {stats}")
    return stats
