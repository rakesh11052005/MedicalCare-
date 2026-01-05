"""
evaluate_brain_mri.py

Industrial-grade evaluation pipeline for multi-class Brain Tumor MRI
classification in MedicalCare+.

TASK TYPE:
    - Multi-class classification (Normal, Glioma, Meningioma, Pituitary)

DESIGN PRINCIPLES:
- Deterministic evaluation
- Abstention-aware reporting (confidence-based)
- Class-wise and macro metrics
- Confusion matrix for auditability
- No diagnostic claims
- Safe for regulatory and clinical review

IMPORTANT:
This script evaluates a decision-support model.
It does NOT provide diagnoses.
"""

from __future__ import annotations

import os
import json
import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from src.common.constants import (
    BATCH_SIZE,
    RANDOM_SEED,
    UNCERTAINTY_LOWER,
    UNCERTAINTY_UPPER,
)
from src.common.logging_utils import setup_logger
from src.datasets.brain_mri_dataset import (
    BrainMRIDataset,
    BRAIN_MRI_CLASS_NAMES,
)
from src.models.model_factory import build_brain_tumor_model

# --------------------------------------------------
# Logger
# --------------------------------------------------
logger = setup_logger(__name__)


# ==================================================
# REPRODUCIBILITY
# ==================================================
def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to {seed}")


# ==================================================
# EVALUATION
# ==================================================
def evaluate_brain_mri_model(
    model_path: str,
    test_csv: str,
    output_dir: str,
    device: str = "cpu",
) -> Dict:
    """
    Evaluates a trained Brain Tumor MRI model.

    Args:
        model_path (str): Path to trained model (.pt)
        test_csv (str): Path to test CSV split
        output_dir (str): Directory to save evaluation metrics
        device (str): 'cpu' or 'cuda'

    Returns:
        Dict: Evaluation summary
    """

    logger.info("Starting Brain MRI model evaluation")
    logger.info(f"Device: {device}")

    _set_seed(RANDOM_SEED)

    # --------------------------------------------------
    # Resolve paths
    # --------------------------------------------------
    model_path = os.path.abspath(model_path)
    test_csv = os.path.abspath(test_csv)
    output_dir = os.path.abspath(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found:\n{model_path}")

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------
    test_dataset = BrainMRIDataset(test_csv)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    logger.info(f"Test samples: {len(test_dataset)}")

    # --------------------------------------------------
    # Load model (STRICT)
    # --------------------------------------------------
    model = build_brain_tumor_model(
        pretrained=False,
        freeze_backbone=True,
        num_classes=len(BRAIN_MRI_CLASS_NAMES),
    )

    model.load_state_dict(
        torch.load(model_path, map_location=device),
        strict=True,
    )

    model.to(device)
    model.eval()

    logger.info("Brain MRI model loaded successfully")

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[float] = []
    abstained = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = F.softmax(logits, dim=1)

            max_probs, preds = torch.max(probs, dim=1)

            for i in range(len(labels)):
                confidence = max_probs[i].item()

                y_true.append(labels[i].item())
                y_prob.append(confidence)

                # -----------------------------
                # Abstention logic
                # -----------------------------
                if UNCERTAINTY_LOWER < confidence < UNCERTAINTY_UPPER:
                    y_pred.append(-1)
                    abstained += 1
                else:
                    y_pred.append(preds[i].item())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # --------------------------------------------------
    # Metrics (exclude abstentions)
    # --------------------------------------------------
    valid_mask = y_pred != -1
    coverage = valid_mask.mean()
    abstention_rate = 1.0 - coverage

    if coverage > 0:
        accuracy = accuracy_score(y_true[valid_mask], y_pred[valid_mask])

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true[valid_mask],
            y_pred[valid_mask],
            labels=list(BRAIN_MRI_CLASS_NAMES.keys()),
            average=None,
            zero_division=0,
        )

        macro_precision, macro_recall, macro_f1, _ = (
            precision_recall_fscore_support(
                y_true[valid_mask],
                y_pred[valid_mask],
                average="macro",
                zero_division=0,
            )
        )

        conf_matrix = confusion_matrix(
            y_true[valid_mask],
            y_pred[valid_mask],
            labels=list(BRAIN_MRI_CLASS_NAMES.keys()),
        )
    else:
        accuracy = 0.0
        precision = recall = f1 = []
        macro_precision = macro_recall = macro_f1 = 0.0
        conf_matrix = []

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------
    logger.info(f"Coverage              : {coverage:.4f}")
    logger.info(f"Abstention Rate       : {abstention_rate:.4f}")
    logger.info(f"Accuracy (Effective)  : {accuracy:.4f}")
    logger.info(f"Macro Precision       : {macro_precision:.4f}")
    logger.info(f"Macro Recall          : {macro_recall:.4f}")
    logger.info(f"Macro F1-score        : {macro_f1:.4f}")
    logger.info("Confusion Matrix (Non-Abstained):")
    logger.info(f"\n{conf_matrix}")

    # --------------------------------------------------
    # Save metrics
    # --------------------------------------------------
    metrics = {
        "coverage": float(coverage),
        "abstention_rate": float(abstention_rate),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": {
            BRAIN_MRI_CLASS_NAMES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
            }
            for i in range(len(precision))
        },
        "confusion_matrix": conf_matrix.tolist()
        if isinstance(conf_matrix, np.ndarray)
        else conf_matrix,
    }

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Brain MRI evaluation completed successfully")
    logger.info(f"Metrics saved to: {metrics_path}")

    return metrics


# ==================================================
# CLI ENTRY POINT
# ==================================================
if __name__ == "__main__":

    MODEL_PATH = "models/brain_tumor/v1.0.0/model.pt"
    TEST_CSV = "data/splits/brain_mri_test.csv"
    OUTPUT_DIR = "models/brain_tumor/v1.0.0"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    evaluate_brain_mri_model(
        model_path=MODEL_PATH,
        test_csv=TEST_CSV,
        output_dir=OUTPUT_DIR,
        device=device,
    )
