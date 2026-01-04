"""
evaluate_multidisease.py

Evaluation script for the MedicalCare+ multi-disease chest X-ray model.

Industrial / clinical design principles:
- Multi-label (NOT multiclass) evaluation
- CSV-driven test split
- Disease-wise metrics (transparent & auditable)
- No clinical decision logic
- Deterministic, reproducible, regulator-friendly
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score
)

from src.datasets.unified_chestxray_dataset import UnifiedChestXrayDataset
from src.models.model_factory import build_multidisease_model
from src.common.constants import (
    FINDINGS,
    BATCH_SIZE
)
from src.common.logging_utils import setup_logger

# --------------------------------------------------
# Logger
# --------------------------------------------------
logger = setup_logger(__name__)


def evaluate_multidisease_model(
    model_path: str,
    test_csv_path: str,
    device: str = "cpu",
    save_metrics: bool = True
):
    """
    Evaluates a trained multi-disease model.

    Args:
        model_path (str): Path to trained model (.pt)
        test_csv_path (str): CSV test split
        device (str): cpu or cuda
        save_metrics (bool): Whether to save metrics.json
    """

    logger.info("Starting multi-disease model evaluation")

    model_path = os.path.abspath(model_path)
    test_csv_path = os.path.abspath(test_csv_path)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found:\n{model_path}")

    if not os.path.isfile(test_csv_path):
        raise FileNotFoundError(f"Test CSV not found:\n{test_csv_path}")

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------
    test_dataset = UnifiedChestXrayDataset(test_csv_path)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    logger.info(f"Test samples: {len(test_dataset)}")

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------
    model = build_multidisease_model()
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.to(device)
    model.eval()

    logger.info("Model loaded successfully")

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            all_targets.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.vstack(all_targets)
    y_prob = np.vstack(all_probs)

    # --------------------------------------------------
    # Metrics (per disease)
    # --------------------------------------------------
    metrics = {}

    for idx, disease in enumerate(FINDINGS):
        y_d_true = y_true[:, idx]
        y_d_prob = y_prob[:, idx]

        # Skip disease if no positive samples
        if y_d_true.sum() == 0:
            logger.warning(
                f"No positive samples for {disease}, skipping metrics"
            )
            continue

        metrics[disease] = {
            "auc": float(
                roc_auc_score(y_d_true, y_d_prob)
            ),
            "precision": float(
                precision_score(y_d_true, y_d_prob >= 0.5)
            ),
            "recall": float(
                recall_score(y_d_true, y_d_prob >= 0.5)
            )
        }

        logger.info(
            f"{disease} | "
            f"AUC: {metrics[disease]['auc']:.4f} | "
            f"Precision: {metrics[disease]['precision']:.4f} | "
            f"Recall: {metrics[disease]['recall']:.4f}"
        )

    # --------------------------------------------------
    # Save metrics (industrial traceability)
    # --------------------------------------------------
    if save_metrics:
        metrics_path = os.path.join(
            os.path.dirname(model_path),
            "metrics_multidisease.json"
        )

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Metrics saved to: {metrics_path}")

    logger.info("Multi-disease evaluation completed")

    return metrics


# --------------------------------------------------
# CLI ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":

    MODEL_PATH = "models/multidisease/v1.0.0/model.pt"
    TEST_CSV_PATH = "data/splits/multidisease_test.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    evaluate_multidisease_model(
        model_path=MODEL_PATH,
        test_csv_path=TEST_CSV_PATH,
        device=device
    )
