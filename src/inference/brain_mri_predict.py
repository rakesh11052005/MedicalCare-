"""
brain_mri_predict.py

Industrial-grade single-image inference for multi-class Brain Tumor MRI
classification in MedicalCare+.

TASK TYPE:
    - Multi-class classification
      (Normal, Glioma, Meningioma, Pituitary Tumor)

DESIGN PRINCIPLES:
- Deterministic inference
- Explicit abstention for uncertain predictions
- No diagnostic claims
- Human-in-the-loop by design
- Shared safety policy across the system
- Explainability-compatible (Grad-CAM ready)

IMPORTANT:
This module provides AI-assisted decision support ONLY.
Final medical interpretation must be performed by a qualified clinician.
"""

from __future__ import annotations

import os
import argparse
from typing import Dict

import torch
import torch.nn.functional as F

from src.common.constants import (
    UNCERTAINTY_LOWER,
    UNCERTAINTY_UPPER,
)
from src.common.image_utils import (
    load_xray_image,
    validate_xray_image,
)
from src.common.logging_utils import setup_logger
from src.models.model_factory import build_brain_tumor_model
from src.datasets.brain_mri_dataset import BRAIN_MRI_CLASS_NAMES

# --------------------------------------------------
# Logger
# --------------------------------------------------
logger = setup_logger(__name__)


# ==================================================
# SINGLE-IMAGE INFERENCE
# ==================================================
def predict_brain_mri(
    image_path: str,
    model_path: str,
    device: str = "cpu",
) -> Dict:
    """
    Runs inference on a single Brain MRI image.

    Args:
        image_path (str): Path to MRI image file
        model_path (str): Path to trained model (.pt)
        device (str): 'cpu' or 'cuda'

    Returns:
        Dict: Safe, auditable prediction result
    """

    logger.info("Starting Brain MRI single-image prediction")

    # --------------------------------------------------
    # Resolve paths
    # --------------------------------------------------
    image_path = os.path.abspath(image_path)
    model_path = os.path.abspath(model_path)

    logger.info(f"Resolved image path: {image_path}")
    logger.info(f"Resolved model path: {model_path}")

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"MRI image not found:\n{image_path}")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found:\n{model_path}")

    # --------------------------------------------------
    # Load & preprocess image
    # --------------------------------------------------
    image_tensor = load_xray_image(image_path)

    if not validate_xray_image(image_tensor):
        raise ValueError("Invalid MRI image after preprocessing")

    image_tensor = image_tensor.unsqueeze(0).to(device)

    # --------------------------------------------------
    # Load model (STRICT & SAFE)
    # --------------------------------------------------
    model = build_brain_tumor_model(
        pretrained=False,
        freeze_backbone=True,
        num_classes=len(BRAIN_MRI_CLASS_NAMES),
    )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    logger.info("Brain MRI model loaded successfully")

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0)

    max_prob, pred_index = torch.max(probabilities, dim=0)
    confidence = float(max_prob.item())
    pred_index = int(pred_index.item())

    logger.info(
        f"Inference completed | "
        f"predicted_index={pred_index}, confidence={confidence:.4f}"
    )

    # --------------------------------------------------
    # Safety & abstention logic
    # --------------------------------------------------
    if UNCERTAINTY_LOWER < confidence < UNCERTAINTY_UPPER:
        logger.warning("Prediction in uncertainty zone → abstaining")

        return {
            "decision": "Uncertain",
            "predicted_class": None,
            "predicted_index": None,
            "confidence_score": confidence,
            "confidence_level": "Medium",
            "safety_status": "Abstained",
            "note": (
                "AI confidence is insufficient for a reliable classification. "
                "Clinical correlation and expert radiologist review are required."
            ),
        }

    # --------------------------------------------------
    # High-confidence result (NOT a diagnosis)
    # --------------------------------------------------
    predicted_label = BRAIN_MRI_CLASS_NAMES.get(pred_index, "Unknown")

    return {
        "decision": "Prediction Available",
        "predicted_class": predicted_label,
        "predicted_index": pred_index,
        "confidence_score": confidence,
        "confidence_level": "High",
        "safety_status": "Passed",
        "note": (
            "This output is an AI-assisted classification result. "
            "It must be reviewed and interpreted by a qualified clinician."
        ),
    }


# ==================================================
# CLI ENTRY POINT
# ==================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run Brain MRI tumor classification on a single image"
    )

    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to Brain MRI image",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="models/brain_tumor/v1.0.0/model.pt",
        help="Path to trained Brain MRI model",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    result = predict_brain_mri(
        image_path=args.image_path,
        model_path=args.model_path,
        device=device,
    )

    print("\nBrain MRI Prediction Result")
    print("-" * 50)
    for key, value in result.items():
        print(f"{key:18}: {value}")

    print(
        "\nNOTE:\n"
        "This is an AI-assisted result intended for clinical decision support.\n"
        "Final diagnosis and treatment decisions must be made by a qualified "
        "medical professional.\n"
    )
