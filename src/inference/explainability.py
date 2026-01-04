# File: MedicalCare+/src/inference/explainability.py

"""
explainability.py

High-level explainability interface for MedicalCare+.

Purpose:
- Provide a unified explainability entry point
- Bridge inference results with Grad-CAM visual evidence
- Enforce medical safety and interpretability standards
- Abstract low-level explainability mechanics

IMPORTANT MEDICAL NOTE:
Explainability DOES NOT justify or validate diagnoses.
It only highlights regions influencing the model.
"""

import os
from typing import Dict, List

import torch

from src.common.logging_utils import setup_logger
from src.common.image_utils import load_xray_image, validate_xray_image
from src.models.model_factory import build_multidisease_model
from src.explainability.gradcam import GradCAM
from src.explainability.heatmap_utils import overlay_heatmap
from src.common.constants import (
    FINDINGS,
    FINDING_THRESHOLDS
)

# --------------------------------------------------
# Logger
# --------------------------------------------------
logger = setup_logger(__name__)


def generate_explanations(
    image_path: str,
    model_path: str,
    output_dir: str,
    requested_findings: List[str] = None,
    device: str = "cpu"
) -> Dict:
    """
    Generates Grad-CAM explanations for selected findings.

    Args:
        image_path (str): Path to chest X-ray image
        model_path (str): Path to trained multi-disease model
        output_dir (str): Directory to save heatmaps
        requested_findings (List[str] or None):
            - None → explain all findings
            - List → explain selected findings
        device (str): cpu or cuda

    Returns:
        Dict: Mapping finding → explanation image path
    """

    logger.info("Starting explainability pipeline")

    image_path = os.path.abspath(image_path)
    model_path = os.path.abspath(model_path)
    output_dir = os.path.abspath(output_dir)

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found:\n{image_path}")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found:\n{model_path}")

    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------
    # Resolve findings to explain
    # --------------------------------------------------
    if not requested_findings:
        active_findings = FINDINGS
    else:
        invalid = [f for f in requested_findings if f not in FINDINGS]
        if invalid:
            raise ValueError(f"Invalid findings requested: {invalid}")
        active_findings = requested_findings

    logger.info(f"Generating explanations for: {active_findings}")

    # --------------------------------------------------
    # Load image
    # --------------------------------------------------
    image = load_xray_image(image_path)

    if not validate_xray_image(image):
        raise ValueError("Invalid X-ray image after preprocessing")

    image_tensor = image.unsqueeze(0).to(device)

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------
    model = build_multidisease_model()
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.to(device)
    model.eval()

    logger.info("Model loaded for explainability")

    # --------------------------------------------------
    # Prepare Grad-CAM
    # --------------------------------------------------
    backbone = model.backbone
    target_layer = backbone.get_last_conv_layer()

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer
    )

    # --------------------------------------------------
    # Forward pass (shared)
    # --------------------------------------------------
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    explanations = {}

    # --------------------------------------------------
    # Generate per-finding explanations
    # --------------------------------------------------
    for idx, finding in enumerate(FINDINGS):
        if finding not in active_findings:
            continue

        probability = probs[idx]
        low_t, high_t = FINDING_THRESHOLDS[finding]

        # Only explain if model is confident enough
        if low_t < probability < high_t:
            logger.info(
                f"Skipping {finding} explanation due to uncertainty"
            )
            continue

        heatmap = gradcam.generate(
            image_tensor=image_tensor,
            target_index=idx
        )

        output_path = os.path.join(
            output_dir,
            f"{os.path.splitext(os.path.basename(image_path))[0]}"
            f"_{finding.lower().replace(' ', '_')}_gradcam.png"
        )

        overlay_heatmap(
            image_gray=image[0].cpu().numpy(),
            heatmap=heatmap,
            output_path=output_path
        )

        explanations[finding] = {
            "probability": float(probability),
            "heatmap_path": output_path
        }

        logger.info(
            f"Grad-CAM generated for {finding}: {output_path}"
        )

    logger.info("Explainability pipeline completed")

    return explanations
