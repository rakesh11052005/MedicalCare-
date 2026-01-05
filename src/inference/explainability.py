"""
explainability.py

High-level explainability interface for MedicalCare+.

Purpose:
- Provide a unified explainability entry point
- Bridge inference results with Grad-CAM visual evidence
- Support multi-disease (Chest X-ray) and multi-class (Brain MRI) models
- Enforce medical safety and interpretability standards

IMPORTANT MEDICAL NOTE:
Explainability DOES NOT justify or validate diagnoses.
It only highlights regions influencing the model.
"""

import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from src.common.logging_utils import setup_logger
from src.common.image_utils import load_xray_image, validate_xray_image
from src.models.model_factory import (
    build_multidisease_model,
    build_brain_tumor_model,
)
from src.explainability.gradcam import GradCAM
from src.explainability.heatmap_utils import overlay_heatmap
from src.common.constants import (
    FINDINGS,
    FINDING_THRESHOLDS,
)

# --------------------------------------------------
# Logger
# --------------------------------------------------
logger = setup_logger(__name__)


def generate_explanations(
    image_path: str,
    model_path: str,
    output_dir: str,
    task_type: str,
    requested_targets: Optional[List[str]] = None,
    device: str = "cpu",
) -> Dict:
    """
    Generate Grad-CAM explanations for a medical image.

    Args:
        image_path (str): Path to medical image (X-ray or MRI)
        model_path (str): Path to trained model
        output_dir (str): Directory to save heatmaps
        task_type (str):
            - "multidisease" → Chest X-ray
            - "brain_mri"    → Multi-class Brain MRI
        requested_targets (List[str] or None):
            - multidisease → findings to explain
            - brain_mri    → class names to explain
            - None → explain all valid targets
        device (str): cpu or cuda

    Returns:
        Dict: Structured explainability report
    """

    logger.info("Starting explainability pipeline")
    logger.info(f"Task type: {task_type}")

    # --------------------------------------------------
    # Resolve paths
    # --------------------------------------------------
    image_path = os.path.abspath(image_path)
    model_path = os.path.abspath(model_path)
    output_dir = os.path.abspath(output_dir)

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found:\n{image_path}")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found:\n{model_path}")

    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------
    # Load & validate image
    # --------------------------------------------------
    image = load_xray_image(image_path)

    if not validate_xray_image(image):
        raise ValueError("Invalid medical image after preprocessing")

    image_tensor = image.unsqueeze(0).to(device)

    # --------------------------------------------------
    # Build model based on task
    # --------------------------------------------------
    if task_type == "multidisease":
        model = build_multidisease_model(pretrained=False)

    elif task_type == "brain_mri":
        model = build_brain_tumor_model(
            pretrained=False,
            num_classes=4,
        )

    else:
        raise ValueError(
            "Invalid task_type. Must be 'multidisease' or 'brain_mri'"
        )

    model.load_state_dict(
        torch.load(model_path, map_location=device),
        strict=True,
    )
    model.to(device)
    model.eval()

    logger.info("Model loaded for explainability")

    # --------------------------------------------------
    # Grad-CAM setup
    # --------------------------------------------------
    backbone = model.backbone
    target_layer = backbone.get_last_conv_layer()

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer,
    )

    explanations: Dict = {}

    # ==================================================
    # MULTI-DISEASE (Chest X-ray)
    # ==================================================
    if task_type == "multidisease":

        if requested_targets is None:
            active_targets = FINDINGS
        else:
            invalid = [f for f in requested_targets if f not in FINDINGS]
            if invalid:
                raise ValueError(f"Invalid findings requested: {invalid}")
            active_targets = requested_targets

        with torch.no_grad():
            logits = model(image_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        for idx, finding in enumerate(FINDINGS):
            if finding not in active_targets:
                continue

            probability = float(probs[idx])
            low_t, high_t = FINDING_THRESHOLDS[finding]

            # Skip uncertain predictions
            if low_t < probability < high_t:
                logger.info(
                    f"Skipping {finding} explanation due to uncertainty"
                )
                explanations[finding] = {
                    "status": "skipped_uncertain",
                    "probability": probability,
                }
                continue

            heatmap = gradcam.generate(
                input_tensor=image_tensor,
                target_index=idx,
            )

            output_path = os.path.join(
                output_dir,
                f"{os.path.splitext(os.path.basename(image_path))[0]}"
                f"_{finding.lower().replace(' ', '_')}_gradcam.png"
            )

            overlay_heatmap(
                image_gray=image[0].cpu().numpy(),
                heatmap=heatmap,
                output_path=output_path,
            )

            explanations[finding] = {
                "status": "generated",
                "probability": probability,
                "heatmap_path": output_path,
            }

    # ==================================================
    # MULTI-CLASS (Brain MRI)
    # ==================================================
    else:  # brain_mri

        with torch.no_grad():
            logits = model(image_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        predicted_class = int(probs.argmax())
        confidence = float(probs[predicted_class])

        heatmap = gradcam.generate(
            input_tensor=image_tensor,
            target_index=predicted_class,
        )

        output_path = os.path.join(
            output_dir,
            f"{os.path.splitext(os.path.basename(image_path))[0]}"
            "_brain_mri_gradcam.png"
        )

        overlay_heatmap(
            image_gray=image[0].cpu().numpy(),
            heatmap=heatmap,
            output_path=output_path,
        )

        explanations["brain_mri"] = {
            "predicted_class_index": predicted_class,
            "confidence": confidence,
            "heatmap_path": output_path,
        }

    logger.info("Explainability pipeline completed successfully")
    return explanations
