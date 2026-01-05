"""
explain.py

Grad-CAM explainability CLI for MedicalCare+.

Purpose:
- Generate visual explanations for model predictions
- Support Chest X-ray AND Brain MRI models
- Highlight image regions influencing the decision
- Support clinical trust and regulatory review

SUPPORTED TASKS:
- Pneumonia (single-disease)
- Multi-disease Chest X-ray
- Multi-class Brain Tumor MRI

IMPORTANT:
- This tool is for explainability ONLY
- NOT a diagnostic tool
- Visualizations require clinician interpretation
"""

import argparse
import os
import cv2
import torch

from src.common.image_utils import load_xray_image
from src.common.logging_utils import setup_logger
from src.explainability.gradcam import GradCAM
from src.explainability.heatmap_utils import overlay_heatmap
from src.models.model_factory import (
    build_pneumonia_model,
    build_multidisease_model,
    build_brain_tumor_model,
)

# --------------------------------------------------
# Logger
# --------------------------------------------------
logger = setup_logger(__name__)


# ==================================================
# CORE FUNCTION
# ==================================================
def run_gradcam_explanation(
    image_path: str,
    model_path: str,
    model_type: str,
    output_dir: str,
):
    """
    Runs Grad-CAM on a single medical image.

    Args:
        image_path (str): Path to medical image (X-ray or MRI)
        model_path (str): Path to trained model (.pt)
        model_type (str):
            - pneumonia
            - multidisease
            - brain_mri
        output_dir (str): Directory to save Grad-CAM output
    """

    logger.info("Starting Grad-CAM explanation")
    logger.info(f"Model type: {model_type}")

    # --------------------------------------------------
    # Resolve absolute paths
    # --------------------------------------------------
    image_path = os.path.abspath(image_path)
    model_path = os.path.abspath(model_path)
    output_dir = os.path.abspath(output_dir)

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found:\n{image_path}")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found:\n{model_path}")

    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------
    # Build model based on type
    # --------------------------------------------------
    if model_type == "pneumonia":
        model = build_pneumonia_model(pretrained=False)

    elif model_type == "multidisease":
        model = build_multidisease_model(pretrained=False)

    elif model_type == "brain_mri":
        model = build_brain_tumor_model(
            pretrained=False,
            num_classes=4,
        )

    else:
        raise ValueError(
            "Invalid model_type. Must be one of: "
            "pneumonia, multidisease, brain_mri"
        )

    # --------------------------------------------------
    # Load model weights (STRICT = FALSE for explainability)
    # --------------------------------------------------
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    logger.info("Model loaded successfully (strict=False)")

    # --------------------------------------------------
    # Load image
    # --------------------------------------------------
    image_tensor = load_xray_image(image_path).unsqueeze(0).to(device)

    original_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_gray is None:
        raise ValueError("Failed to load image for visualization")

    # --------------------------------------------------
    # Grad-CAM setup
    # --------------------------------------------------
    backbone = model.backbone
    target_layer = backbone.get_last_conv_layer()

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer,
    )

    # --------------------------------------------------
    # Generate heatmap
    # --------------------------------------------------
    heatmap = gradcam.generate(image_tensor)

    # --------------------------------------------------
    # Save overlay
    # --------------------------------------------------
    output_filename = (
        os.path.splitext(os.path.basename(image_path))[0]
        + f"_{model_type}_gradcam.png"
    )

    output_path = os.path.join(output_dir, output_filename)

    overlay_heatmap(
        image_gray=original_gray,
        heatmap=heatmap,
        output_path=output_path,
    )

    logger.info(f"Grad-CAM saved at: {output_path}")

    print("\nGrad-CAM Explanation Generated")
    print("-" * 50)
    print(f"Model Type   : {model_type}")
    print(f"Input Image : {image_path}")
    print(f"Output File : {output_path}")
    print(
        "\nNOTE:\n"
        "Grad-CAM highlights regions influencing the AI output.\n"
        "This visualization does NOT indicate diagnosis and must be\n"
        "interpreted by a qualified medical professional.\n"
    )


# ==================================================
# CLI ENTRY POINT
# ==================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM explanation for MedicalCare+ models"
    )

    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to medical image (X-ray or MRI)",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model (.pt)",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["pneumonia", "multidisease", "brain_mri"],
        help="Type of model to explain",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/gradcam",
        help="Directory to save Grad-CAM results",
    )

    args = parser.parse_args()

    run_gradcam_explanation(
        image_path=args.image_path,
        model_path=args.model_path,
        model_type=args.model_type,
        output_dir=args.output_dir,
    )
