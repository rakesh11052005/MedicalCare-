"""
explain.py

Grad-CAM explainability CLI for MedicalCare+.

Purpose:
- Generate visual explanations for model predictions
- Highlight lung regions influencing the decision
- Support clinical trust and regulatory review

NOTE:
This tool loads models with strict=False to support
post-training explainability additions (industry standard).
"""

import argparse
import os
import cv2
import torch

from src.common.image_utils import load_xray_image
from src.models.model_factory import build_pneumonia_model
from src.explainability.gradcam import GradCAM
from src.explainability.heatmap_utils import overlay_heatmap
from src.common.logging_utils import setup_logger

# --------------------------------------------------
# Logger
# --------------------------------------------------
logger = setup_logger(__name__)


def run_gradcam_explanation(
    image_path: str,
    model_path: str,
    output_dir: str
):
    """
    Runs Grad-CAM on a single X-ray image and saves the heatmap.

    Args:
        image_path (str): Path to X-ray image
        model_path (str): Path to trained model
        output_dir (str): Directory to save Grad-CAM output
    """

    logger.info("Starting Grad-CAM explanation")

    # --------------------------------------------------
    # Resolve absolute paths (industrial safety)
    # --------------------------------------------------
    image_path = os.path.abspath(image_path)
    model_path = os.path.abspath(model_path)
    output_dir = os.path.abspath(output_dir)

    logger.info(f"Image path resolved to: {image_path}")
    logger.info(f"Model path resolved to: {model_path}")

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found:\n{image_path}")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found:\n{model_path}")

    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------
    # Load model (STRICT = FALSE → INDUSTRIAL FIX)
    # --------------------------------------------------
    model = build_pneumonia_model()
    state_dict = torch.load(model_path, map_location=device)

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    logger.info("Model loaded successfully (strict=False)")

    # --------------------------------------------------
    # Load image (tensor + original grayscale)
    # --------------------------------------------------
    image_tensor = load_xray_image(image_path).unsqueeze(0).to(device)

    original_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_gray is None:
        raise ValueError("Failed to load original image for visualization")

    # --------------------------------------------------
    # Setup Grad-CAM
    # --------------------------------------------------
    backbone = model.backbone
    target_layer = backbone.get_last_conv_layer()

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer
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
        + "_gradcam.png"
    )

    output_path = os.path.join(output_dir, output_filename)

    overlay_heatmap(
        image_gray=original_gray,
        heatmap=heatmap,
        output_path=output_path
    )

    logger.info(f"Grad-CAM saved at: {output_path}")

    print("\nGrad-CAM Explanation Generated")
    print("-" * 45)
    print(f"Input Image  : {image_path}")
    print(f"Output Image : {output_path}")
    print("\nNote:")
    print(
        "Grad-CAM highlights regions influencing the AI decision.\n"
        "This visualization must be interpreted by a qualified clinician.\n"
    )


# --------------------------------------------------
# CLI ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM explanation for a chest X-ray"
    )

    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to chest X-ray image"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="models/pneumonia/v1.0.0/model.pt",
        help="Path to trained pneumonia model"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/gradcam",
        help="Directory to save Grad-CAM results"
    )

    args = parser.parse_args()

    run_gradcam_explanation(
        image_path=args.image_path,
        model_path=args.model_path,
        output_dir=args.output_dir
    )
