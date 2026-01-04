"""
multi_disease_predict.py

Multi-disease chest X-ray inference for MedicalCare+.

Capabilities:
- Automatically evaluates ALL supported findings
- Allows user to select a subset of findings
- Applies uncertainty & abstention policy
- Generates per-finding Grad-CAM explainability
- Outputs FINDINGS (not diagnoses)
- Clinically safe, auditable, and deterministic

IMPORTANT:
This output is NOT a diagnosis.
It is an AI-assisted radiological analysis.
"""

# ==================================================
# Imports
# ==================================================
import argparse
import os
from pathlib import Path
from typing import List, Dict

import torch
import numpy as np
import cv2

from src.common.image_utils import load_xray_image, validate_xray_image
from src.models.model_factory import build_multidisease_model
from src.common.constants import FINDINGS
from src.inference.safety import safety_check
from src.common.logging_utils import setup_logger
from src.explainability.gradcam import GradCAM

# ==================================================
# Logger
# ==================================================
logger = setup_logger(__name__)


# ==================================================
# Grad-CAM visualization utility (READ-ONLY)
# ==================================================
def save_gradcam_overlay(
    original_image: np.ndarray,
    cam: np.ndarray,
    output_path: Path
) -> None:
    """
    Save Grad-CAM heatmap overlay (resolution-safe).

    Args:
        original_image (np.ndarray): RGB image (H, W, 3)
        cam (np.ndarray): Grad-CAM heatmap (h, w)
        output_path (Path): Output file path
    """

    # --------------------------------------------------
    # Resize CAM → image resolution (CRITICAL FIX)
    # --------------------------------------------------
    cam_resized = cv2.resize(
        cam,
        (original_image.shape[1], original_image.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )

    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = 0.6 * original_image + 0.4 * heatmap
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), overlay)

    if not success:
        raise IOError(f"Failed to write Grad-CAM image: {output_path}")


# ==================================================
# CORE PREDICTION LOGIC
# ==================================================
def predict_multidisease(
    image_path: str,
    model_path: str,
    selected_findings: List[str],
    device: str = "cpu",
    generate_gradcam: bool = False
) -> Dict[str, Dict]:
    """
    Runs multi-disease inference on a single chest X-ray.
    """

    logger.info("Starting multi-disease inference")

    # --------------------------------------------------
    # Resolve paths
    # --------------------------------------------------
    image_path = os.path.abspath(image_path)
    model_path = os.path.abspath(model_path)

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found:\n{image_path}")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found:\n{model_path}")

    # --------------------------------------------------
    # Load image (tensor only)
    # --------------------------------------------------
    image_tensor = load_xray_image(image_path)

    if not validate_xray_image(image_tensor):
        raise ValueError("Invalid X-ray image after preprocessing")

    # --------------------------------------------------
    # Convert tensor → numpy for visualization
    # (C, H, W) → (H, W, C)
    # --------------------------------------------------
    image_np = image_tensor.clone()
    image_np = image_np.permute(1, 2, 0).cpu().numpy()

    image_np = image_np - image_np.min()
    if image_np.max() > 0:
        image_np = image_np / image_np.max()
    image_np = (image_np * 255).astype(np.uint8)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------
    model = build_multidisease_model()
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.to(device)
    model.eval()

    logger.info("Multi-disease model loaded successfully")

    # --------------------------------------------------
    # Optional Grad-CAM setup
    # --------------------------------------------------
    gradcam = None
    if generate_gradcam:
        gradcam = GradCAM(
            model=model,
            target_layer=model.backbone.features.denseblock4
        )

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits).squeeze(0)

    # --------------------------------------------------
    # Post-processing per finding
    # --------------------------------------------------
    results: Dict[str, Dict] = {}

    for idx, finding in enumerate(FINDINGS):
        if selected_findings and finding not in selected_findings:
            continue

        prob = probabilities[idx].item()
        decision, confidence_level = safety_check(prob)

        results[finding] = {
            "decision": decision,
            "probability": round(prob, 4),
            "confidence_level": confidence_level
        }

        # --------------------------------------------------
        # Grad-CAM generation (only for likely findings)
        # --------------------------------------------------
        if generate_gradcam and decision == "Finding Likely":
            cam = gradcam.generate(
                input_tensor=image_tensor,
                target_index=idx
            )

            output_path = Path(
                f"outputs/gradcam/"
                f"{Path(image_path).stem}_{finding}.png"
            )

            save_gradcam_overlay(
                original_image=image_np,
                cam=cam,
                output_path=output_path
            )

            logger.info(
                f"Grad-CAM saved for {finding} → {output_path}"
            )

    logger.info("Multi-disease inference completed successfully")
    return results


# ==================================================
# CLI ENTRY POINT
# ==================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run multi-disease chest X-ray analysis"
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
        default="models/multidisease/v1.0.0/model.pt",
        help="Path to trained multi-disease model"
    )

    parser.add_argument(
        "--findings",
        type=str,
        nargs="*",
        default=[],
        help=(
            "Optional list of findings to display. "
            "If omitted, all findings are shown."
        )
    )

    parser.add_argument(
        "--gradcam",
        action="store_true",
        help="Generate Grad-CAM for likely findings"
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------
    # Validate requested findings
    # --------------------------------------------------
    selected_findings: List[str] = []
    if args.findings:
        for f in args.findings:
            if f not in FINDINGS:
                raise ValueError(
                    f"Unknown finding '{f}'. "
                    f"Valid options: {FINDINGS}"
                )
            selected_findings.append(f)

    results = predict_multidisease(
        image_path=args.image_path,
        model_path=args.model_path,
        selected_findings=selected_findings,
        device=device,
        generate_gradcam=args.gradcam
    )

    # --------------------------------------------------
    # PRINT OUTPUT (CLINICAL FORMAT)
    # --------------------------------------------------
    print("\n🩺 MedicalCare+ — AI X-ray Analysis")
    print("-" * 40)
    print("\nPossible Findings:\n")

    for finding, info in results.items():
        print(
            f"• {finding:<15}: "
            f"{info['decision']} "
            f"({info['probability']})"
        )

    print("\n" + "-" * 40)
    print("⚠️ Important Notice:")
    print(
        "This output is NOT a diagnosis.\n"
        "It is an AI-assisted radiological analysis\n"
        "and must be reviewed by a qualified clinician.\n"
    )
