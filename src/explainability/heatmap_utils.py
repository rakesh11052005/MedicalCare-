"""
heatmap_utils.py

Utilities for visualizing Grad-CAM heatmaps on medical images
(Chest X-ray, Brain MRI).

Medical AI principles:
- No modification of original image data
- Visual overlay only (interpretability aid)
- Deterministic and reproducible outputs
- Modality-agnostic (X-ray, MRI)
- Clinician-friendly color mapping

IMPORTANT:
Heatmaps indicate regions influencing the AI output.
They do NOT imply diagnosis or causality.
"""

import os
import numpy as np
import cv2

from src.common.logging_utils import setup_logger

# --------------------------------------------------
# Logger
# --------------------------------------------------
logger = setup_logger(__name__)


def overlay_heatmap(
    image_gray: np.ndarray,
    heatmap: np.ndarray,
    output_path: str,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> None:
    """
    Overlays a Grad-CAM heatmap on a grayscale medical image
    and saves the result.

    Args:
        image_gray (np.ndarray):
            Original grayscale image (H, W).
            This array is NOT modified.

        heatmap (np.ndarray):
            Normalized heatmap (H, W), values in [0, 1].

        output_path (str):
            File path to save the overlay image.

        alpha (float):
            Heatmap transparency (0.0–1.0).

        colormap (int):
            OpenCV colormap (default: JET).

    Returns:
        None
    """

    logger.info("Starting heatmap overlay generation")

    # --------------------------------------------------
    # Input validation
    # --------------------------------------------------
    if image_gray.ndim != 2:
        raise ValueError("image_gray must be a 2D grayscale image")

    if heatmap.ndim != 2:
        raise ValueError("heatmap must be a 2D array")

    if not np.isfinite(heatmap).all():
        raise ValueError("heatmap contains non-finite values")

    if heatmap.min() < 0.0 or heatmap.max() > 1.0:
        raise ValueError(
            "heatmap values must be normalized to [0, 1]"
        )

    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be between 0.0 and 1.0")

    # --------------------------------------------------
    # Resize heatmap to match image resolution
    # --------------------------------------------------
    heatmap_resized = cv2.resize(
        heatmap,
        (image_gray.shape[1], image_gray.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    # --------------------------------------------------
    # Convert heatmap to color representation
    # --------------------------------------------------
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        colormap,
    )

    # --------------------------------------------------
    # Convert grayscale image to BGR
    # (copy to avoid in-place modification)
    # --------------------------------------------------
    image_bgr = cv2.cvtColor(
        image_gray.copy(),
        cv2.COLOR_GRAY2BGR,
    )

    # --------------------------------------------------
    # Overlay heatmap
    # --------------------------------------------------
    overlay = cv2.addWeighted(
        image_bgr,
        1.0 - alpha,
        heatmap_color,
        alpha,
        0,
    )

    # --------------------------------------------------
    # Save result safely
    # --------------------------------------------------
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    success = cv2.imwrite(output_path, overlay)
    if not success:
        raise IOError(
            f"Failed to save Grad-CAM overlay at {output_path}"
        )

    logger.info(f"Grad-CAM overlay saved at: {output_path}")
