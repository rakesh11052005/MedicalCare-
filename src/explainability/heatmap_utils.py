"""
heatmap_utils.py

Utilities for visualizing Grad-CAM heatmaps on chest X-ray images.

Medical AI principles:
- No modification of original image data
- Visual overlay only (interpretability aid)
- Deterministic and reproducible outputs
- Clinician-friendly color mapping
"""

import os
from typing import Tuple

import cv2
import numpy as np


def overlay_heatmap(
    image_gray: np.ndarray,
    heatmap: np.ndarray,
    output_path: str,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET
) -> None:
    """
    Overlays a Grad-CAM heatmap on a grayscale X-ray image and saves it.

    Args:
        image_gray (np.ndarray): Original grayscale image (H, W)
        heatmap (np.ndarray): Normalized heatmap (H, W), values in [0, 1]
        output_path (str): File path to save the overlay image
        alpha (float): Heatmap transparency (0.0–1.0)
        colormap (int): OpenCV colormap (default: JET)

    Returns:
        None
    """

    # --------------------------------------------------
    # Validate inputs
    # --------------------------------------------------
    if image_gray.ndim != 2:
        raise ValueError("image_gray must be a 2D grayscale image")

    if heatmap.ndim != 2:
        raise ValueError("heatmap must be a 2D array")

    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be between 0.0 and 1.0")

    # --------------------------------------------------
    # Resize heatmap to match image
    # --------------------------------------------------
    heatmap_resized = cv2.resize(
        heatmap,
        (image_gray.shape[1], image_gray.shape[0])
    )

    # --------------------------------------------------
    # Convert heatmap to color
    # --------------------------------------------------
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)

    # --------------------------------------------------
    # Convert grayscale image to BGR
    # --------------------------------------------------
    image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

    # --------------------------------------------------
    # Overlay heatmap
    # --------------------------------------------------
    overlay = cv2.addWeighted(
        image_bgr,
        1.0 - alpha,
        heatmap_color,
        alpha,
        0
    )

    # --------------------------------------------------
    # Save result safely
    # --------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    success = cv2.imwrite(output_path, overlay)
    if not success:
        raise IOError(f"Failed to save Grad-CAM overlay at {output_path}")
