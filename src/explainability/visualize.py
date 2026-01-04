# File: MedicalCare+/src/explainability/visualize.py

"""
visualize.py

Industrial-grade Grad-CAM visualization utilities for MedicalCare+.

Design guarantees:
- Read-only visualization (no model interaction)
- Deterministic and reproducible outputs
- Clinically interpretable overlays
- Robust input validation
- Safe for audit, reporting, and regulatory review

IMPORTANT:
Grad-CAM visualizations highlight regions of model attention.
They do NOT localize disease and must NOT be interpreted as diagnosis.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import cv2

from src.common.logging_utils import setup_logger

# --------------------------------------------------
# Logger
# --------------------------------------------------
logger = setup_logger(__name__)


# ==================================================
# Core Visualization Utilities
# ==================================================
def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to uint8 [0, 255].

    Args:
        image (np.ndarray): Image array

    Returns:
        np.ndarray: Normalized uint8 image
    """

    if image.ndim not in (2, 3):
        raise ValueError("Image must be 2D or 3D array")

    img = image.astype(np.float32)

    min_val = img.min()
    max_val = img.max()

    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = np.zeros_like(img)

    img = (img * 255.0).clip(0, 255).astype(np.uint8)

    return img


def apply_colormap(
    heatmap: np.ndarray,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Apply OpenCV colormap to Grad-CAM heatmap.

    Args:
        heatmap (np.ndarray): Normalized heatmap (H, W)
        colormap (int): OpenCV colormap

    Returns:
        np.ndarray: Colorized heatmap (H, W, 3)
    """

    if heatmap.ndim != 2:
        raise ValueError("Heatmap must be 2D")

    heatmap_uint8 = (heatmap * 255).clip(0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, colormap)

    # Convert BGR → RGB for consistency
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    return colored


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image.

    Args:
        image (np.ndarray): Original image (H, W, 3)
        heatmap (np.ndarray): Grad-CAM heatmap (H, W)
        alpha (float): Heatmap transparency

    Returns:
        np.ndarray: Overlay image (H, W, 3)
    """

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must be RGB (H, W, 3)")

    if heatmap.ndim != 2:
        raise ValueError("Heatmap must be (H, W)")

    image_uint8 = normalize_image(image)

    color_heatmap = apply_colormap(heatmap)

    overlay = (
        (1.0 - alpha) * image_uint8 +
        alpha * color_heatmap
    )

    overlay = overlay.clip(0, 255).astype(np.uint8)

    return overlay


# ==================================================
# Public API
# ==================================================
def save_gradcam_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    output_path: Path,
    alpha: float = 0.4
) -> None:
    """
    Save Grad-CAM overlay image to disk.

    Args:
        image (np.ndarray): Original image (H, W, 3)
        heatmap (np.ndarray): Grad-CAM heatmap (H, W)
        output_path (Path): Output file path
        alpha (float): Heatmap transparency
    """

    logger.info(f"Saving Grad-CAM visualization → {output_path}")

    overlay = overlay_heatmap(
        image=image,
        heatmap=heatmap,
        alpha=alpha
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), overlay)

    if not success:
        raise IOError(f"Failed to write Grad-CAM image: {output_path}")

    logger.info("Grad-CAM visualization saved successfully")


def save_raw_heatmap(
    heatmap: np.ndarray,
    output_path: Path
) -> None:
    """
    Save raw Grad-CAM heatmap (no overlay).

    Args:
        heatmap (np.ndarray): Grad-CAM heatmap (H, W)
        output_path (Path): Output file path
    """

    logger.info(f"Saving raw Grad-CAM heatmap → {output_path}")

    if heatmap.ndim != 2:
        raise ValueError("Heatmap must be 2D")

    heatmap_uint8 = (heatmap * 255).clip(0, 255).astype(np.uint8)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), heatmap_uint8)

    if not success:
        raise IOError(f"Failed to write heatmap image: {output_path}")

    logger.info("Raw Grad-CAM heatmap saved successfully")
