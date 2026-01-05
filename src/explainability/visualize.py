"""
visualize.py

Industrial-grade Grad-CAM visualization utilities for MedicalCare+.

Design guarantees:
- Read-only visualization (no model interaction)
- Deterministic and reproducible outputs
- Clinically interpretable overlays
- Modality-agnostic (Chest X-ray, Brain MRI)
- Robust numerical validation
- Safe for audit, reporting, and regulatory review

IMPORTANT:
Grad-CAM visualizations highlight regions influencing the AI output.
They do NOT localize disease and must NOT be interpreted as diagnosis.
"""

from pathlib import Path
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
        image (np.ndarray): Image array (grayscale or RGB)

    Returns:
        np.ndarray: Normalized uint8 image
    """

    if image.ndim not in (2, 3):
        raise ValueError("Image must be 2D (grayscale) or 3D (RGB)")

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
        heatmap (np.ndarray): Normalized heatmap (H, W), values in [0,1]
        colormap (int): OpenCV colormap

    Returns:
        np.ndarray: Colorized heatmap (H, W, 3)
    """

    if heatmap.ndim != 2:
        raise ValueError("Heatmap must be 2D")

    if not np.isfinite(heatmap).all():
        raise ValueError("Heatmap contains non-finite values")

    if heatmap.min() < 0.0 or heatmap.max() > 1.0:
        raise ValueError("Heatmap must be normalized to [0, 1]")

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, colormap)

    # OpenCV returns BGR → convert to RGB
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    return colored


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on an RGB medical image.

    Args:
        image (np.ndarray):
            Original image (H, W, 3).
            May originate from X-ray or MRI after RGB conversion.

        heatmap (np.ndarray):
            Grad-CAM heatmap (H, W), normalized [0,1].

        alpha (float):
            Heatmap transparency (0.0–1.0).

    Returns:
        np.ndarray: Overlay image (H, W, 3)
    """

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must be RGB with shape (H, W, 3)")

    if heatmap.ndim != 2:
        raise ValueError("Heatmap must be 2D")

    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be between 0.0 and 1.0")

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
        image (np.ndarray): Original RGB image (H, W, 3)
        heatmap (np.ndarray): Grad-CAM heatmap (H, W), normalized [0,1]
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
        heatmap (np.ndarray): Grad-CAM heatmap (H, W), normalized [0,1]
        output_path (Path): Output file path
    """

    logger.info(f"Saving raw Grad-CAM heatmap → {output_path}")

    if heatmap.ndim != 2:
        raise ValueError("Heatmap must be 2D")

    if not np.isfinite(heatmap).all():
        raise ValueError("Heatmap contains non-finite values")

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), heatmap_uint8)
    if not success:
        raise IOError(f"Failed to write heatmap image: {output_path}")

    logger.info("Raw Grad-CAM heatmap saved successfully")
