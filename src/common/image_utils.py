# File: MedicalCare+/src/common/image_utils.py

"""
image_utils.py

MedicalCare+ — Chest X-ray image preprocessing utilities.

Industrial & clinical guarantees:
- Absolute path resolution
- Explicit file validation
- Deterministic preprocessing
- ImageNet-compatible tensor output
- Safe failure modes (no silent errors)

IMPORTANT MEDICAL NOTE:
- Chest X-rays are grayscale
- ImageNet-pretrained CNNs expect 3-channel input
- Grayscale images are replicated → (3, H, W)
"""

import os
import cv2
import numpy as np
import torch

from src.common.constants import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    IMAGE_CHANNELS
)

# --------------------------------------------------
# Image loading & preprocessing
# --------------------------------------------------
def load_xray_image(image_path: str) -> torch.Tensor:
    """
    Loads and preprocesses a chest X-ray image.

    Steps:
    1. Resolve absolute path
    2. Verify file existence
    3. Load image in grayscale
    4. Resize to model input size
    5. Normalize pixel values
    6. Convert to ImageNet-compatible 3-channel tensor

    Args:
        image_path (str): Path to X-ray image file

    Returns:
        torch.Tensor:
            Shape → (C, H, W)
            C = IMAGE_CHANNELS (expected: 3)
    """

    # --------------------------------------------------
    # Resolve absolute path (critical for CLI / APIs)
    # --------------------------------------------------
    image_path = os.path.abspath(image_path)

    # --------------------------------------------------
    # Validate file existence
    # --------------------------------------------------
    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"X-ray image file not found:\n{image_path}"
        )

    # --------------------------------------------------
    # Load image (grayscale)
    # --------------------------------------------------
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(
            f"Failed to read image using OpenCV:\n{image_path}"
        )

    # --------------------------------------------------
    # Resize (deterministic)
    # --------------------------------------------------
    image = cv2.resize(
        image,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------
    # Normalize to [0, 1]
    # --------------------------------------------------
    image = image.astype(np.float32) / 255.0  # (H, W)

    # --------------------------------------------------
    # Convert grayscale → multi-channel
    # --------------------------------------------------
    if IMAGE_CHANNELS == 3:
        # Replicate grayscale channel (ImageNet standard)
        image = np.stack([image, image, image], axis=0)
    else:
        # Fallback (not recommended for pretrained models)
        image = np.expand_dims(image, axis=0)

    # --------------------------------------------------
    # Convert to PyTorch tensor
    # --------------------------------------------------
    image_tensor = torch.from_numpy(image).float()

    return image_tensor


# --------------------------------------------------
# Validation utility
# --------------------------------------------------
def validate_xray_image(image_tensor: torch.Tensor) -> bool:
    """
    Validates a preprocessed X-ray tensor.

    This function is used as a safety gate before inference.

    Args:
        image_tensor (torch.Tensor)

    Returns:
        bool: True if tensor is valid
    """

    if not isinstance(image_tensor, torch.Tensor):
        return False

    if image_tensor.ndim != 3:
        return False

    if image_tensor.shape[0] != IMAGE_CHANNELS:
        return False

    if image_tensor.shape[1] != IMAGE_HEIGHT:
        return False

    if image_tensor.shape[2] != IMAGE_WIDTH:
        return False

    if not torch.isfinite(image_tensor).all():
        return False

    return True
