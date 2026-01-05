"""
image_utils.py

MedicalCare+ — Medical image preprocessing utilities.

CRITICAL FILE:
This module defines the ONLY approved image preprocessing
path for ALL modalities in MedicalCare+.

Supported modalities:
- Chest X-ray (grayscale)
- Brain MRI (2D slices)

Industrial & clinical guarantees:
- Explicit modality handling (NO guessing)
- Absolute path resolution
- Deterministic preprocessing
- ImageNet-compatible tensor output
- Safe failure modes (fail fast, no silent errors)
"""

import os
import cv2
import numpy as np
import torch

from src.common.constants import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    IMAGE_CHANNELS,
)

# ==================================================
# PUBLIC API — IMAGE LOADING
# ==================================================
def load_medical_image(
    image_path: str,
    modality: str,
) -> torch.Tensor:
    """
    Loads and preprocesses a medical image.

    Args:
        image_path (str): Path to image file
        modality (str):
            - "xray"
            - "brain_mri"

    Returns:
        torch.Tensor:
            Shape → (C, H, W)
    """

    image_path = os.path.abspath(image_path)

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Medical image file not found:\n{image_path}"
        )

    modality = modality.lower()

    if modality == "xray":
        return _load_xray(image_path)

    if modality == "brain_mri":
        return _load_brain_mri(image_path)

    raise ValueError(
        f"Unsupported modality '{modality}'. "
        "Supported: xray, brain_mri"
    )


# ==================================================
# MODALITY-SPECIFIC LOADERS (INTERNAL)
# ==================================================
def _load_xray(image_path: str) -> torch.Tensor:
    """
    Chest X-ray preprocessing (grayscale).
    """

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(
            f"Failed to read X-ray image:\n{image_path}"
        )

    image = cv2.resize(
        image,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    image = image.astype(np.float32) / 255.0

    # Replicate grayscale → 3 channels (ImageNet standard)
    image = np.stack([image, image, image], axis=0)

    return torch.from_numpy(image).float()


def _load_brain_mri(image_path: str) -> torch.Tensor:
    """
    Brain MRI preprocessing (2D slice).

    Notes:
    - MRI intensity ranges vary
    - Normalization is per-image (safe default)
    """

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(
            f"Failed to read Brain MRI image:\n{image_path}"
        )

    image = cv2.resize(
        image,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    image = image.astype(np.float32)

    # Per-image normalization (robust for MRI)
    min_val = image.min()
    max_val = image.max()

    if max_val > min_val:
        image = (image - min_val) / (max_val - min_val)
    else:
        image = np.zeros_like(image)

    # Replicate to 3 channels
    image = np.stack([image, image, image], axis=0)

    return torch.from_numpy(image).float()


# ==================================================
# VALIDATION (MODALITY-AGNOSTIC)
# ==================================================
def validate_medical_image(image_tensor: torch.Tensor) -> bool:
    """
    Validates a preprocessed medical image tensor.

    Used as a hard safety gate before:
    - Training
    - Inference
    - Explainability
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


# ==================================================
# BACKWARD-COMPATIBILITY ALIASES (CRITICAL)
# ==================================================

"""
DO NOT REMOVE.
These keep existing X-ray pipelines working
while migrating to modality-aware loaders.
"""

def load_xray_image(image_path: str) -> torch.Tensor:
    return load_medical_image(image_path, modality="xray")


def validate_xray_image(image_tensor: torch.Tensor) -> bool:
    return validate_medical_image(image_tensor)
