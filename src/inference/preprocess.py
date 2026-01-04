# File: MedicalCare+/src/inference/preprocess.py

"""
preprocess.py

Unified preprocessing pipeline for MedicalCare+ inference.

Design goals:
- Single, authoritative preprocessing path for inference
- Consistent behavior across CLI, API, explainability
- Strict validation and deterministic output
- No silent corrections (fail fast, log clearly)

Medical guarantees:
- No data augmentation at inference
- No normalization mismatch with training
- ImageNet-compatible channel handling
"""

import os
import torch

from src.common.logging_utils import setup_logger
from src.common.image_utils import (
    load_xray_image,
    validate_xray_image
)
from src.common.constants import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    IMAGE_CHANNELS
)

# --------------------------------------------------
# Logger
# --------------------------------------------------
logger = setup_logger(__name__)


def preprocess_for_inference(
    image_path: str,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Preprocesses a chest X-ray image for model inference.

    This function is the ONLY preprocessing entry point
    that inference code should use.

    Args:
        image_path (str): Path to chest X-ray image
        device (str): cpu or cuda

    Returns:
        torch.Tensor:
            Preprocessed image tensor
            Shape -> (1, C, H, W)
    """

    logger.info("Starting inference preprocessing")

    # --------------------------------------------------
    # Resolve and validate path
    # --------------------------------------------------
    image_path = os.path.abspath(image_path)
    logger.info(f"Resolved image path: {image_path}")

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"X-ray image file does not exist:\n{image_path}"
        )

    # --------------------------------------------------
    # Load image using shared utility
    # --------------------------------------------------
    image_tensor = load_xray_image(image_path)

    # --------------------------------------------------
    # Validate tensor integrity
    # --------------------------------------------------
    if not validate_xray_image(image_tensor):
        raise ValueError(
            "Preprocessed X-ray image failed validation checks.\n"
            f"Expected shape: ({IMAGE_CHANNELS}, "
            f"{IMAGE_HEIGHT}, {IMAGE_WIDTH})"
        )

    # --------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------
    image_tensor = image_tensor.unsqueeze(0)

    # --------------------------------------------------
    # Move to device
    # --------------------------------------------------
    image_tensor = image_tensor.to(device)

    logger.info(
        "Inference preprocessing completed successfully "
        f"(shape={tuple(image_tensor.shape)}, device={device})"
    )

    return image_tensor
