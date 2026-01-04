# File: MedicalCare+/src/datasets/base_dataset.py

"""
base_dataset.py

Base dataset abstraction for MedicalCare+.

Industrial & clinical guarantees:
- Centralized image loading logic
- Dataset-agnostic (single-disease or multi-disease)
- Deterministic indexing
- Safe failure modes
- Compatible with training, evaluation, inference, and explainability
- Extensible to future modalities (CT, MRI, DICOM)

This class MUST NOT contain disease-specific logic.
"""

from typing import Callable, List, Union, Optional

import torch
from torch.utils.data import Dataset

from src.common.image_utils import load_xray_image, validate_xray_image
from src.common.logging_utils import setup_logger

# --------------------------------------------------
# Logger
# --------------------------------------------------
logger = setup_logger(__name__)


class BaseXRayDataset(Dataset):
    """
    Base PyTorch Dataset for chest X-ray images.

    Child datasets must only provide:
    - image_paths: List[str]
    - labels: List[int] OR List[Tensor]
    - optional image_loader override

    This class handles:
    - Image loading
    - Image validation
    - Tensor normalization
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[Union[int, torch.Tensor]],
        image_loader: Optional[Callable[[str], torch.Tensor]] = None,
    ):
        """
        Initializes the dataset.

        Args:
            image_paths (List[str]):
                Absolute or relative paths to X-ray images

            labels (List[int] | List[torch.Tensor]):
                - int → single-label classification
                - Tensor → multi-label (multi-disease)

            image_loader (Callable, optional):
                Custom image loading function.
                Defaults to load_xray_image.
        """

        if len(image_paths) != len(labels):
            raise ValueError(
                "image_paths and labels must have the same length"
            )

        if len(image_paths) == 0:
            raise ValueError(
                "Dataset initialized with zero samples"
            )

        self.image_paths = image_paths
        self.labels = labels

        # Allow modality-specific loader injection (future-proof)
        self.image_loader = image_loader or load_xray_image

        logger.info(
            f"BaseXRayDataset initialized with {len(self.image_paths)} samples"
        )

    def __len__(self) -> int:
        """
        Returns dataset size.
        """
        return len(self.image_paths)

    def __getitem__(self, index: int):
        """
        Retrieves one sample.

        Returns:
            image (torch.Tensor):
                Shape → (C, H, W)

            label (torch.Tensor):
                Shape:
                - ()     for single-label
                - (N,)   for multi-label
        """

        image_path = self.image_paths[index]
        label = self.labels[index]

        # --------------------------------------------------
        # Load image
        # --------------------------------------------------
        image = self.image_loader(image_path)

        if not validate_xray_image(image):
            raise RuntimeError(
                f"Invalid image tensor after preprocessing:\n{image_path}"
            )

        # --------------------------------------------------
        # Normalize label type
        # --------------------------------------------------
        if isinstance(label, torch.Tensor):
            label_tensor = label.float()
        else:
            label_tensor = torch.tensor(label, dtype=torch.long)

        return image, label_tensor
