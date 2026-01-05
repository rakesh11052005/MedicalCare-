"""
brain_mri_dataset.py

Industrial-grade dataset loader for multi-class Brain Tumor MRI
classification in MedicalCare+.

SUPPORTED CLASSES (FIXED & EXPLICIT):
    0 -> Normal
    1 -> Glioma
    2 -> Meningioma
    3 -> Pituitary

MEDICAL & ENGINEERING GUARANTEES:
- CSV-driven (reproducible & auditable)
- Deterministic ordering
- Strict file validation
- Modality-agnostic preprocessing reuse
- Compatible with CrossEntropyLoss
- Safe failure on corrupted / missing data
- Logging aligned with production standards

IMPORTANT:
This dataset performs *classification*, NOT diagnosis.
Final clinical decisions must always be made by a qualified radiologist.
"""

from __future__ import annotations

import csv
import os
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from src.common.image_utils import load_xray_image
from src.common.logging_utils import setup_logger

# --------------------------------------------------
# Logger
# --------------------------------------------------
logger = setup_logger(__name__)


# --------------------------------------------------
# Constants (DO NOT CHANGE ORDER)
# --------------------------------------------------
BRAIN_MRI_CLASS_NAMES = {
    0: "Normal",
    1: "Glioma",
    2: "Meningioma",
    3: "Pituitary",
}

NUM_BRAIN_MRI_CLASSES = len(BRAIN_MRI_CLASS_NAMES)


class BrainMRIDataset(Dataset):
    """
    Multi-class Brain Tumor MRI Dataset.

    Input:
        CSV file with columns:
            - image_path : str (absolute or relative path)
            - label      : int (0–3)

    Output (per sample):
        image : torch.Tensor  -> shape (3, H, W), float32
        label : torch.Tensor  -> scalar, dtype=torch.long

    Training Compatibility:
        - Loss: torch.nn.CrossEntropyLoss
        - Model output: raw logits of shape (B, 4)
    """

    def __init__(self, csv_path: str):
        """
        Initialize the Brain MRI dataset.

        Args:
            csv_path (str): Path to CSV split file
        """
        super().__init__()

        self.csv_path = os.path.abspath(csv_path)
        self.samples: List[Tuple[str, int]] = []

        logger.info("Initializing BrainMRIDataset")
        logger.info(f"CSV path resolved to: {self.csv_path}")

        if not os.path.isfile(self.csv_path):
            raise FileNotFoundError(
                f"Brain MRI CSV split not found:\n{self.csv_path}"
            )

        self._load_csv()
        self._validate_dataset()

        logger.info(
            f"BrainMRIDataset initialized successfully "
            f"with {len(self.samples)} samples"
        )

    # --------------------------------------------------
    # CSV LOADING
    # --------------------------------------------------
    def _load_csv(self) -> None:
        """Loads samples from CSV file (deterministic order)."""

        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)

            required_columns = {"image_path", "label"}
            if not required_columns.issubset(reader.fieldnames or []):
                raise ValueError(
                    "CSV file must contain columns: image_path, label"
                )

            for row_idx, row in enumerate(reader):
                image_path = row["image_path"].strip()
                label_str = row["label"].strip()

                if image_path == "":
                    raise ValueError(
                        f"Empty image_path at CSV row {row_idx + 2}"
                    )

                try:
                    label = int(label_str)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid label at row {row_idx + 2}: {label_str}"
                    ) from exc

                self.samples.append((image_path, label))

    # --------------------------------------------------
    # DATASET VALIDATION
    # --------------------------------------------------
    def _validate_dataset(self) -> None:
        """
        Performs strict dataset validation.

        This function fails fast to prevent silent training corruption.
        """

        if len(self.samples) == 0:
            raise RuntimeError(
                "BrainMRIDataset is empty. "
                "Check CSV file and dataset paths."
            )

        for idx, (image_path, label) in enumerate(self.samples):

            # Validate label range
            if label not in BRAIN_MRI_CLASS_NAMES:
                raise ValueError(
                    f"Invalid Brain MRI label at index {idx}: {label}. "
                    f"Allowed labels: {list(BRAIN_MRI_CLASS_NAMES.keys())}"
                )

            # Validate image existence (absolute resolution)
            resolved_path = os.path.abspath(image_path)
            if not os.path.isfile(resolved_path):
                raise FileNotFoundError(
                    f"Missing MRI image file:\n{resolved_path}"
                )

        logger.info("Brain MRI dataset validation passed")

    # --------------------------------------------------
    # PYTORCH DATASET API
    # --------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Returns one dataset sample.

        Returns:
            image (torch.Tensor): shape (3, H, W), float32
            label (torch.Tensor): scalar, dtype=torch.long
        """

        if idx < 0 or idx >= len(self.samples):
            raise IndexError("Dataset index out of range")

        image_path, label = self.samples[idx]

        # --------------------------------------------------
        # Load & preprocess MRI image
        # NOTE:
        # We intentionally reuse load_xray_image()
        # because:
        #   - MRI is grayscale
        #   - DenseNet expects 3 channels
        #   - Ensures identical preprocessing pipeline
        # --------------------------------------------------
        try:
            image_tensor = load_xray_image(image_path)
        except Exception as exc:
            logger.error(
                f"Failed to load MRI image at index {idx}: {image_path}"
            )
            raise exc

        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor
