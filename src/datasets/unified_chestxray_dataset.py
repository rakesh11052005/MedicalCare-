"""
unified_chestxray_dataset.py

Unified dataset for multi-disease chest X-ray learning.

Design principles:
- CSV-driven (no directory assumptions)
- Multi-label output (not mutually exclusive)
- Supports arbitrary number of findings
- Safe, deterministic, auditable
- Industry-standard medical ML design
"""

import os
import pandas as pd
import torch
from typing import List

from src.datasets.base_dataset import BaseXRayDataset
from src.common.image_utils import load_xray_image
from src.common.constants import FINDINGS
from src.common.logging_utils import setup_logger

# --------------------------------------------------
# Logger
# --------------------------------------------------
logger = setup_logger(__name__)


class UnifiedChestXrayDataset(BaseXRayDataset):
    """
    Unified Chest X-ray Dataset (Multi-Disease).

    Expected CSV format:
    --------------------------------------------------
    image_path,Pneumonia,Tuberculosis,COVID-19,Lung Opacity
    data/raw/.../img1.png,1,0,0,1
    data/raw/.../img2.png,0,0,1,1
    --------------------------------------------------
    """

    def __init__(self, csv_path: str):
        """
        Initializes the unified dataset.

        Args:
            csv_path (str): Path to CSV split file
        """

        logger.info("Initializing UnifiedChestXrayDataset")

        csv_path = os.path.abspath(csv_path)

        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"Dataset CSV not found:\n{csv_path}"
            )

        df = pd.read_csv(csv_path)

        # --------------------------------------------------
        # Validate CSV schema
        # --------------------------------------------------
        required_columns = ["image_path"] + FINDINGS
        missing = set(required_columns) - set(df.columns)

        if missing:
            raise ValueError(
                f"CSV missing required columns: {missing}"
            )

        image_paths: List[str] = []
        labels: List[List[int]] = []

        for _, row in df.iterrows():

            img_path = os.path.abspath(row["image_path"])

            if not os.path.isfile(img_path):
                logger.warning(
                    f"Skipping missing image: {img_path}"
                )
                continue

            label_vector = []

            for disease in FINDINGS:
                value = row[disease]

                if value not in (0, 1):
                    raise ValueError(
                        f"Invalid label value for {disease}: {value}"
                    )

                label_vector.append(int(value))

            image_paths.append(img_path)
            labels.append(label_vector)

        if len(image_paths) == 0:
            raise RuntimeError("No valid samples found in dataset")

        logger.info(
            f"Loaded {len(image_paths)} samples "
            f"with findings: {FINDINGS}"
        )

        super().__init__(
            image_paths=image_paths,
            labels=torch.tensor(labels, dtype=torch.float32),
            image_loader=load_xray_image
        )
