"""
generate_brain_mri_splits.py

Industrial-grade dataset split generator for multi-class Brain Tumor MRI
classification in MedicalCare+.

SUPPORTED CLASSES (FIXED CONTRACT):
    0 -> Normal
    1 -> Glioma
    2 -> Meningioma
    3 -> Pituitary

DESIGN PRINCIPLES:
- Deterministic & reproducible splits
- Explicit class mapping (no auto-inference)
- Strict file validation
- CSV-based splits for auditability
- Safe failure on missing or corrupted data
- Compatible with regulated medical ML workflows

IMPORTANT:
This script ONLY prepares dataset splits.
It does NOT perform training, inference, or labeling.
"""

from __future__ import annotations

import os
import csv
import random
from typing import Dict, List, Tuple

# --------------------------------------------------
# Configuration
# --------------------------------------------------
RAW_DATA_DIR = "data/raw/brain_mri"
OUTPUT_DIR = "data/splits"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# --------------------------------------------------
# Fixed class mapping (DO NOT CHANGE ORDER)
# --------------------------------------------------
CLASS_MAP: Dict[str, int] = {
    "NormalBrain": 0,
    "GliomaBrain": 1,
    "MeningiomaBrain": 2,
    "PituitaryBrain": 3,
}

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg")


# ==================================================
# VALIDATION
# ==================================================
def _validate_ratios() -> None:
    total = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total}"
        )


def _validate_raw_structure() -> None:
    if not os.path.isdir(RAW_DATA_DIR):
        raise FileNotFoundError(
            f"Brain MRI raw data directory not found:\n{RAW_DATA_DIR}"
        )

    for class_name in CLASS_MAP.keys():
        class_dir = os.path.join(RAW_DATA_DIR, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"Missing class directory:\n{class_dir}"
            )


# ==================================================
# DATA COLLECTION
# ==================================================
def collect_samples() -> List[Tuple[str, int]]:
    """
    Collects all MRI image paths and labels.

    Returns:
        List of (image_path, label)
    """

    samples: List[Tuple[str, int]] = []

    for class_name, label in CLASS_MAP.items():
        class_dir = os.path.join(RAW_DATA_DIR, class_name)

        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith(SUPPORTED_EXTENSIONS):
                continue

            image_path = os.path.join(class_dir, fname)
            abs_path = os.path.abspath(image_path)

            if not os.path.isfile(abs_path):
                raise FileNotFoundError(
                    f"Expected image file missing:\n{abs_path}"
                )

            samples.append((abs_path, label))

    if len(samples) == 0:
        raise RuntimeError(
            "No MRI images found. "
            "Check dataset structure and file extensions."
        )

    return samples


# ==================================================
# SPLIT & SAVE
# ==================================================
def split_and_save(samples: List[Tuple[str, int]]) -> None:
    """
    Splits dataset into train / val / test and saves CSV files.
    """

    random.seed(RANDOM_SEED)
    random.shuffle(samples)

    n_total = len(samples)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    train_samples = samples[:n_train]
    val_samples = samples[n_train : n_train + n_val]
    test_samples = samples[n_train + n_val :]

    splits = {
        "brain_mri_train.csv": train_samples,
        "brain_mri_val.csv": val_samples,
        "brain_mri_test.csv": test_samples,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for filename, rows in splits.items():
        output_path = os.path.join(OUTPUT_DIR, filename)

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "label"])
            writer.writerows(rows)

        print(
            f"✅ {filename} written | "
            f"samples={len(rows)}"
        )

    print("\nDataset split summary")
    print("-" * 40)
    print(f"Total samples : {n_total}")
    print(f"Train samples : {len(train_samples)}")
    print(f"Val samples   : {len(val_samples)}")
    print(f"Test samples  : {len(test_samples)}")


# ==================================================
# ENTRY POINT
# ==================================================
def main() -> None:
    print("Starting Brain MRI dataset split generation")
    print("-" * 50)

    _validate_ratios()
    _validate_raw_structure()

    samples = collect_samples()
    split_and_save(samples)

    print("\nBrain MRI dataset splits generated successfully")


if __name__ == "__main__":
    main()
