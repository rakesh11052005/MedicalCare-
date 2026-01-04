"""
generate_multidisease_splits.py

Industrial-grade dataset preparation script for MedicalCare+.

Purpose:
- Scan raw chest X-ray folders (multi-disease)
- Generate a unified multi-label CSV
- Create reproducible train / val / test splits
- Serve as the SINGLE source of truth for dataset splits

Medical & Engineering Guarantees:
- No hard-coded assumptions inside training code
- Deterministic splits (seeded)
- Multi-label (NOT multi-class)
- Compatible with selective disease inference
- Auditable & reproducible
"""

import os
import csv
import random
from pathlib import Path
from typing import Dict, List

from src.common.constants import (
    RANDOM_SEED,
    DISEASES
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

RAW_DATA_DIR = Path("data/raw/chestxray")

OUTPUT_DIR = Path("data/splits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Map folder names → disease labels
FOLDER_TO_DISEASE = {
    "NormalLungs": None,                 # No disease
    "PneumoniaLungs": "Pneumonia",
    "TBLungs": "Tuberculosis",
    "COVIDLungs": "COVID-19",
    "OpacityLungs": "Lung Opacity"
}

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

random.seed(RANDOM_SEED)

# --------------------------------------------------
# UTILITIES
# --------------------------------------------------

def _empty_label_row() -> Dict[str, int]:
    """Create zero-initialized multi-disease label row."""
    return {disease: 0 for disease in DISEASES}


def scan_dataset() -> List[Dict]:
    """
    Scan raw folders and build multi-label records.

    Returns:
        List[Dict]: dataset records
    """
    records = []

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Raw dataset directory not found: {RAW_DATA_DIR}"
        )

    for folder_name, disease in FOLDER_TO_DISEASE.items():
        folder_path = RAW_DATA_DIR / folder_name

        if not folder_path.exists():
            raise FileNotFoundError(
                f"Expected folder missing: {folder_path}"
            )

        for img_file in folder_path.iterdir():
            if not img_file.suffix.lower() in IMAGE_EXTENSIONS:
                continue

            labels = _empty_label_row()

            if disease is not None:
                labels[disease] = 1

            record = {
                "image_path": str(img_file).replace("\\", "/"),
                **labels
            }

            records.append(record)

    if not records:
        raise RuntimeError("No images found during dataset scan.")

    return records


def split_dataset(records: List[Dict]):
    """
    Split dataset into train / val / test.
    """
    random.shuffle(records)

    total = len(records)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    return (
        records[:train_end],
        records[train_end:val_end],
        records[val_end:]
    )


def write_csv(filename: str, rows: List[Dict]):
    """
    Write dataset split to CSV.
    """
    path = OUTPUT_DIR / filename

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path"] + DISEASES
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Written {len(rows)} samples → {path}")


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():
    print("\n🩺 MedicalCare+ — Multi-Disease Dataset Generation")
    print("-" * 55)

    records = scan_dataset()

    print(f"📊 Total images discovered: {len(records)}")
    print("🧬 Diseases included:", ", ".join(DISEASES))

    train_set, val_set, test_set = split_dataset(records)

    write_csv("multidisease_train.csv", train_set)
    write_csv("multidisease_val.csv", val_set)
    write_csv("multidisease_test.csv", test_set)

    print("\n✅ Dataset preparation completed successfully")
    print("⚠️ Reminder: This dataset provides FINDINGS, not diagnoses\n")


if __name__ == "__main__":
    main()
