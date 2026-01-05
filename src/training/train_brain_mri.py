"""
train_brain_mri.py

Industrial-grade training pipeline for multi-class Brain Tumor MRI
classification in MedicalCare+.

TASK TYPE:
    - Multi-class classification (Normal, Glioma, Meningioma, Pituitary)

DESIGN PRINCIPLES:
- Deterministic & reproducible training
- Explicit separation of data, model, loss, metrics
- No inference or threshold logic
- Strict logging for auditability
- Medical safety–aligned defaults

IMPORTANT:
This script trains a decision-support model.
It does NOT produce a diagnostic system.
"""

from __future__ import annotations

import os
import json
import random
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.common.constants import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    RANDOM_SEED,
)
from src.common.logging_utils import setup_logger
from src.datasets.brain_mri_dataset import BrainMRIDataset
from src.models.model_factory import build_brain_tumor_model

# --------------------------------------------------
# Logger
# --------------------------------------------------
logger = setup_logger(__name__)


# ==================================================
# REPRODUCIBILITY
# ==================================================
def _set_seed(seed: int) -> None:
    """
    Enforces deterministic behavior across libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to {seed}")


# ==================================================
# TRAINING LOOP
# ==================================================
def train_brain_mri_model(
    train_csv: str,
    val_csv: str,
    output_dir: str,
    device: str = "cpu",
) -> Dict:
    """
    Trains a multi-class Brain Tumor MRI model.

    Args:
        train_csv (str): Path to training CSV split
        val_csv (str): Path to validation CSV split
        output_dir (str): Directory to save model & metrics
        device (str): 'cpu' or 'cuda'

    Returns:
        Dict: Training summary metrics
    """

    logger.info("Starting Brain MRI model training")
    logger.info(f"Device: {device}")

    _set_seed(RANDOM_SEED)

    # --------------------------------------------------
    # Resolve paths
    # --------------------------------------------------
    train_csv = os.path.abspath(train_csv)
    val_csv = os.path.abspath(val_csv)
    output_dir = os.path.abspath(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------
    # Load datasets
    # --------------------------------------------------
    train_dataset = BrainMRIDataset(train_csv)
    val_dataset = BrainMRIDataset(val_csv)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    logger.info(
        f"Training samples: {len(train_dataset)} | "
        f"Validation samples: {len(val_dataset)}"
    )

    # --------------------------------------------------
    # Build model
    # --------------------------------------------------
    model = build_brain_tumor_model(
        pretrained=True,
        freeze_backbone=True,
        num_classes=4,
    )

    model.to(device)
    model.train()

    logger.info("Brain tumor model instantiated successfully")

    # --------------------------------------------------
    # Loss & Optimizer
    # --------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------
    # Training state
    # --------------------------------------------------
    best_val_accuracy = 0.0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    # --------------------------------------------------
    # Epoch loop
    # --------------------------------------------------
    for epoch in range(1, EPOCHS + 1):
        logger.info(f"Epoch [{epoch}/{EPOCHS}]")

        # -----------------------------
        # Training phase
        # -----------------------------
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_dataset)
        history["train_loss"].append(train_loss)

        logger.info(f"Training loss: {train_loss:.6f}")

        # -----------------------------
        # Validation phase
        # -----------------------------
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)

                val_loss += loss.item() * images.size(0)

                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_dataset)
        val_accuracy = correct / total

        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        logger.info(
            f"Validation loss: {val_loss:.6f} | "
            f"Validation accuracy: {val_accuracy:.4f}"
        )

        # -----------------------------
        # Checkpointing (BEST MODEL)
        # -----------------------------
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            model_path = os.path.join(output_dir, "model.pt")
            torch.save(model.state_dict(), model_path)

            logger.info(
                f"New best model saved | "
                f"val_accuracy={best_val_accuracy:.4f}"
            )

    # --------------------------------------------------
    # Save metrics
    # --------------------------------------------------
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "best_val_accuracy": best_val_accuracy,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "history": history,
            },
            f,
            indent=2,
        )

    logger.info("Training completed successfully")
    logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")

    return {
        "best_val_accuracy": best_val_accuracy,
        "metrics_path": metrics_path,
        "model_path": os.path.join(output_dir, "model.pt"),
    }


# ==================================================
# CLI ENTRY POINT
# ==================================================
if __name__ == "__main__":

    TRAIN_CSV = "data/splits/brain_mri_train.csv"
    VAL_CSV = "data/splits/brain_mri_val.csv"
    OUTPUT_DIR = "models/brain_tumor/v1.0.0"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_brain_mri_model(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        output_dir=OUTPUT_DIR,
        device=device,
    )
