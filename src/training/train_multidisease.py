"""
train_multidisease.py

Training pipeline for MedicalCare+ multi-disease chest X-ray model.

Industrial / clinical design:
- Multi-label (NOT multiclass) learning
- CSV-driven dataset (auditable & reproducible)
- Explicit class-imbalance handling
- Deterministic training where possible
- No decision thresholds in training
- Safe, explainable, regulator-friendly
"""

import os
import json
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm

from src.datasets.unified_chestxray_dataset import UnifiedChestXrayDataset
from src.models.model_factory import build_multidisease_model
from src.common.constants import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    RANDOM_SEED,
    FINDINGS
)
from src.common.logging_utils import setup_logger

# --------------------------------------------------
# Logger
# --------------------------------------------------
logger = setup_logger(__name__)


# ==================================================
# Reproducibility (Best-effort)
# ==================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==================================================
# Class Weights (for imbalance)
# ==================================================
def compute_pos_weights(dataset: UnifiedChestXrayDataset) -> torch.Tensor:
    """
    Computes positive class weights per disease.

    Supports:
    - dataset.labels as Tensor (N, D)
    - dataset.labels as List[Tensor] (length N)

    Returns:
        Tensor of shape (num_diseases,)
    """

    labels = dataset.labels

    # --------------------------------------------------
    # Normalize label representation
    # --------------------------------------------------
    if isinstance(labels, torch.Tensor):
        # Expected: (N, D)
        if labels.dim() != 2:
            raise ValueError(
                f"Expected labels tensor of shape (N, D), got {labels.shape}"
            )
        label_matrix = labels.float()
    else:
        # List[Tensor] → stack
        label_matrix = torch.stack(
            [lbl.float() for lbl in labels]
        )

    # --------------------------------------------------
    # Compute class frequencies
    # --------------------------------------------------
    pos_counts = label_matrix.sum(dim=0)               # (D,)
    neg_counts = label_matrix.size(0) - pos_counts     # (D,)

    # Avoid division by zero
    pos_weights = neg_counts / (pos_counts + 1e-6)

    logger.info(f"Computed positive class weights: {pos_weights.tolist()}")

    return pos_weights


# ==================================================
# Training Function
# ==================================================
def train_multidisease_model(
    train_csv_path: str,
    val_csv_path: str,
    save_dir: str,
    device: str = "cpu"
):
    """
    Trains a multi-disease chest X-ray model.

    Args:
        train_csv_path (str): Training CSV
        val_csv_path (str): Validation CSV
        save_dir (str): Directory to save model & artifacts
        device (str): cpu / cuda
    """

    logger.info("Starting multi-disease model training")

    # --------------------------------------------------
    # Setup
    # --------------------------------------------------
    set_seed(RANDOM_SEED)

    train_csv_path = os.path.abspath(train_csv_path)
    val_csv_path = os.path.abspath(val_csv_path)
    save_dir = os.path.abspath(save_dir)

    os.makedirs(save_dir, exist_ok=True)

    # --------------------------------------------------
    # Datasets
    # --------------------------------------------------
    train_dataset = UnifiedChestXrayDataset(train_csv_path)
    val_dataset = UnifiedChestXrayDataset(val_csv_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=(device == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=(device == "cuda")
    )

    logger.info(f"Training samples   : {len(train_dataset)}")
    logger.info(f"Validation samples : {len(val_dataset)}")

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = build_multidisease_model()
    model.to(device)

    # --------------------------------------------------
    # Loss (Multi-label)
    # --------------------------------------------------
    pos_weights = compute_pos_weights(train_dataset).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weights
    )

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------
    # Training Loop
    # --------------------------------------------------
    history = {
        "train_loss": [],
        "val_loss": []
    }

    for epoch in range(EPOCHS):
        logger.info(f"Epoch [{epoch + 1}/{EPOCHS}]")

        # -------------------------
        # Train
        # -------------------------
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc="Training"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        logger.info(f"Train Loss: {avg_train_loss:.4f}")

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        history["val_loss"].append(avg_val_loss)

        logger.info(f"Validation Loss: {avg_val_loss:.4f}")

    # --------------------------------------------------
    # Save Model
    # --------------------------------------------------
    model_path = os.path.join(save_dir, "model.pt")
    torch.save(model.state_dict(), model_path)

    # --------------------------------------------------
    # Save Training Metadata
    # --------------------------------------------------
    metadata = {
        "model_type": "multi-disease",
        "task": "multi-label classification",
        "findings": FINDINGS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "history": history
    }

    with open(os.path.join(save_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Model saved at: {model_path}")
    logger.info("Multi-disease training completed successfully")


# --------------------------------------------------
# CLI ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":

    TRAIN_CSV = "data/splits/multidisease_train.csv"
    VAL_CSV = "data/splits/multidisease_val.csv"
    SAVE_DIR = "models/multidisease/v1.0.0"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_multidisease_model(
        train_csv_path=TRAIN_CSV,
        val_csv_path=VAL_CSV,
        save_dir=SAVE_DIR,
        device=device
    )
