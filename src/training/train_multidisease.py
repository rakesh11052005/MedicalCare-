"""
train_multidisease.py

Industrial-grade training pipeline for MedicalCare+ multi-disease chest X-ray model.

Design guarantees:
- Multi-label (NOT multiclass)
- Windows-safe DataLoader
- Deterministic where possible
- CLI-driven (reproducible experiments)
- Explicit safety checks
- Regulator-friendly logging
"""

import os
import json
import random
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm

from src.datasets.unified_chestxray_dataset import UnifiedChestXrayDataset
from src.models.model_factory import build_multidisease_model
from src.common.constants import (
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
# Reproducibility (Best-effort, medical-safe)
# ==================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================================================
# Class imbalance handling
# ==================================================
def compute_pos_weights(dataset: UnifiedChestXrayDataset) -> torch.Tensor:
    labels = dataset.labels

    if isinstance(labels, torch.Tensor):
        label_matrix = labels.float()
    else:
        label_matrix = torch.stack([lbl.float() for lbl in labels])

    pos_counts = label_matrix.sum(dim=0)
    neg_counts = label_matrix.size(0) - pos_counts

    pos_weights = neg_counts / (pos_counts + 1e-6)
    logger.info(f"Computed positive class weights: {pos_weights.tolist()}")

    return pos_weights


# ==================================================
# Training function
# ==================================================
def train_multidisease_model(
    train_csv_path: str,
    val_csv_path: str,
    save_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
):
    logger.info("Starting multi-disease model training")
    set_seed(RANDOM_SEED)

    os.makedirs(save_dir, exist_ok=True)

    # -------------------------
    # Datasets
    # -------------------------
    train_dataset = UnifiedChestXrayDataset(train_csv_path)
    val_dataset = UnifiedChestXrayDataset(val_csv_path)

    # Windows-safe DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # CRITICAL: Windows-safe
        pin_memory=(device == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    logger.info(f"Training samples   : {len(train_dataset)}")
    logger.info(f"Validation samples : {len(val_dataset)}")

    # -------------------------
    # Model
    # -------------------------
    model = build_multidisease_model()
    model.to(device)

    # -------------------------
    # Loss
    # -------------------------
    pos_weights = compute_pos_weights(train_dataset).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # -------------------------
    # Optimizer
    # -------------------------
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    history = {"train_loss": [], "val_loss": []}

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(epochs):
        logger.info(f"Epoch [{epoch + 1}/{epochs}]")

        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc="Training"):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            # Safety checks (CRITICAL)
            if logits.shape != labels.shape:
                raise RuntimeError(
                    f"Logits shape {logits.shape} "
                    f"does not match labels {labels.shape}"
                )

            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                raise RuntimeError("Loss became NaN or Inf")

            optimizer.zero_grad()
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

    # -------------------------
    # Save model
    # -------------------------
    model_path = os.path.join(save_dir, "model.pt")
    torch.save(model.state_dict(), model_path)

    metadata = {
        "task": "multi-disease",
        "findings": FINDINGS,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "history": history,
    }

    with open(os.path.join(save_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Model saved at: {model_path}")
    logger.info("Multi-disease training completed successfully")


# ==================================================
# CLI
# ==================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_multidisease_model(
        train_csv_path="data/splits/multidisease_train.csv",
        val_csv_path="data/splits/multidisease_val.csv",
        save_dir="models/multidisease/v1.0.0",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )
