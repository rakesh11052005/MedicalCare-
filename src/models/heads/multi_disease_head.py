"""
multi_disease_head.py

Multi-disease classification head for MedicalCare+.

Medical & industrial guarantees:
- Multi-label output (NOT softmax)
- Produces raw logits (sigmoid applied ONLY in inference)
- Disease-agnostic (output size configurable)
- Batch-size agnostic (NO BatchNorm)
- Compatible with uncertainty & abstention logic
- Safe for Grad-CAM and regulatory review

IMPORTANT:
This head predicts FINDINGS, not diagnoses.
"""

import torch
import torch.nn as nn

from src.common.logging_utils import setup_logger

# --------------------------------------------------
# Logger
# --------------------------------------------------
logger = setup_logger(__name__)


class MultiDiseaseHead(nn.Module):
    """
    Classification head for multi-disease chest X-ray analysis.

    Input:
        Feature maps from backbone → (B, C, H, W)

    Output:
        Raw logits per finding → (B, num_diseases)

    Training:
        Use BCEWithLogitsLoss

    Inference:
        Apply sigmoid externally
    """

    def __init__(
        self,
        in_features: int,
        num_diseases: int,
        hidden_ratio: float = 0.5,
        dropout: float = 0.5,
    ):
        """
        Args:
            in_features (int): Channel dimension from backbone
            num_diseases (int): Number of disease findings
            hidden_ratio (float): Reduction ratio for hidden layer
            dropout (float): Dropout probability
        """
        super().__init__()

        if num_diseases <= 0:
            raise ValueError("num_diseases must be > 0")

        hidden_dim = int(in_features * hidden_ratio)

        logger.info("Initializing MultiDiseaseHead (BatchNorm-free)")
        logger.info(f"Input features   : {in_features}")
        logger.info(f"Hidden features  : {hidden_dim}")
        logger.info(f"Output diseases  : {num_diseases}")
        logger.info(f"Dropout          : {dropout}")

        # Global pooling preserves spatial explainability
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification MLP
        # NOTE: No BatchNorm → safe for batch_size = 1
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_diseases),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features (Tensor): Backbone feature maps
                               Shape → (B, C, H, W)

        Returns:
            logits (Tensor): Raw logits per disease
                             Shape → (B, num_diseases)
        """

        if features.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor (B, C, H, W), got shape {features.shape}"
            )

        x = self.global_pool(features)
        logits = self.classifier(x)

        return logits
