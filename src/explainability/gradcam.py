"""
gradcam.py

Industrial-grade Grad-CAM implementation for MedicalCare+.

Design guarantees:
- Supports single-disease, multi-disease, and multi-class models
- Uses raw logits (NO softmax / sigmoid)
- Read-only (does not alter model behavior)
- Deterministic and auditable
- Compatible with DenseNet / ResNet backbones
- Safe default behavior for clinical explainability
"""

from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np

from src.common.logging_utils import setup_logger

# --------------------------------------------------
# Logger
# --------------------------------------------------
logger = setup_logger(__name__)


class GradCAM:
    """
    Grad-CAM engine for CNN-based medical imaging models.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module
    ):
        """
        Initialize Grad-CAM.

        Args:
            model (torch.nn.Module): Trained model (eval mode)
            target_layer (torch.nn.Module): Last convolutional layer
        """

        self.model = model
        self.target_layer = target_layer

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        self._register_hooks()

        logger.info("Grad-CAM initialized successfully")

    # --------------------------------------------------
    # Hook registration
    # --------------------------------------------------
    def _register_hooks(self):
        """
        Register forward & backward hooks on target layer.
        """

        def forward_hook(module, inputs, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

        logger.info("Grad-CAM hooks registered on target layer")

    # --------------------------------------------------
    # Heatmap generation
    # --------------------------------------------------
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_index: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor (torch.Tensor):
                Shape (1, C, H, W)

            target_index (Optional[int]):
                - Explicit index to explain (recommended for audits)
                - If None:
                    • Single-disease  → index 0
                    • Multi-class     → argmax(logits)
                    • Multi-label     → argmax(logits)

        Returns:
            np.ndarray: Normalized heatmap (H, W)
        """

        # --------------------------------------------------
        # Input validation
        # --------------------------------------------------
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError(
                "Grad-CAM expects input tensor of shape (1, C, H, W)"
            )

        logger.info("Generating Grad-CAM heatmap")

        self.model.eval()

        # Enable gradients
        input_tensor = input_tensor.requires_grad_(True)
        self.model.zero_grad(set_to_none=True)

        # --------------------------------------------------
        # Forward pass
        # --------------------------------------------------
        logits = self.model(input_tensor)

        if logits.ndim != 2:
            raise RuntimeError(
                "Model output must be 2D (B, num_outputs)"
            )

        num_outputs = logits.shape[1]

        # --------------------------------------------------
        # Determine target index safely
        # --------------------------------------------------
        if target_index is None:
            target_index = int(torch.argmax(logits, dim=1).item())
            logger.info(
                f"No target_index provided → using argmax={target_index}"
            )

        if target_index < 0 or target_index >= num_outputs:
            raise IndexError(
                f"target_index {target_index} out of range "
                f"(num_outputs={num_outputs})"
            )

        # --------------------------------------------------
        # Select scalar score (CRITICAL)
        # --------------------------------------------------
        score = logits[:, target_index].sum()

        # --------------------------------------------------
        # Backward pass
        # --------------------------------------------------
        score.backward(retain_graph=False)

        # --------------------------------------------------
        # Safety checks
        # --------------------------------------------------
        if self._activations is None:
            raise RuntimeError(
                "Grad-CAM forward hook did not capture activations"
            )

        if self._gradients is None:
            raise RuntimeError(
                "Grad-CAM backward hook did not capture gradients"
            )

        # --------------------------------------------------
        # Compute channel weights
        # --------------------------------------------------
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)

        # --------------------------------------------------
        # Weighted combination
        # --------------------------------------------------
        cam = (weights * self._activations).sum(dim=1)

        cam = F.relu(cam)  # Grad-CAM standard

        # --------------------------------------------------
        # Normalize heatmap
        # --------------------------------------------------
        cam = cam.squeeze(0)

        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        heatmap = cam.detach().cpu().numpy()

        logger.info(
            f"Grad-CAM heatmap generated (target_index={target_index})"
        )

        return heatmap
