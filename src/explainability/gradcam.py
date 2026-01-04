"""
gradcam.py

Industrial-grade Grad-CAM implementation for MedicalCare+.

Design guarantees:
- Multi-label safe (one disease at a time)
- Uses raw logits (NO softmax / sigmoid)
- Read-only (does not alter model behavior)
- Deterministic and auditable
- Compatible with DenseNet / ResNet backbones
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
        target_index: int
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a specific disease.

        Args:
            input_tensor (torch.Tensor): Shape (1, C, H, W)
            target_index (int): Disease index to explain

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

        logger.info(
            f"Generating Grad-CAM for target_index={target_index}"
        )

        # Ensure model is in eval mode
        self.model.eval()

        # Ensure gradients are enabled
        input_tensor = input_tensor.requires_grad_(True)

        self.model.zero_grad(set_to_none=True)

        # --------------------------------------------------
        # Forward pass
        # --------------------------------------------------
        logits = self.model(input_tensor)

        if logits.ndim != 2:
            raise RuntimeError(
                "Model output must be 2D (B, num_findings)"
            )

        if target_index < 0 or target_index >= logits.shape[1]:
            raise IndexError(
                f"target_index {target_index} out of range"
            )

        # --------------------------------------------------
        # Select ONE disease logit (CRITICAL)
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

        # Apply ReLU (Grad-CAM standard)
        cam = F.relu(cam)

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

        logger.info("Grad-CAM heatmap generated successfully")

        return heatmap
