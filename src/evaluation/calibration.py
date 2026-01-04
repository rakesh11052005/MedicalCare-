# File: MedicalCare+/src/evaluation/calibration.py

"""
calibration.py

Probability calibration utilities for MedicalCare+.

Purpose:
- Measure how well model confidence aligns with reality
- Detect overconfidence (critical for medical AI)
- Support abstention & uncertainty policies
- Disease-agnostic (single or multi-disease)

IMPORTANT:
This module DOES NOT modify predictions.
It ONLY evaluates calibration quality.
"""

from typing import Union

import numpy as np

from src.common.logging_utils import setup_logger

# --------------------------------------------------
# Logger
# --------------------------------------------------
logger = setup_logger(__name__)


class ProbabilityCalibrator:
    """
    Probability calibration evaluator.

    Uses Expected Calibration Error (ECE),
    a standard metric in medical and safety-critical ML.
    """

    def __init__(self, n_bins: int = 10):
        """
        Initializes the calibrator.

        Args:
            n_bins (int): Number of bins for calibration histogram
        """
        if n_bins <= 1:
            raise ValueError("n_bins must be greater than 1")

        self.n_bins = n_bins
        logger.info(
            f"ProbabilityCalibrator initialized with {n_bins} bins"
        )

    def expected_calibration_error(
        self,
        y_true: Union[np.ndarray, list],
        y_prob: Union[np.ndarray, list]
    ) -> float:
        """
        Computes Expected Calibration Error (ECE).

        Args:
            y_true:
                Ground truth labels
                - shape (N,) for single-label
                - shape (N,) per finding for multi-label

            y_prob:
                Predicted probabilities
                - shape (N,) for single-label
                - shape (N,) per finding for multi-label

        Returns:
            float: Expected Calibration Error (lower is better)
        """

        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob).astype(float)

        if y_true.shape != y_prob.shape:
            raise ValueError(
                "y_true and y_prob must have the same shape"
            )

        # --------------------------------------------------
        # Bin probabilities
        # --------------------------------------------------
        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        ece = 0.0
        total_samples = len(y_true)

        for i in range(self.n_bins):
            bin_lower = bin_edges[i]
            bin_upper = bin_edges[i + 1]

            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

            if not np.any(in_bin):
                continue

            bin_prob = y_prob[in_bin]
            bin_true = y_true[in_bin]

            # Average confidence and accuracy in bin
            avg_confidence = np.mean(bin_prob)
            avg_accuracy = np.mean(bin_true)

            bin_weight = np.sum(in_bin) / total_samples
            ece += bin_weight * abs(avg_confidence - avg_accuracy)

        return float(ece)
