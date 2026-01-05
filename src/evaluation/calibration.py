"""
calibration.py

Probability calibration utilities for MedicalCare+.

Purpose:
- Measure how well model confidence aligns with reality
- Detect overconfidence (critical for medical AI)
- Support abstention & uncertainty policies
- Explicitly supports:
    - Binary classification
    - Multi-class classification
    - Multi-label classification

IMPORTANT:
This module DOES NOT modify predictions.
It ONLY evaluates calibration quality.
"""

from typing import Union, Literal

import numpy as np

from src.common.logging_utils import setup_logger

# --------------------------------------------------
# Logger
# --------------------------------------------------
logger = setup_logger(__name__)


class ProbabilityCalibrator:
    """
    Probability calibration evaluator using Expected Calibration Error (ECE).

    This implementation is:
    - Task-aware (binary / multi-class / multi-label)
    - Conservative (medical-safety aligned)
    - Deterministic & auditable
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

    # ==================================================
    # PUBLIC API
    # ==================================================
    def expected_calibration_error(
        self,
        y_true: Union[np.ndarray, list],
        y_prob: Union[np.ndarray, list],
        task_type: Literal[
            "binary",
            "multi_class",
            "multi_label"
        ] = "binary",
        y_pred: Union[np.ndarray, list, None] = None,
    ) -> float:
        """
        Computes Expected Calibration Error (ECE).

        Args:
            y_true:
                Ground truth labels
                - binary      : shape (N,)
                - multi_class : shape (N,)
                - multi_label : shape (N, D)

            y_prob:
                Confidence scores
                - binary      : shape (N,)
                - multi_class : shape (N,) → max softmax probability
                - multi_label : shape (N, D)

            task_type:
                One of ["binary", "multi_class", "multi_label"]

            y_pred:
                Required for multi_class
                - shape (N,)
                Predicted class indices

        Returns:
            float: Expected Calibration Error (lower is better)
        """

        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        if task_type == "multi_class":
            if y_pred is None:
                raise ValueError(
                    "y_pred must be provided for multi_class calibration"
                )
            y_pred = np.asarray(y_pred)

        # --------------------------------------------------
        # Dispatch by task type
        # --------------------------------------------------
        if task_type == "binary":
            return self._ece_binary(y_true, y_prob)

        if task_type == "multi_class":
            return self._ece_multiclass(y_true, y_pred, y_prob)

        if task_type == "multi_label":
            return self._ece_multilabel(y_true, y_prob)

        raise ValueError(f"Unsupported task_type: {task_type}")

    # ==================================================
    # INTERNAL IMPLEMENTATIONS
    # ==================================================
    def _ece_binary(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> float:
        """
        Binary ECE (Normal vs Disease).
        """

        y_true = y_true.astype(int)
        y_prob = y_prob.astype(float)

        if y_true.shape != y_prob.shape:
            raise ValueError("Shape mismatch in binary calibration")

        return self._compute_ece(
            confidences=y_prob,
            correctness=y_true
        )

    def _ece_multiclass(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidences: np.ndarray,
    ) -> float:
        """
        Multi-class ECE using max softmax confidence.

        correctness = 1 if predicted class == true class else 0
        """

        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        confidences = confidences.astype(float)

        if not (
            y_true.shape == y_pred.shape == confidences.shape
        ):
            raise ValueError(
                "Shape mismatch in multi-class calibration inputs"
            )

        correctness = (y_pred == y_true).astype(int)

        return self._compute_ece(
            confidences=confidences,
            correctness=correctness
        )

    def _ece_multilabel(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> float:
        """
        Multi-label ECE.

        ECE is computed independently per label
        and averaged (medical-conservative approach).
        """

        if y_true.shape != y_prob.shape:
            raise ValueError(
                "Shape mismatch in multi-label calibration"
            )

        num_labels = y_true.shape[1]
        eces = []

        for i in range(num_labels):
            ece_i = self._compute_ece(
                confidences=y_prob[:, i],
                correctness=y_true[:, i]
            )
            eces.append(ece_i)

        return float(np.mean(eces))

    # ==================================================
    # CORE ECE COMPUTATION
    # ==================================================
    def _compute_ece(
        self,
        confidences: np.ndarray,
        correctness: np.ndarray,
    ) -> float:
        """
        Core Expected Calibration Error computation.

        Args:
            confidences (np.ndarray): Model confidence scores in [0,1]
            correctness (np.ndarray): 0/1 correctness indicator

        Returns:
            float: ECE
        """

        confidences = confidences.astype(float)
        correctness = correctness.astype(int)

        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        ece = 0.0
        total_samples = len(confidences)

        for i in range(self.n_bins):
            bin_lower = bin_edges[i]
            bin_upper = bin_edges[i + 1]

            in_bin = (
                (confidences >= bin_lower) &
                (confidences < bin_upper)
            )

            if not np.any(in_bin):
                continue

            bin_confidence = np.mean(confidences[in_bin])
            bin_accuracy = np.mean(correctness[in_bin])

            bin_weight = np.sum(in_bin) / total_samples
            ece += bin_weight * abs(bin_confidence - bin_accuracy)

        return float(ece)
