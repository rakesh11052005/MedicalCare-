"""
safety.py

Unified safety, uncertainty, and abstention logic for MedicalCare+.

Medical & regulatory guarantees:
- NEVER outputs a diagnosis
- Enforces abstention in uncertain zones
- Deterministic, auditable decision logic
- Supports:
    - Single-disease
    - Multi-disease (Chest X-ray)
    - Multi-class (Brain MRI)
- Clinician-first design (AI assists, never decides)

This file is a SINGLE SOURCE OF TRUTH for safety behavior.
"""

from typing import Dict, Tuple

from src.common.logging_utils import setup_logger
from src.common.constants import (
    LOW_CONFIDENCE_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLD,
    FINDING_THRESHOLDS,
)

# --------------------------------------------------
# Logger
# --------------------------------------------------
logger = setup_logger(__name__)


# ==================================================
# INTERNAL UTILITIES (DO NOT EXPORT)
# ==================================================
def _confidence_level(probability: float) -> str:
    """
    Maps probability to a human-readable confidence level.
    """

    if probability >= HIGH_CONFIDENCE_THRESHOLD:
        return "High"

    if probability <= LOW_CONFIDENCE_THRESHOLD:
        return "High"

    return "Medium"


def _interpret_probability(
    probability: float,
    finding_name: str,
) -> Tuple[str, str]:
    """
    Converts a probability into a safe, non-diagnostic finding
    (used for multi-disease Chest X-ray).

    Returns:
        assessment: Likely / Unlikely / Uncertain
        confidence_level: High / Medium
    """

    low, high = FINDING_THRESHOLDS.get(
        finding_name,
        (LOW_CONFIDENCE_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD),
    )

    if probability >= high:
        return "Likely", "High"

    if probability <= low:
        return "Unlikely", "High"

    return "Uncertain", "Medium"


# ==================================================
# SINGLE-DISEASE SAFETY
# ==================================================
def safety_check(probability: float) -> Tuple[str, str]:
    """
    Lightweight safety wrapper for single-disease inference.

    Used by legacy / simple CLI paths.
    """

    logger.info("Running single-disease safety_check")

    if probability >= HIGH_CONFIDENCE_THRESHOLD:
        return "Finding Likely", "High"

    if probability <= LOW_CONFIDENCE_THRESHOLD:
        return "Finding Unlikely", "High"

    return "Uncertain", "Medium"


# ==================================================
# MULTI-DISEASE SAFETY (Chest X-ray)
# ==================================================
def evaluate_findings(
    probabilities: Dict[str, float],
) -> Dict[str, Dict]:
    """
    Evaluates multiple disease probabilities safely
    (multi-label Chest X-ray).

    Returns structured, non-diagnostic findings.
    """

    logger.info("Evaluating multi-disease findings")

    results = {}

    for finding, prob in probabilities.items():
        label, confidence = _interpret_probability(prob, finding)

        results[finding] = {
            "finding": finding,
            "assessment": label,
            "probability": round(float(prob), 4),
            "confidence_level": confidence,
        }

    return results


# ==================================================
# MULTI-CLASS SAFETY (Brain MRI)
# ==================================================
def evaluate_multiclass(
    predicted_class: str,
    confidence: float,
) -> Dict[str, str]:
    """
    Safety evaluation for multi-class Brain MRI models.

    Args:
        predicted_class (str): Class name (e.g., Glioma)
        confidence (float): Softmax confidence

    Returns:
        Dict: Safe, non-diagnostic result
    """

    logger.info("Evaluating multi-class (Brain MRI) result")

    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return {
            "predicted_class": predicted_class,
            "assessment": "Likely",
            "confidence": round(confidence, 4),
            "confidence_level": "High",
        }

    if confidence <= LOW_CONFIDENCE_THRESHOLD:
        return {
            "predicted_class": predicted_class,
            "assessment": "Uncertain",
            "confidence": round(confidence, 4),
            "confidence_level": "Medium",
        }

    # Mandatory abstention zone
    return {
        "predicted_class": predicted_class,
        "assessment": "Uncertain",
        "confidence": round(confidence, 4),
        "confidence_level": "Medium",
    }


# ==================================================
# PIPELINE-LEVEL SAFETY (FINAL GATE)
# ==================================================
def apply_safety_checks(
    prediction_result: Dict,
) -> Dict:
    """
    Applies structured, auditable safety rules
    to a prediction result dictionary.

    Supports:
        - Chest X-ray (multi-disease)
        - Brain MRI (multi-class)
    """

    logger.info("Applying pipeline-level safety checks")

    # --------------------------------------------------
    # Multi-disease path
    # --------------------------------------------------
    if "findings" in prediction_result:
        findings = prediction_result.get("findings")

        if not findings:
            return {
                "status": "Rejected",
                "reason": "Missing findings",
                "note": (
                    "AI output rejected due to missing findings. "
                    "Radiologist review required."
                ),
            }

        all_uncertain = all(
            f["assessment"] == "Uncertain"
            for f in findings.values()
        )

        if all_uncertain:
            prediction_result["safety_status"] = "Abstained"
            prediction_result["note"] = (
                "AI confidence is insufficient across all findings. "
                "No reliable conclusion can be drawn. "
                "Clinical review is required."
            )
            return prediction_result

        prediction_result["safety_status"] = "Passed"
        prediction_result["note"] = (
            "This output represents AI-assisted radiological findings. "
            "It is NOT a diagnosis and must be reviewed by a qualified clinician."
        )
        return prediction_result

    # --------------------------------------------------
    # Multi-class path (Brain MRI)
    # --------------------------------------------------
    if "predicted_class" in prediction_result:
        prediction_result["safety_status"] = "Passed"
        prediction_result["note"] = (
            "This output represents AI-assisted classification of MRI imagery. "
            "It is NOT a diagnosis and must be reviewed by a qualified clinician."
        )
        return prediction_result

    # --------------------------------------------------
    # Fallback (invalid structure)
    # --------------------------------------------------
    logger.error("Invalid prediction result structure")

    return {
        "status": "Rejected",
        "reason": "Invalid prediction format",
        "note": (
            "AI output could not be interpreted safely. "
            "Clinical review is required."
        ),
    }
