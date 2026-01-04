# File: MedicalCare+/src/inference/safety.py

"""
safety.py

Unified safety, uncertainty, and abstention logic for MedicalCare+.

Medical & regulatory guarantees:
- NEVER outputs a diagnosis
- Enforces abstention in uncertain zones
- Deterministic, auditable decision logic
- Shared policy for single-disease & multi-disease inference
- Clinician-first design (AI assists, never decides)

This file is a SINGLE SOURCE OF TRUTH for safety behavior.
"""

from typing import Dict, Tuple

from src.common.logging_utils import setup_logger
from src.common.constants import (
    LOW_CONFIDENCE_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLD,
    FINDING_THRESHOLDS
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
    finding_name: str
) -> Tuple[str, str]:
    """
    Converts a probability into a safe, non-diagnostic finding.

    Args:
        probability (float): Model probability
        finding_name (str): Name of disease / finding

    Returns:
        finding_label (str): Likely / Possible / Uncertain / Unlikely
        confidence_level (str): High / Medium
    """

    low, high = FINDING_THRESHOLDS.get(
        finding_name,
        (LOW_CONFIDENCE_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD)
    )

    # High confidence positive
    if probability >= high:
        return "Likely", "High"

    # High confidence negative
    if probability <= low:
        return "Unlikely", "High"

    # Mandatory abstention zone
    return "Uncertain", "Medium"


# ==================================================
# LOW-LEVEL SAFETY (USED BY CLI PREDICTION)
# ==================================================
def safety_check(probability: float) -> Tuple[str, str]:
    """
    Lightweight safety wrapper for single-disease inference.

    IMPORTANT:
    - This does NOT return a diagnosis
    - This does NOT mention disease names
    - Used by predict.py (legacy & simple CLI)

    Args:
        probability (float): Model probability

    Returns:
        decision (str): Safe decision-support label
        confidence_level (str): High / Medium
    """

    logger.info("Running single-disease safety_check")

    if probability >= HIGH_CONFIDENCE_THRESHOLD:
        return "Finding Likely", "High"

    if probability <= LOW_CONFIDENCE_THRESHOLD:
        return "Finding Unlikely", "High"

    return "Uncertain", "Medium"


# ==================================================
# MULTI-DISEASE SAFETY (CORE LOGIC)
# ==================================================
def evaluate_findings(
    probabilities: Dict[str, float]
) -> Dict[str, Dict]:
    """
    Evaluates multiple disease probabilities safely.

    Args:
        probabilities (Dict):
            {
                "Pneumonia": 0.92,
                "Tuberculosis": 0.08,
                "COVID-19": 0.47,
                "Lung Opacity": 0.61
            }

    Returns:
        Dict[str, Dict]:
            Structured, safe findings for UI / CLI / API
    """

    logger.info("Evaluating multi-disease findings")

    results = {}

    for finding, prob in probabilities.items():
        label, confidence = _interpret_probability(prob, finding)

        results[finding] = {
            "finding": finding,
            "assessment": label,
            "probability": round(float(prob), 4),
            "confidence_level": confidence
        }

    return results


# ==================================================
# PIPELINE-LEVEL SAFETY (APIs / FUTURE UI)
# ==================================================
def apply_safety_checks(
    prediction_result: Dict
) -> Dict:
    """
    Applies structured, auditable safety rules
    to a prediction result dictionary.

    Expected input:
        {
            "findings": Dict[str, Dict],
            "model_version": str
        }

    Returns:
        Dict:
            Safety-verified output
    """

    logger.info("Applying pipeline-level safety checks")

    findings = prediction_result.get("findings")

    if not findings:
        logger.error("No findings present in prediction result")

        return {
            "status": "Rejected",
            "reason": "Missing findings",
            "note": (
                "AI output rejected due to missing findings. "
                "Radiologist review required."
            )
        }

    # Check if everything is uncertain
    all_uncertain = all(
        f["assessment"] == "Uncertain"
        for f in findings.values()
    )

    if all_uncertain:
        logger.warning("All findings fall into abstention zone")

        prediction_result["safety_status"] = "Abstained"
        prediction_result["note"] = (
            "AI confidence is insufficient across all findings. "
            "No reliable conclusion can be drawn. "
            "Clinical review is required."
        )
        return prediction_result

    # Otherwise allowed (still not a diagnosis)
    prediction_result["safety_status"] = "Passed"
    prediction_result["note"] = (
        "This output represents AI-assisted radiological findings. "
        "It is NOT a diagnosis and must be reviewed by a qualified clinician."
    )

    return prediction_result
