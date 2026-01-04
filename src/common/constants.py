# File: MedicalCare+/src/common/constants.py

"""
constants.py

Centralized configuration for MedicalCare+.

Design principles:
- Single source of truth
- Clinically safe defaults
- Supports multi-disease inference
- Allows user-selected or automatic findings
- Findings ≠ Diagnoses (regulatory safe)
"""

# ==================================================
# PROJECT INFORMATION
# ==================================================

PROJECT_NAME = "MedicalCare+"
PROJECT_VERSION = "1.0.0"

# ==================================================
# IMAGE PROCESSING
# ==================================================

# Standard input resolution (ImageNet compatible)
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

# Model input channels
# X-rays are grayscale → replicated to 3 channels
IMAGE_CHANNELS = 3

# ==================================================
# TRAINING CONFIGURATION
# ==================================================

BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
RANDOM_SEED = 42

# ==================================================
# DEPLOYMENT PROFILE
# ==================================================

# "research"  → permissive thresholds
# "clinical"  → conservative, abstains more
DEPLOYMENT_MODE = "clinical"

# ==================================================
# MULTI-DISEASE / FINDINGS CONFIGURATION
# ==================================================

"""
IMPORTANT:
These are FINDINGS, not diagnoses.
The system scans for all by default.
User may optionally request a subset.
"""

FINDINGS = [
    "Pneumonia",
    "Tuberculosis",
    "COVID-19",
    "Lung Opacity",
]

NUM_FINDINGS = len(FINDINGS)

# --------------------------------------------------
# Canonical model-facing constants
# --------------------------------------------------

# Multidisease X-ray is multi-label (findings are NOT mutually exclusive)
MULTILABEL = True

# Model output dimensionality (EXPECTED BY model_factory, losses, metrics)
NUM_CLASSES = NUM_FINDINGS

# Backward compatibility aliases
DISEASES = FINDINGS
NUM_DISEASES = NUM_FINDINGS

# ==================================================
# USER SELECTION POLICY
# ==================================================

"""
USER_SELECTION_MODE:

- "auto" → model evaluates ALL findings (default)
- "selected" → user provides a subset of findings
"""

DEFAULT_SELECTION_MODE = "auto"

# Special keyword meaning "evaluate everything"
ALL_FINDINGS_KEYWORD = "ALL"

# ==================================================
# CONFIDENCE & UNCERTAINTY THRESHOLDS
# ==================================================

"""
Threshold semantics per finding:

probability <= LOW  → Unlikely
LOW < prob < HIGH   → Uncertain (abstain)
probability >= HIGH → Likely
"""

FINDING_THRESHOLDS = {
    "Pneumonia": (0.30, 0.70),
    "Tuberculosis": (0.35, 0.75),
    "COVID-19": (0.40, 0.80),
    "Lung Opacity": (0.30, 0.70),
}

# Global fallback thresholds
LOW_CONFIDENCE_THRESHOLD = 0.30
HIGH_CONFIDENCE_THRESHOLD = 0.70

# ==================================================
# UNCERTAINTY / ABSTENTION POLICY
# ==================================================

"""
Hard abstention zone (never produce strong claims here)
Used by safety layer and evaluation
"""

UNCERTAINTY_LOWER = 0.20
UNCERTAINTY_UPPER = 0.80

# ==================================================
# OUTPUT TERMINOLOGY (CLINICAL SAFE)
# ==================================================

STATUS_LIKELY = "Likely"
STATUS_POSSIBLE = "Possible"
STATUS_UNCERTAIN = "Uncertain"
STATUS_UNLIKELY = "Unlikely"

CONFIDENCE_HIGH = "High"
CONFIDENCE_MEDIUM = "Medium"
CONFIDENCE_LOW = "Low"

# ==================================================
# LEGAL / MEDICAL DISCLAIMER
# ==================================================

DISCLAIMER_TEXT = (
    "This output is NOT a diagnosis. "
    "It is an AI-assisted radiological analysis "
    "and must be reviewed by a qualified clinician."
)
