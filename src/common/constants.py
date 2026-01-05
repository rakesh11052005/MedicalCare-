"""
constants.py

Centralized configuration for MedicalCare+.

CRITICAL ROLE:
- Single source of truth for ALL tasks
- Governs model wiring, loss selection, metrics, safety
- Must remain task-explicit and audit-safe

Design principles:
- Clinically conservative defaults
- Explicit task separation (NO ambiguity)
- Findings ≠ Diagnoses (regulatory safe)
"""

# ==================================================
# PROJECT INFORMATION
# ==================================================

PROJECT_NAME = "MedicalCare+"
PROJECT_VERSION = "1.0.0"

# ==================================================
# IMAGE PROCESSING (SHARED)
# ==================================================

# Standard input resolution (ImageNet compatible)
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

# All modalities are converted to 3-channel input
IMAGE_CHANNELS = 3

# ==================================================
# TRAINING DEFAULTS (SAFE BASELINES)
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
# TASK REGISTRY (CRITICAL)
# ==================================================

"""
ALL supported tasks MUST be declared here.
No task string should be invented elsewhere.
"""

TASK_PNEUMONIA = "pneumonia"
TASK_MULTIDISEASE = "multidisease"
TASK_BRAIN_MRI = "brain_mri"

SUPPORTED_TASKS = [
    TASK_PNEUMONIA,
    TASK_MULTIDISEASE,
    TASK_BRAIN_MRI,
]

# ==================================================
# CHEST X-RAY — MULTI-DISEASE (MULTI-LABEL)
# ==================================================

"""
IMPORTANT:
These are FINDINGS, not diagnoses.
They are NOT mutually exclusive.
"""

XRAY_FINDINGS = [
    "Pneumonia",
    "Tuberculosis",
    "COVID-19",
    "Lung Opacity",
]

NUM_XRAY_FINDINGS = len(XRAY_FINDINGS)

# Backward compatibility aliases (DO NOT REMOVE)
FINDINGS = XRAY_FINDINGS
DISEASES = XRAY_FINDINGS
NUM_FINDINGS = NUM_XRAY_FINDINGS
NUM_DISEASES = NUM_XRAY_FINDINGS

# Multi-disease X-ray is multi-label by definition
MULTILABEL = True

# ==================================================
# BRAIN MRI — MULTI-CLASS (MUTUALLY EXCLUSIVE)
# ==================================================

"""
Brain MRI task characteristics:
- Exactly ONE class per image
- Uses softmax + CrossEntropyLoss
"""

BRAIN_MRI_CLASSES = [
    "Normal",
    "Glioma",
    "Meningioma",
    "Pituitary",
]

NUM_BRAIN_MRI_CLASSES = len(BRAIN_MRI_CLASSES)

# ==================================================
# USER SELECTION POLICY (XRAY ONLY)
# ==================================================

DEFAULT_SELECTION_MODE = "auto"
ALL_FINDINGS_KEYWORD = "ALL"

# ==================================================
# CONFIDENCE & UNCERTAINTY THRESHOLDS
# ==================================================

"""
X-RAY FINDING THRESHOLDS
(probability <= LOW  → Unlikely
 LOW < prob < HIGH   → Uncertain
 prob >= HIGH        → Likely)
"""

FINDING_THRESHOLDS = {
    "Pneumonia": (0.30, 0.70),
    "Tuberculosis": (0.35, 0.75),
    "COVID-19": (0.40, 0.80),
    "Lung Opacity": (0.30, 0.70),
}

# --------------------------------------------------
# GLOBAL CONFIDENCE THRESHOLDS (TASK-AGNOSTIC)
# --------------------------------------------------

LOW_CONFIDENCE_THRESHOLD = 0.30
HIGH_CONFIDENCE_THRESHOLD = 0.70

# --------------------------------------------------
# HARD ABSTENTION ZONE (ALL TASKS)
# --------------------------------------------------

UNCERTAINTY_LOWER = 0.20
UNCERTAINTY_UPPER = 0.80

# ==================================================
# OUTPUT TERMINOLOGY (CLINICALLY SAFE)
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
    "It represents AI-assisted medical image analysis "
    "and must be reviewed by a qualified clinician."
)
