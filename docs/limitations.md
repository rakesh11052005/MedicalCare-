# System Limitations – MedicalCare+

## 1. Overview

MedicalCare+ is a machine learning–based **clinical decision support system**
designed to assist healthcare professionals in analyzing medical imaging data,
including **chest X-ray images and brain MRI scans**.

Despite careful design and validation, MedicalCare+ has inherent limitations.
Understanding these limitations is critical for safe and responsible use.

---

## 2. Not a Diagnostic Authority

MedicalCare+ does NOT provide medical diagnoses.

- The system outputs probability-based assessments only
- Final diagnosis must always be made by a qualified healthcare professional
- MedicalCare+ must never be used as the sole basis for treatment decisions

The system is an **assistive tool**, not a replacement for clinical expertise.

---

## 3. Imaging Modality Limitations

MedicalCare+ supports a **limited set of imaging modalities**:

### Supported Modalities
- Chest X-ray (2D)
- Brain MRI (2D slices)

### Modality Constraints
- Diseases not visible in the provided imaging modality cannot be detected
- Subtle or early-stage abnormalities may be missed
- The system does not analyze:
  - CT scans
  - Ultrasound
  - Pathology slides
  - Laboratory or clinical records

MedicalCare+ does **not** perform multi-modal fusion at this stage.

---

## 4. Disease & Task Scope Limitations

MedicalCare+ supports a **limited and predefined set of tasks**:

### Chest X-ray Models
- Detect specific radiological findings
- Treat each finding independently (multi-label)
- Do not provide differential diagnoses

### Brain MRI Models
- Perform **multi-class classification** on 2D MRI slices
- Do NOT perform:
  - Tumor segmentation
  - Tumor grading or staging
  - Progression or longitudinal analysis
  - Surgical or treatment planning

The system does not determine disease causes or recommend treatments.

---

## 5. Data-Related Limitations

Model performance depends heavily on training data.

Potential issues include:
- Dataset imbalance across classes or findings
- Limited demographic diversity in public datasets
- Variability in image quality, scanners, and acquisition protocols

These factors may reduce generalization to unseen populations or institutions.

---

## 6. Confidence & Uncertainty

Although MedicalCare+ includes calibration and safety checks:

- Confidence scores are probabilistic estimates, not guarantees
- Some predictions may be marked as **“Uncertain”**
- The system may abstain from making a recommendation
- Low-confidence outputs must always prompt human review

Uncertainty handling is intentional and prioritizes patient safety.

---

## 7. Bias & Fairness Constraints

MedicalCare+ acknowledges the risk of bias in medical AI systems.

Limitations include:
- Incomplete representation of global populations
- Potential bias from dataset collection practices
- Limited subgroup performance analysis in early versions

Bias monitoring and mitigation are ongoing efforts.

---

## 8. Explainability Constraints

While explainability tools (e.g., Grad-CAM) are provided:

- Heatmaps indicate regions influencing the model, not causation
- Highlighted regions do not guarantee pathological relevance
- Explainability outputs require expert clinical interpretation

Explainability is a support mechanism, not definitive evidence.

---

## 9. Deployment & Environment Limitations

MedicalCare+ performance may vary depending on:
- Hardware (CPU vs GPU)
- Image preprocessing pipelines
- Integration environments (local, cloud, hospital systems)

Deployment-specific validation is required before clinical use.

---

## 10. Regulatory & Legal Limitations

MedicalCare+ is **not certified as a medical device**.

- Regulatory approval may be required before clinical deployment
- Compliance with local healthcare regulations is mandatory
- This project does not claim regulatory clearance

Users are responsible for ensuring regulatory compliance.

---

## 11. Continuous Improvement Disclaimer

MedicalCare+ is under continuous development.

Limitations may be:
- Reduced through additional data
- Addressed through model improvements
- Updated through clinician feedback

This document will evolve alongside the system.

---

## 12. Limitation Statement

MedicalCare+ is designed with transparency and caution.

> Awareness of limitations is essential to safe and ethical use.  
> Ignoring system limitations can lead to misuse and harm.

Users are expected to review and understand this document
before relying on the system in any context.
