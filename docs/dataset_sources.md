# Dataset Sources – MedicalCare+

## 1. Purpose of This Document

This document lists and describes the datasets used or planned to be used
in the MedicalCare+ project.

Transparent documentation of dataset sources is critical for:
- Ethical AI development
- Scientific reproducibility
- Bias analysis
- Clinical trust
- Regulatory review

Only legally available and ethically sourced datasets are used.

---

## 2. Primary Dataset (Current)

### RSNA Pneumonia Detection Challenge Dataset

**Source:**
- Radiological Society of North America (RSNA)
- Hosted via public research platforms (e.g., Kaggle)

**Imaging Modality:**
- Chest X-ray (CXR)

**Primary Use:**
- Training and evaluating pneumonia detection models

**Labels:**
- Normal
- Pneumonia

**Key Characteristics:**
- Large-scale dataset
- Anonymized patient data
- Clinically reviewed annotations
- Widely used in academic and industrial research

**Reason for Selection:**
- High data quality
- Medical relevance
- Public availability
- Benchmark dataset in medical AI literature

---

## 3. Data Usage Scope

For MedicalCare+, the dataset is used to:
- Train machine learning models
- Evaluate model performance
- Analyze safety and calibration
- Develop explainability methods

The dataset is **NOT used** to:
- Identify individual patients
- Infer personal or demographic identities
- Replace clinical diagnosis

---

## 4. Data Licensing & Compliance

MedicalCare+ complies with dataset licensing terms:

- Data is used strictly for research and development
- No redistribution of raw images is performed
- Dataset credits are preserved
- Original dataset terms are respected

Users of MedicalCare+ are responsible for ensuring
compliance with dataset licenses in their environment.

---

## 5. Data Preprocessing & Handling

To ensure safety and consistency:

- Images are resized to a fixed resolution
- Pixel values are normalized
- No identifying metadata is retained
- Corrupted or invalid images are excluded

Original raw data remains unchanged.

---

## 6. Known Dataset Limitations

Public medical datasets may have limitations:

- Class imbalance (more normal than disease cases)
- Limited demographic diversity
- Variability in image acquisition protocols
- Institution-specific biases

These limitations are acknowledged and documented
in the system limitations.

---

## 7. Future Dataset Expansion (Planned)

MedicalCare+ is designed to support additional datasets in the future, such as:

- NIH ChestX-ray14
- CheXpert
- MIMIC-CXR

Future dataset integration will follow:
- Ethical review
- Bias assessment
- Updated documentation

No additional datasets are included without explicit documentation updates.

---

## 8. Data Governance Principles

MedicalCare+ follows these data governance principles:

- Transparency in data sourcing
- Respect for patient privacy
- Ethical usage of medical data
- Clear documentation of limitations
- Accountability in data handling

These principles guide all dataset-related decisions.

---

## 9. Dataset Attribution Statement

MedicalCare+ acknowledges and credits all dataset creators
and institutions responsible for data collection and annotation.

The success of this project depends on the contributions
of the global medical research community.

---

## 10. Disclaimer

This project does not claim ownership of any dataset.

Datasets remain the property of their respective owners,
and all rights and responsibilities remain with the original providers.

Users must independently verify dataset suitability
for their intended use cases.
