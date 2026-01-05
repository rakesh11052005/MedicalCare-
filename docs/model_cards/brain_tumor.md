# Model Card: Brain Tumor MRI Classification Model

## Project
**MedicalCare+**

## Model Name
**Brain Tumor MRI Classification Model**

## Version
**v1.0.0**

## Modality
**Magnetic Resonance Imaging (MRI)**  
(2D brain MRI slices)

## Task Type
**Multi-class image classification**

---

## 1. Model Overview

The Brain Tumor MRI Classification Model is a deep learning–based
**clinical decision support system** designed to assist healthcare
professionals in analyzing brain MRI images.

The model classifies a single MRI image into one of four **mutually
exclusive categories** based on learned imaging patterns.

This model is intended to **support**, not replace, expert clinical judgment.

---

## 2. Intended Use

### Primary Intended Use
- Assist radiologists and clinicians in **screening and prioritization**
  of brain MRI images
- Provide **probability-based classification** of tumor type
- Support clinical workflows with **explainable AI outputs (Grad-CAM)**

### Intended Users
- Radiologists
- Neurologists
- Qualified healthcare professionals
- Medical researchers (non-diagnostic use)

### Out-of-Scope Uses
This model is **NOT intended** to:
- Provide a medical diagnosis
- Determine tumor grade, stage, or aggressiveness
- Replace radiologist interpretation
- Be used directly by patients
- Be used as the sole basis for treatment decisions

---

## 3. Classification Labels

The model predicts one of the following classes:

| Label ID | Class Name     |
|--------:|---------------|
| 0       | Normal        |
| 1       | Glioma        |
| 2       | Meningioma    |
| 3       | Pituitary     |

> **Note:**  
> These labels represent **image-level findings**, not confirmed clinical diagnoses.

---

## 4. Model Architecture

### High-Level Architecture

