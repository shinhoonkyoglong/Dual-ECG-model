# Capstone-2025-Autumn Semester
# Dual-ECG Patient Improvement Tracking Model

> **Pioneering a novel approach to quantify patient trajectory by directly comparing dual-timepoint ECGs, overcoming the limitations of single-snapshot diagnostics.**

## Background & Motivation: The Clinical Need

### The Problem: Limitations of Conventional Analysis
* **Single-Snapshot View:** Most existing ECG analysis focuses on classifying diseases at a single point in time.
* **Lack of Longitudinal Insight:** This approach fails to capture **how a patient's condition changes over time**, making it difficult to quantitatively assess treatment efficacy or recovery progress.
* **Clinical Consequence:** This limitation hinders personalized treatment planning and can lead to unnecessary hospital re-visits and increased medical costs.

### Our Solution: A Paradigm Shift to Dual-Comparison
* **Defining a Novel Task:** This project addresses a previously under-researched area by developing a model that directly compares **two distinct ECG time-series data points** (e.g., pre-admission vs. post-discharge).
* **Objective Objectives:** The goal is to objectively track disease progression and evaluate drug efficacy, moving beyond simple diagnosis to continuous monitoring of patient trajectory.
* **Future Impact:** This model paves the way for real-time patient status checks (e.g., 10-minute intervals), pre/post-medication efficacy testing, and enabling reliable **remote patient monitoring**.

## Project Overview
In clinical settings, comparing a patient's past and present ECGs is crucial for monitoring disease progression and verifying drug efficacy. However, automating this comparison is challenging due to the subtle nature of localized signal changes over time.

This project developed a deep learning model designed to classify patient improvement ("Improved" vs. "Others") by analyzing dual ECG inputs.

## Key Innovation: Re-architecting ST-MEM
The core innovation of this project lies in identifying and overcoming the limitations of existing state-of-the-art models for this specific task.

* **The Challenge:** Standard **ST-MEM (Spatiotemporal-Masked Autoencoder)** models rely on **Vision Transformers (ViT)**. While powerful, ViTs tend to average attention across leads, losing critical, localized signal variations necessary for comparing two different timepoints.
* **My Solution (Architectural shift):** I **re-architected the ST-MEM autoencoder by replacing the conventional ViT encoder/decoder with a custom 1D CNN structure.**
* **Why 1D CNN?:** Unlike Transformers that look at global context, CNNs are inherently designed to capture **local patterns**. This modification significantly enhanced the model's sensitivity to minute, localized changes between the dual ECGs.

## Technical Approach & Experiments
* **Data:** Real-world clinical data from Pusan National University Hospital.
* **Preprocessing:** Raw signals were resampled to 500Hz and cropped to 10s, resulting in 2500-length sequences.
* **Feature Fusion:** Experimented with various techniques to combine features from the two ECGs, including **subtraction, concatenation, and attention mechanisms**, to maximize comparison accuracy.

## Results
The proposed 1D CNN-based ST-MEM model achieved significant performance gains in the challenging binary classification task of tracking improvement.

| Model | F1 Score | Note |
| :--- | :---: | :--- |
| Baseline (ViT-based ST-MEM) | 0.109 - 0.565 | Typical performance for this specific dual-task |
| **Proposed Model (1D CNN ST-MEM)** | **0.599** | **Significant Improvement** |

*Achieved a noteworthy F1 score of 0.6, substantially outperforming baseline models.*

## Tech Stack
- **Deep Learning:** PyTorch
- **Model Architecture:** Custom ST-MEM (Autoencoder based on 1D CNN)
- **Data Processing:** Pandas, NumPy
![2025캡스톤 포스터](https://github.com/user-attachments/assets/8195d70a-c1b6-4eeb-bf89-7cc255a12a24)
