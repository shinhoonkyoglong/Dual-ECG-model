## Capstone-2025-Autumn Semester
# Dual-ECG Patient Improvement Tracking Model

> **A novel deep learning approach using a modified ST-MEM autoencoder to detect patient improvement by comparing serial ECGs.**

<p align="center">
  <img src="[여기에 ViT vs 1D CNN 구조 비교 다이어그램 이미지 주소]" width="80%">
  <br>
  <em>Figure 1: Architecture modification from standard ViT-based ST-MEM to the proposed 1D CNN-based structure for enhanced local feature sensitivity.</em>
</p>

## 📖 Project Overview
In clinical settings, comparing a patient's past and present ECGs is crucial for monitoring disease progression and verifying drug efficacy. However, automating this comparison is challenging due to the subtle nature of localized signal changes over time.

This project developed a deep learning model designed to classify patient improvement ("Improved" vs. "Others") by analyzing dual ECG inputs.

## 💡 Key Innovation: Re-architecting ST-MEM
The core innovation of this project lies in identifying and overcoming the limitations of existing state-of-the-art models for this specific task.

* **The Challenge:** Standard **ST-MEM (Spatiotemporal-Masked Autoencoder)** models rely on **Vision Transformers (ViT)**. While powerful, ViTs tend to average attention across leads, losing critical, localized signal variations necessary for comparing two different timepoints.
* **My Solution (Architectural shift):** I **re-architected the ST-MEM autoencoder by replacing the conventional ViT encoder/decoder with a custom 1D CNN structure.**
* **Why 1D CNN?:** Unlike Transformers that look at global context, CNNs are inherently designed to capture **local patterns**. This modification significantly enhanced the model's sensitivity to minute, localized changes between the dual ECGs.

## 🔬 Technical Approach & Experiments
* **Data:** Real-world clinical data from Pusan National University Hospital.
* **Preprocessing:** Raw signals were resampled to 500Hz and cropped to 10s, resulting in 2500-length sequences.
* **Feature Fusion:** Experimented with various techniques to combine features from the two ECGs, including **subtraction, concatenation, and attention mechanisms**, to maximize comparison accuracy.

## 🏆 Results
The proposed 1D CNN-based ST-MEM model achieved significant performance gains in the challenging binary classification task of tracking improvement.

| Model | F1 Score | Note |
| :--- | :---: | :--- |
| Baseline (ViT-based ST-MEM) | 0.3 - 0.5 | Typical performance for this specific dual-task |
| **Proposed Model (1D CNN ST-MEM)** | **0.60** | **Significant Improvement** |

*Achieved a noteworthy F1 score of 0.6, substantially outperforming baseline models.*

## 🛠 Tech Stack
- **Deep Learning:** PyTorch
- **Model Architecture:** Custom ST-MEM (Autoencoder based on 1D CNN)
- **Data Processing:** Pandas, NumPy
![2025캡스톤 포스터](https://github.com/user-attachments/assets/8195d70a-c1b6-4eeb-bf89-7cc255a12a24)
