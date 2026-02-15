# Optical Flow-Based Probe Stability Analysis

## Overview

This project analyzes ultrasound probe motion stability using dense optical flow and motion feature engineering.  

The system processes raw ultrasound videos and computes a quantitative **Probe Stability Index (0–100)** based on motion magnitude, direction consistency, and temporal acceleration patterns.

The goal is to evaluate probe handling smoothness through motion analysis without relying on supervised learning.

---

## Problem Statement

In ultrasound imaging, stable probe movement is essential for acquiring consistent and high-quality scans.  
This project proposes an unsupervised motion analytics pipeline to quantify probe stability using optical flow.

---

## Methodology

The pipeline consists of:

1. **Video Preprocessing**
   - Frame extraction
   - Grayscale conversion
   - Noise reduction using Gaussian blur

2. **Optical Flow Computation**
   - Dense Optical Flow (Farneback method)
   - Pixel-wise motion estimation between consecutive frames

3. **Feature Extraction**
   - Mean motion magnitude
   - Motion standard deviation
   - Maximum motion
   - Direction variance
   - Acceleration variability

4. **Stability Index Computation**
   - Composite score (0–100)
   - Penalizes instability, abrupt motion, and inconsistency

5. **Visualization & Reporting**
   - Motion magnitude vs frame plots
   - Structured motion report generation

---

## Project Structure

Data/
raw_videos/

Results/
plots/
motion_reports/

src/
preprocessing.py
optical_flow.py
feature_extraction.py
stability_index.py
visualization.py
main.py

---

## How to Run

1. Place ultrasound videos inside:

Data/raw_videos/
2. Navigate to the source directory:

cd src
3. Run:

python main.py

4. Outputs will be saved inside:

---

## Output

For each video:

- Motion magnitude plot
- Extracted feature values
- Probe Stability Index (0–100)
- Text-based motion analysis report

---

## Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib

---

## Notes

- This system performs motion stability analysis only.
- It does not perform medical diagnosis.
- The approach is unsupervised and feature-driven.

