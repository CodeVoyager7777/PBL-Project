# 🎯 Real-Time Optical Flow for Probe Trajectory Scoring

### Extracting Motion Patterns from Fetal Ultrasound (US)

---

## 📌 Overview

This project focuses on analyzing **probe movement in fetal ultrasound videos** using **Optical Flow** and **Deep Learning** techniques.

The system computes motion-based **stability scores** and uses them to automatically generate labels, followed by training a **3D Convolutional Neural Network (3D CNN)** for classification of probe trajectory quality.

---

## 🚀 Key Features

* 🎥 Video preprocessing and frame extraction
* 🌊 Optical Flow-based motion analysis
* 📊 Stability score computation
* 🏷️ Automatic dataset labeling
* 🗂️ Dataset organization (stable vs unstable)
* 🧠 3D CNN model for classification
* 📈 Model training and evaluation

---

## 🧠 Project Pipeline

```plaintext
Raw Ultrasound Videos
        ↓
Optical Flow Analysis (main.py)
        ↓
Motion Features + Stability Score
        ↓
Auto Label Generation (sort_dataset.py)
        ↓
Structured Dataset
   ├── stable/
   └── unstable/
        ↓
3D CNN Training (train.py)
        ↓
Classification Accuracy
```

---

## 📁 Project Structure

```plaintext
PBL-Project/
│
├── src/
│   ├── main.py              # Optical flow + feature extraction
│   ├── sort_dataset.py     # Label generation + sorting
│   ├── train.py            # Model training
│   ├── data_loader.py      # Dataset loading
│   ├── model_3dcnn.py      # 3D CNN model
│
├── dataset/
│   ├── stable/
│   └── unstable/
│
├── data/
│   └── labels.xlsx
│
├── Results/
│   └── final_results.xlsx
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/CodeVoyager7777/PBL-Project.git
cd PBL-Project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## ▶️ Usage

### 1️⃣ Optical Flow Analysis

```bash
python src/main.py
```

### 2️⃣ Dataset Preparation

```bash
python src/sort_dataset.py
```

### 3️⃣ Model Training

```bash
python src/train.py
```

---

## 🧪 Model Details

* Model: **3D Convolutional Neural Network (3D CNN)**
* Input: Video clips
* Frame size: `64 × 64`
* Frames per video: `8`
* Classes:

  * Stable Probe Movement
  * Unstable Probe Movement

---

## 💡 Core Idea

Instead of manual labeling, this system:

> **Automatically generates labels using motion-based stability scores derived from optical flow analysis**

This enables scalable and intelligent dataset creation.

---

## 📈 Future Scope

* Real-time deployment in ultrasound systems
* Integration with medical decision support
* Improved motion feature extraction
* Higher accuracy deep learning models

---

## 👨‍💻 Author

Laksh Makkar

---

## ⭐ Note

This project demonstrates a complete pipeline combining **Computer Vision (Optical Flow)** and **Deep Learning (3D CNN)** for real-world medical video analysis.
