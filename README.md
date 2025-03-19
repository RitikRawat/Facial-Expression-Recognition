# 😀 Facial Expression Recognition using CNNs

This project focuses on detecting and classifying human emotions from facial images using deep Convolutional Neural Networks (CNNs). The model is trained to recognize key facial expressions such as happiness, sadness, anger, surprise, fear, disgust, and neutrality.

---

## 📖 Introduction

Facial Expression Recognition (FER) is a vital task in computer vision and affective computing, enabling machines to interpret human emotions based on visual cues. Applications span across:
- Human-Computer Interaction (HCI)
- Healthcare and mental wellness
- Surveillance and security
- Entertainment and AR/VR systems

This project builds a robust FER system using CNN architectures trained on standard datasets, enabling real-time emotion classification from facial images.

---

## 🎯 Objectives

- Build a CNN model to classify 7 basic emotions from facial images.
- Achieve high accuracy while minimizing overfitting.
- Support real-time emotion detection (optional extension).
- Evaluate model generalization using unseen test samples.

---

## 🧠 Methodology

### 🔹 Data Preprocessing

- **Dataset**: FER2013 (or similar), grayscale 48x48 facial images.
- **Augmentation**:
  - Random rotations
  - Zoom
  - Horizontal flips
- **Normalization**: Scale pixel values to `[0, 1]`.

### 🔹 CNN Architecture

- Input: 48x48 grayscale images
- Conv2D + ReLU + MaxPooling layers
- Dropout for regularization
- Flatten + Fully Connected Layers
- Output: Softmax layer with 7 emotion classes

### 🔹 Training

- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam / SGD
- **Metrics**: Accuracy, Confusion Matrix
- **Epochs**: 30–50
- **Callbacks**: Early stopping, model checkpointing

---

## 📊 Results

| Metric         | Value        |
|----------------|--------------|
| Train Accuracy | ~98%         |
| Test Accuracy  | ~67–75%      |
| Best Loss      | ~0.85        |
| Confusion Matrix | Included in notebook |

- The model performs well on common emotions like **happy**, **neutral**, and **sad**.
- Minor confusion between **fear**, **surprise**, and **disgust** due to visual similarity.
- Generalizes reasonably to new facial data.

---

## 🧪 Evaluation

- Accuracy and loss curves (train vs. val)
- Confusion matrix for per-class performance
- Manual testing with custom images (optional)

---


