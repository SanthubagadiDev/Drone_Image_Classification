# 🚁 Drone Image Classification Using Machine Learning

A supervised machine learning project to classify different types of drones from images using classical machine learning techniques and image preprocessing with OpenCV.

---
## 📌 Overview

With the increasing use of drones in surveillance, defense, and commercial applications, automatically identifying drone types from images has become an important task.
This project focuses on building a machine learning-based image classification system that predicts the type of drone present in an image.

The system uses image preprocessing techniques to convert images into numerical features and applies a **Logistic Regression classifier** to perform multi-class classification. 
The trained model is deployed using Streamlit for real-time prediction.

---
# 🔗LIVE DEMO 

**Live_Website:** (http://droneimageclassification-nqv4x7awcimkvvkyppyvaa.streamlit.app/)

## 📊 Dataset Information

**Source:** Kaggle – Synthetic Drone Classification Dataset
**Dataset Link:** https://www.kaggle.com/datasets/balajikartheek/drone-type-classification

**Data Type:** Image data
**Classes:**

- no_drone
- dji_inspire
- dji_mavic
- dji_phantom

---

## 🎯 Problem Statement

Manually identifying drone types from images is time-consuming and error-prone.
The objective of this project is to develop a supervised machine learning model that can automatically classify drone images into predefined categories.

---

## 🛠 Tech Stack

**Language:** Python
**Libraries & Tools:**
- Pandas
- NumPy
- Matplotlib
- OpenCV
- Scikit-learn
- Streamlit
- Pickle

---
## Methodology

**1.Image Loading**
  - Images loaded from class-wise folders using OpenCV
**2.Image Preprocessing**
  - Resized to 64 × 64
  - Converted to grayscale
  - Flattened into 1D feature vectors
**3.Label Encoding**
  - Manual label mapping for consistent class encoding
**4.Feature Scaling**
  - StandardScaler applied to normalize pixel values
**5.Model Training**
  - Logistic Regression 
**6.Model Evaluation**
  - Accuracy used as primary metric

---

## 🤖 Model Used

### Logistic Regression

**Why Logistic Regression?**
  - Simple and fast
  - Works well with linearly separable features
  - Easy to interpret
  - Suitable for classical ML image pipelines
---

## 📈 Model Performance

### Accuracy: 93%
