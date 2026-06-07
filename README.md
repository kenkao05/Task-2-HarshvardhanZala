# Project 2 — Data Classification Using AI 🤖

**DecodeLabs AI Internship | Batch 2026**
**Intern:** Harshvardhan Zala
**Track:** Artificial Intelligence

---

## Overview

A supervised machine learning pipeline built on the **K-Nearest Neighbors (KNN)** algorithm — no neural networks, no black boxes. Every classification decision is traceable back to the training data through measurable distances.

This project demonstrates the foundational supervised learning pipeline used in production AI systems: clean data in, trained model out, validated predictions back.

---

## Architecture — The IPO Framework

```
INPUT            →       PROCESS          →       OUTPUT
(Raw Feed)                (Logic Skeleton)          (Feedback Loop)

Iris Dataset          Train-Test Split          Confusion Matrix
Feature Scaling       KNN Algorithm             F1 Score
```

### Why Feature Scaling?

KNN is a **distance-based algorithm**. Without scaling, features with larger numeric ranges dominate the distance calculation, creating a biased model.

| Approach | Effect |
|---|---|
| Raw data (unscaled) | Sepal length (4–8) drowns out petal width (0.1–2.5) |
| **StandardScaler** | **Mean = 0, Variance = 1 — all features weighted equally** |

### Why F1 Score over Accuracy?

| Metric | Problem |
|---|---|
| Accuracy | In imbalanced data, a model predicting only the majority class can score 99% |
| **F1 Score** | **Harmonic mean of Precision & Recall — catches both false alarms and missed detections** |

---

## Dataset — The Iris Benchmark

| Property | Value |
|---|---|
| Samples | 150 (balanced) |
| Classes | 3 (Setosa, Versicolor, Virginica) |
| Features | 4 (sepal length, sepal width, petal length, petal width) |
| Class distribution | 50 samples per class |

---

## Project Specifications (Pipeline)

- ✅ **LOAD** — Iris dataset with correct column names, no nulls
- ✅ **EDA** — Class distribution, statistical summary, pairplot, heatmap, boxplots
- ✅ **SCALING** — StandardScaler applied before any model training
- ✅ **SPLIT** — 80% train / 20% test, shuffled to remove order bias
- ✅ **TUNE K** — Elbow method across K=1 to K=20, optimal K selected
- ✅ **TRAIN** — KNeighborsClassifier: instantiate → fit → predict
- ✅ **EVALUATE** — Accuracy, F1 Score, Classification Report, Confusion Matrix

---

## How to Run

**Requirements:** Python 3.x + the libraries below

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

**On Kaggle / Google Colab:** all libraries are pre-installed.

Open `decodelab-p2.ipynb` and run all cells top to bottom.

**Example output:**

```
✅ CHECKPOINT 1 PASSED: 150 rows, 5 columns, correct column names.
✅ CHECKPOINT 2 PASSED: No null values found.
✅ CHECKPOINT 3 PASSED: 3 balanced classes, 50 samples each.
✅ CHECKPOINT 4 PASSED: All feature ranges are valid.
✅ CHECKPOINT 5 PASSED: All features scaled to mean≈0, std≈1.
✅ CHECKPOINT 6 PASSED: 120 train / 30 test, 3 classes in each split, no data leakage.
✅ CHECKPOINT 7 PASSED: Best K = 2, Min error rate = 0.0000
✅ CHECKPOINT 8 PASSED: 30 valid predictions produced.

Accuracy : 1.0000
F1 Score : 1.0000

✅ CHECKPOINT 9 PASSED: Model meets performance thresholds.
```

---

## Checkpoints

Each major stage has an automated assertion checkpoint that prints PASSED or raises a descriptive FAIL message.

| Checkpoint | Stage | What It Verifies |
|---|---|---|
| ✅ 1 | Data Load | 150 rows, 5 columns, correct column names |
| ✅ 2 | Null Check | Zero missing values across all columns |
| ✅ 3 | Class Balance | Exactly 50 samples per class |
| ✅ 4 | Feature Ranges | All values within known Iris dataset bounds |
| ✅ 5 | Scaling | Mean ≈ 0 and Std ≈ 1 per feature |
| ✅ 6 | Train-Test Split | 120 train / 30 test, all 3 classes in both sets, no leakage |
| ✅ 7 | K Tuning | Best K in range [1, 20], error rate < 10% |
| ✅ 8 | Predictions | 30 outputs, all valid species labels |
| ✅ 9 | Performance | Accuracy ≥ 90%, F1 ≥ 90%, confusion matrix diagonal dominant |

---

## Concepts Demonstrated

- **Supervised Learning** — the machine learns from labeled training examples
- **Feature Scaling** — critical pre-processing for distance-based algorithms
- **Train-Test Split** — isolates evaluation data to prevent overfitting
- **K Tuning (Elbow Method)** — balances overfitting (K=1) vs underfitting (K=100)
- **KNN Algorithm** — proximity principle: similar things exist in close proximity
- **Confusion Matrix** — per-class breakdown of true/false positives and negatives
- **F1 Score** — harmonic mean of Precision & Recall; robust to class imbalance
- **IPO Model** — Input → Process → Output architecture for transparent AI

---

## Project Structure

```
decodelabs_ai_project2/
├── decodelab-p2.ipynb   # Main notebook — fully documented with checkpoints
├── iris_data.csv        # Iris dataset (150 samples, no header row)
└── README.md            # This file
```

---

*DecodeLabs | Batch 2026 | Artificial Intelligence Track*
