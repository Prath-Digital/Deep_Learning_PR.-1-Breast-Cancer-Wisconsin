# 🎗️ Breast Cancer Diagnostic Classification 🚀

### _Deep Learning — Practical Report 1_

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.15-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

---

## 🌟 Project Overview

Welcome to the **Breast Cancer Diagnostic Classification** project! 🎯 This repository explores the powerful world of **Deep Learning** to differentiate between benign and malignant breast lesions. By leveraging artificial neural network architectures—progressing from elementary **Single-Layer Perceptrons** to deep **Multi-Layer Perceptrons** regularized with **Early Stopping**, **Dropout**, and **Weight Decay**—we construct a high-precision, clinically calibrated diagnostic decision support tool.

### 🎯 Core Objectives

- 🔍 **Explore:** Deep dive into 30 cytological nuclear morphological features through exploratory analysis and correlation heatmaps.
- ⚙️ **Preprocess:** Implement stratified sampling and `StandardScaler` normalization to prevent data leakage and ensure isotropic loss optimization.
- 🧠 **Architect:** Design and benchmark **Single-Layer Perceptrons (SLP)** and **Multi-Layer Perceptrons (MLP)** across ReLU, Tanh, and Sigmoid activations.
- 🛡️ **Regularize:** Master over-fitting suppression using **Early Stopping**, **Dropout** ensembling, and **Weight Norm Penalties** (L1, L2, ElasticNet).
- ⚖️ **Compare:** Benchmark training dynamics, generalization gaps, and test loss trajectories across 11 model configurations.
- 💡 **Insight:** Transform classification outputs into calibrated **Clinical Decision Support Intelligence** to eliminate fatal false negatives.

### 🎓 Academic Context

| 🏫 Institute                    | 📚 Subject    | 📝 Project                | 📊 Domain                            |
| :------------------------------ | :------------ | :------------------------ | :----------------------------------- |
| **Red & White Skill Education** | Deep Learning | Practical Report 1 (PR 1) | Healthcare AI & Oncology Diagnostics |

---

## 📊 The Data Journey

### 📂 Dataset Details

The project utilizes the **Breast Cancer Wisconsin (Diagnostic) Dataset** sourced directly from the **UCI Machine Learning Repository** via `ucimlrepo.fetch_ucirepo(id=17)` (Wolberg, Mangasarian, Street, 1993).

- **Source:** [UCI Machine Learning Repository (ID: 17)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) 🌐
- **API Fetch:** `breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)`
- **Size:** 569 Biopsy Samples | 30 Continuous Cytological Predictors | 1 Target Column 📏
- **Quality:** 💎 0 Missing Values | 0 Duplicates | 100% Data Integrity
- **Target Distribution:**
  - 🟢 **Benign (1):** 357 Patients (62.74%)
  - 🔴 **Malignant (0):** 212 Patients (37.26%)

| 📋 Feature Group | 🔬 Attributes Included | 🏛️ Morphological Domain | 📝 Clinical Description |
| :--- | :--- | :--- | :--- |
| **Mean Values (1–10)** | `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, etc. | Mean Nuclear Centroid | Baseline nuclear morphometry per fine-needle aspirate |
| **Standard Error (11–20)** | `radius_se`, `texture_se`, `perimeter_se`, `area_se`, etc. | Nuclear Variability | Variance across sampled cell nuclei per slide |
| **"Worst" / Extreme (21–30)** | `radius_worst`, `texture_worst`, `perimeter_worst`, `area_worst`, etc. | Extreme Pleomorphism | Mean of the three largest (most aberrant) nuclear measurements |
| **Diagnosis (Target)** | `Diagnosis` ('M'/'B') $\to$ `target` (0 = Malignant, 1 = Benign) | Ground-Truth Pathology | Binary ground-truth pathology confirmation |

### 🛠️ Data Preprocessing & Column Engineering Pipeline

1. **📥 Dataset Ingestion:** Ingested directly via `ucimlrepo.fetch_ucirepo(id=17)`, extracting features, targets, metadata, and variables.
2. **🏷️ Systematic Column Standardization:** Converted cryptic raw UCI suffixes (`1` $\to$ `_mean`, `2` $\to$ `_se`, `3` $\to$ `_worst`) into standard clinical descriptors. Constructed a full 30-feature taxonomy across Size & Geometry, Shape & Boundary, and Texture & Symmetry.
3. **🎯 Target Column Engineering:** Preserved categorical `Diagnosis` (`'M'` / `'B'`) and engineered numeric `target` (`0 = Malignant`, `1 = Benign`) to align with oncology decision loss functions.
4. **🔬 Column Predictive Power Ranking:** Computed feature-to-target Pearson correlation, discovering that concavity and boundary features (`concave_points_worst` $|r|=0.794$, `perimeter_worst` $|r|=0.783$, `concave_points_mean` $|r|=0.777$, `radius_worst` $|r|=0.776$) possess the highest discriminatory power.
5. **⚖️ Stratified Splitting:** Partitioned into an 80/20 train-test split (`random_state=42`, stratified on `y`), yielding **455 training biopsies** (285 benign, 170 malignant) and **114 test biopsies** (72 benign, 42 malignant) to preserve exact class ratios.
6. **📏 Feature Scaling:** Fit `StandardScaler` strictly on `X_train` and transformed `X_test` to neutralize dimensional disparities (`area_mean` ~1000s vs `smoothness_mean` ~0.1), preventing feature dominance and ensuring isotropic loss contours.
7. **🔬 Collinearity Analysis:** Mapped correlation heatmaps, identifying extreme collinearity ($r > 0.95$) across geometric dimension triplets (`radius`, `perimeter`, `area`), necessitating regularized architectures.

---

## 🤖 Deep Learning Architecture Suite

We designed, trained, and benchmarked five progressive neural network paradigms:

### 1️⃣ Single-Layer Perceptron (SLP) — Linear Baseline 🎯

- **Architecture:** 1 Dense layer (1 Sigmoid unit), 31 trainable parameters.
- **Optimization:** Trained with Binary Cross-Entropy loss for 50 epochs.
- **Result:** Achieved **94.74% Test Accuracy** and **0.1732 Test Loss**.
- **Theory & Strengths:** Establishes linear separability benchmark; highlights the geometric limits of single hyperplanes (the classic Minsky-Papert XOR problem).

### 2️⃣ Multi-Layer Perceptron (MLP) — Universal Approximation 🧠

- **Architecture:** 3 Dense layers (`[64, 32, 1]`), 4,097 trainable parameters.
- **Activation Benchmark:** Compared **ReLU** vs. **Tanh** vs. **Sigmoid** over 100 epochs.
- **Result:** **ReLU** triumphed with **96.49% Test Accuracy** and **0.1292 Test Loss** (vs. Tanh at 95.61%, Sigmoid at 93.86%).
- **Theory & Strengths:** Non-linear hidden units unlock Cybenko’s Universal Approximation theorem. ReLU mitigates the vanishing gradient problem by maintaining a constant derivative of 1.0 for positive pre-activations.

### 3️⃣ High-Capacity MLP & Early Stopping Dynamics 🛑

- **Architecture:** 3 Dense layers (`[128, 64, 1]`), 12,289 trainable parameters, trained for 300 epochs.
- **Overfitting Demonstration:** Unconstrained training drove training loss to near-zero while validation loss blew up from 0.089 to 0.2486 (test loss 0.2486).
- **Early Stopping:** `EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)` halted training at optimal validation trough (Epoch 42).
- **Result:** Restored best weights achieved **96.49% Accuracy** and the lowest overall test loss of **0.1272**.

### 4️⃣ Dropout Regularization & Weight Penalties 🛡️

- **Dropout Ensembling:** Integrated stochastic neuron deactivation masks ($p = 0.3$), acting as an implicit exponential pseudo-ensemble with **0 additional parameters** and shrinking the train-val generalization gap.
- **Weight Penalties ($L_1$, $L_2$, ElasticNet):** Evaluated $L_2$ weight decay ($\lambda=0.001$), $L_1$ lasso sparsity ($\lambda=0.001$), and ElasticNet ($L_1+L_2$).
- **Result:** $L_2$ regularization achieved **95.61% Accuracy** and **0.1278 Test Loss**, smoothly penalizing large weights to insulate against multicollinear cytological triplets (`radius`, `perimeter`, `area`).

### 5️⃣ Final Production Combined Model 🏆

- **Architecture:** `Dense(128, ReLU, L2)` → `Dropout(0.3)` → `Dense(64, ReLU, L2)` → `Dropout(0.3)` → `Dense(1, Sigmoid)`.
- **Training Strategy:** Optimized with Adam and `EarlyStopping(patience=20, restore_best_weights=True)`.
- **Result:** **95.61% Test Accuracy**, **0.1328 Test Loss**, **0.9855 Precision**, and **97.62% Malignant Sensitivity** (41 of 42 malignant cases correctly identified).
- **Strengths:** Maximum clinical stability, negligible overfitting, and robust defense-in-depth regularization.

---

## 📈 Evaluation & Comparison

| 🚀 Model Architecture             | 🔢 Parameters | 🛡️ Regularization | 🎲 Dropout | 🛑 Early Stopping | 🎯 Test Accuracy | 📉 Test Loss | 💎 Precision | ⚡ Recall | 📊 F1-Score |
| :-------------------------------- | :-----------: | :---------------: | :--------: | :---------------: | :--------------: | :----------: | :----------: | :-------: | :---------: |
| **Single-Layer Perceptron (SLP)** |      31       |       None        |    None    |    No (50 eps)    |      94.74%      |    0.1732    |    0.9459    |  0.9722   |   0.9589    |
| **MLP (ReLU Baseline)**           |     4,097     |       None        |    None    |   No (100 eps)    |    **96.49%**    |    0.1292    |  **0.9857**  |  0.9583   | **0.9718**  |
| **MLP + Early Stopping**          |    12,289     |       None        |    None    | Yes (patience=15) |    **96.49%**    |  **0.1272**  |  **0.9857**  |  0.9583   | **0.9718**  |
| **MLP without Early Stopping**    |    12,289     |       None        |    None    |   No (300 eps)    |      97.37%      |    0.2486    |    0.9859    |  0.9722   |   0.9790    |
| **MLP + Dropout (0.3)**           |    12,289     |       None        |    0.3     | Yes (patience=20) |      95.61%      |    0.1725    |    0.9855    |  0.9444   |   0.9645    |
| **MLP + L1 Regularization**       |    12,289     |   L1 (λ=0.001)    |    None    | Yes (patience=20) |    **96.49%**    |    0.1357    |  **0.9857**  |  0.9583   | **0.9718**  |
| **MLP + L2 Regularization**       |    12,289     |   L2 (λ=0.001)    |    None    | Yes (patience=20) |      95.61%      |    0.1278    |    0.9855    |  0.9444   |   0.9645    |
| **MLP + ElasticNet**              |    12,289     |  L1+L2 (0.0005)   |    None    | Yes (patience=20) |      95.61%      |    0.1424    |    0.9855    |  0.9444   |   0.9645    |
| **Final Production Combined**     |    12,289     |   L2 (λ=0.001)    |    0.3     | Yes (patience=20) |      95.61%      |    0.1328    |    0.9855    |  0.9444   |   0.9645    |

> [!TIP]
> **Final Production Combined Model** emerged as the clinical deployment winner! While raw unregularized models suffer severe late-epoch divergence (validation loss escalating from 0.089 to 0.2486), the combined model unites L2 weight decay, Dropout (0.3), and Early Stopping to provide robust generalization and **97.62% Malignant Sensitivity**! 🏆

---

## 💡 Clinical Intelligence & Decision Support

### 🏥 Clinical Triage Protocol & Decision Boundaries

| 🩺 Diagnostic Category             | 👤 Model Output Confidence                        | 🚀 Clinical Action Protocol                                                      |
| :--------------------------------- | :------------------------------------------------ | :------------------------------------------------------------------------------- |
| **High-Confidence Benign**         | $\hat{y} \ge 0.65$ ($P(\text{Malignant}) < 0.35$) | Routine screening follow-up; discharge from acute oncology concern.              |
| **Borderline / Ambiguous Nodules** | $0.40 \le \hat{y} < 0.65$                         | **Automated Flag for Review:** Second-look ultrasound & core needle biopsy.      |
| **High-Confidence Malignant**      | $\hat{y} < 0.40$ ($P(\text{Malignant}) \ge 0.60$) | **Immediate Oncology Triage:** Staging, surgical consultation, & histopathology. |

### 🎯 Key Clinical Takeaways

- **Non-Symmetric Error Costs:** In breast cancer screening, a **False Negative** (missing a malignancy) is potentially fatal. A **False Positive** causes transient anxiety and prompts confirmatory biopsy.
- **Threshold Calibration ($\tau = 0.65$):** Shifting the decision boundary from default 0.50 to 0.65 forces higher certainty before discharging a patient as benign, successfully driving Malignant Recall to **97.62%** (only 1 false negative on the 114-patient test set).

---

## 🛠️ Technical Deep Dive

### 💻 Tech Stack

- **Language:** 🐍 `Python 3.12`
- **Frameworks:** 🧠 `TensorFlow 2.21`, ⚡ `Keras 3.15`
- **Data:** 🐼 `Pandas`, 🔢 `NumPy`
- **ML & Metrics:** 🤖 `Scikit-Learn`
- **Viz:** 🎨 `Matplotlib`, 🌊 `Seaborn`
- **Env:** 📓 `Jupyter Notebook`

### 🚀 Getting Started

1. **Clone it:**
   ```bash
   git clone https://github.com/Prath-Digital/Deep_Learning_PR.-1-Breast-Cancer-Wisconsin.git
   cd Deep_Learning_PR.-1-Breast-Cancer-Wisconsin
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Analysis:**
   Launch the notebook and run all cells! 🚀
   ```bash
   jupyter notebook project.ipynb
   ```
4. **View Offline HTML:**
   Open [`project.html`](./project.html) in any browser for immediate interactive review.

---

## 🎬 Media & History

### 📸 Visualizations

The project includes publication-quality Seaborn & Matplotlib plots:  
✅ Diagnostic Class Distribution | ✅ 30×30 Correlation Heatmap | ✅ SLP Confusion Matrix | ✅ 1×3 Activation Loss Curves (ReLU/Tanh/Sigmoid) | ✅ Early Stopping U-Curve with Optimal Epoch Marker | ✅ Dropout Rate Comparison (0.1/0.3/0.5) | ✅ 1×3 Regularization Trajectories (L2/L1/ElasticNet) | ✅ Model Comparison Horizontal Bar Chart | ✅ Production Combined Learning Curves & Confusion Matrix

### 📹 Project Video

[🎥 Click here to watch the walkthrough!](./video.mp4)

### 📄 Executive Summary Report

[📊 Click here to read the full Executive Summary Report!](./summary_report.md)

---

### 📜 Git History

- ✨ `Add Task 1: Dataset exploration, distribution plots, and StandardScaler preprocessing`
- ✨ `Add Task 2: Single-Layer Perceptron baseline, BCE loss, and confusion matrix`
- ✨ `Add Task 3: Multi-Layer Perceptron architecture and ReLU vs Tanh vs Sigmoid activation benchmark`
- ✨ `Add Task 4: High-capacity MLP, 300-epoch overfit demonstration, and Early Stopping implementation`
- ✨ `Add Task 5: Dropout regularization ensembling and rate sensitivity analysis (0.1, 0.3, 0.5)`
- ✨ `Add Task 6: Weight penalties (L1, L2, ElasticNet) and unified loss trajectory evaluation`
- ✨ `Add Task 7: Production combined model, master comparison table, clinical decision report, and HTML export`

---

## ✅ Project Status

| 🧩 Component                                       | 🚦 Status    |
| :------------------------------------------------- | :----------- |
| Dataset Loading & Exploration                      | ✅ Completed |
| Stratified Split & StandardScaler Preprocessing    | ✅ Completed |
| Single-Layer Perceptron (Linear Baseline)          | ✅ Completed |
| Multi-Layer Perceptron & Activation Benchmark      | ✅ Completed |
| Early Stopping & Overfitting Prevention            | ✅ Completed |
| Dropout Regularization & Rate Sensitivity          | ✅ Completed |
| Weight Penalties (L1, L2, ElasticNet)              | ✅ Completed |
| Production Combined Model Architecture             | ✅ Completed |
| Full Results Comparison Table                      | ✅ Completed |
| Clinical Decision Support & Threshold Analysis     | ✅ Completed |
| Requirements & Documentation (`summary_report.md`) | ✅ Completed |
| Standalone HTML Export (`project.html`)            | ✅ Completed |

---

## 👤 Author & License

**Prath Udhnawala**  
🔗 [GitHub Profile](https://github.com/Prath-Digital)

---

_Built with ❤️ for Deep Learning Education._

**License:** This project is created for academic and educational purposes. The Breast Cancer Wisconsin (Diagnostic) dataset is sourced from the UCI Machine Learning Repository and is used in accordance with its open research license.
