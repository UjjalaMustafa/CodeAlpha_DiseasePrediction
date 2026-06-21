# 🏥 Disease Prediction from Medical Data
### CodeAlpha Machine Learning Internship — Task 4

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Accuracy](https://img.shields.io/badge/Best%20Accuracy-100%25-success)

---

## 📌 Project Overview

This project predicts the **possibility of diseases** — Heart Disease, Diabetes, Breast Cancer, or Healthy — from structured patient medical data.  
It is built as part of the **CodeAlpha Machine Learning Internship (Task 4)**.

**Approach:** Apply supervised classification techniques on structured medical data and compare multiple ML algorithms to find the best-performing model.

---

## 🎯 Objective

> Predict which disease (if any) a patient is likely to have based on their clinical measurements, lifestyle factors, and lab test results.

---

## 📂 Project Structure

```
disease_prediction/
│
├── disease_prediction.csv          # Dataset (1200 patients, 23 features)
├── Disease_Prediction_CodeAlpha.ipynb   # Full Jupyter Notebook
├── README.md                       # This file
└── (generated plots saved during notebook run)
    ├── target_distribution.png
    ├── age_bmi_distribution.png
    ├── boxplots_features.png
    ├── correlation_heatmap.png
    ├── categorical_analysis.png
    ├── model_comparison.png
    ├── confusion_matrix.png
    └── feature_importance.png
```

---

## 📊 Dataset Description

| Property | Details |
|---|---|
| **File** | `disease_prediction.csv` |
| **Rows** | 1200 patient records |
| **Features** | 23 (numerical + categorical) |
| **Target** | `diagnosis` — 4 classes |

### 🏷️ Target Classes

| Class | Description |
|---|---|
| ❤️ Heart Disease | Cardiovascular conditions |
| 🩸 Diabetes | Blood sugar / insulin disorders |
| 🎗️ Breast Cancer | Oncological condition |
| ✅ Healthy | No significant disease |

### 📋 Features Used

| Feature | Type | Description |
|---|---|---|
| `age` | Numerical | Patient age (years) |
| `gender` | Categorical | Male / Female |
| `bmi` | Numerical | Body Mass Index |
| `blood_pressure_systolic` | Numerical | Systolic BP (mmHg) |
| `blood_pressure_diastolic` | Numerical | Diastolic BP (mmHg) |
| `cholesterol` | Numerical | Total cholesterol (mg/dL) |
| `fasting_blood_glucose` | Numerical | Fasting glucose (mg/dL) |
| `hba1c` | Numerical | Glycated hemoglobin (%) |
| `insulin` | Numerical | Insulin level |
| `smoking_status` | Categorical | Never / Former / Current |
| `alcohol_use` | Categorical | None / Moderate / Heavy |
| `physical_activity` | Categorical | Low / Moderate / High |
| `family_history` | Binary | 1 = Yes, 0 = No |
| `chest_pain_type` | Ordinal | 0–3 scale |
| `resting_ecg` | Ordinal | ECG result (0–2) |
| `max_heart_rate` | Numerical | Max heart rate achieved |
| `exercise_induced_angina` | Binary | 1 = Yes, 0 = No |
| `st_depression` | Numerical | ST depression on ECG |
| `white_blood_cells` | Numerical | WBC count |
| `red_blood_cells` | Numerical | RBC count |
| `hemoglobin` | Numerical | Hemoglobin (g/dL) |
| `creatinine` | Numerical | Creatinine level |

---

## ⚙️ Installation & Setup

### 1. Clone / Download the project
```bash
git clone https://github.com/YourUsername/CodeAlpha_ProjectName.git
cd CodeAlpha_ProjectName/disease_prediction
```

### 2. Install required libraries
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 3. Launch Jupyter Notebook
```bash
jupyter notebook Disease_Prediction_CodeAlpha.ipynb
```

> **Tip:** Run all cells top-to-bottom with `Kernel > Restart & Run All`

---

## 🧠 ML Pipeline

```
Raw Data
    ↓
Exploratory Data Analysis (EDA)
    ↓
Preprocessing
 • Drop patient_id
 • Label Encode categoricals
 • StandardScaler for numerical features
 • Stratified Train/Test Split (80/20)
    ↓
Model Training + 5-Fold Cross Validation
    ↓
Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
    ↓
Best Model Selection
    ↓
Predict on New Patient
```

---

## 🤖 Models Compared

| Model | CV Accuracy | Test Accuracy |
|---|---|---|
| 🥇 Random Forest | ~99.79% | **100.00%** |
| 🥈 Gradient Boosting | ~99.48% | **100.00%** |
| 🥉 Logistic Regression | ~98.85% | 99.58% |
| SVM (RBF) | ~98.23% | 99.58% |
| KNN (k=7) | ~95.62% | 95.83% |

**Best Model: Random Forest Classifier** with `n_estimators=200`

---

## 📈 Key Findings

- **Cholesterol, Blood Glucose & HbA1c** are the most powerful predictors of disease
- **BMI + Age** together clearly separate disease risk groups
- **Family history** and **exercise-induced angina** are strong indicators for Heart Disease
- **HbA1c > 6.5** is the defining marker for Diabetes prediction
- **Ensemble methods** (Random Forest, Gradient Boosting) significantly outperform single models

---

## 🔮 How to Predict for a New Patient

```python
sample_patient = {
    'age': 55,
    'gender': 'Male',
    'bmi': 28.5,
    'blood_pressure_systolic': 145,
    'blood_pressure_diastolic': 92,
    'cholesterol': 245,
    'fasting_blood_glucose': 130,
    'hba1c': 6.8,
    'insulin': 15.0,
    'smoking_status': 'Former',
    'alcohol_use': 'Moderate',
    'physical_activity': 'Low',
    'family_history': 1,
    'chest_pain_type': 2,
    'resting_ecg': 1,
    'max_heart_rate': 140,
    'exercise_induced_angina': 1,
    'st_depression': 1.5,
    'white_blood_cells': 7.2,
    'red_blood_cells': 4.8,
    'hemoglobin': 14.2,
    'creatinine': 1.1,
}

result = predict_disease(sample_patient)
# Output: 🩺 Prediction: Heart Disease
```

---

## 📚 Libraries Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading & manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `matplotlib` | Plotting charts |
| `seaborn` | Statistical visualizations |

---

## 📌 Internship Details

| | |
|---|---|
| **Organization** | CodeAlpha |
| **Domain** | Machine Learning |
| **Task** | Task 4 — Disease Prediction from Medical Data |
| **GitHub Repo** | `CodeAlpha_ProjectName` |

---

## 👤 Author

**Ujjala Mustafa**  
Machine Learning Intern @ CodeAlpha  
🔗 [LinkedIn](https://linkedin.com) | 💻 [GitHub](https://github.com)

---


