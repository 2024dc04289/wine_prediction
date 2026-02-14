# ML Assignment 2 - Classification Models

## Problem Statement

The objective of this project is to build and evaluate multiple machine learning classification models to predict wine quality. Given physicochemical properties of red wine samples, the task is to classify whether a wine is of **high quality** (quality score ≥ 6) or **low quality** (quality score < 6).

This is a binary classification problem that demonstrates the implementation, evaluation, and comparison of 6 different ML algorithms.

---

## Dataset Description

**Dataset:** Wine Quality Dataset (Red Wine)  
**Source:** UCI Machine Learning Repository  
**Link:** https://archive.ics.uci.edu/ml/datasets/wine+quality

### Dataset Characteristics:
| Attribute | Value |
|-----------|-------|
| **Number of Instances** | 1,599 |
| **Number of Features** | 11 |
| **Target Variable** | Binary (High Quality ≥ 6, Low Quality < 6) |
| **Missing Values** | None |

### Features Description:

| Feature | Description | Type |
|---------|-------------|------|
| fixed acidity | Tartaric acid concentration (g/dm³) | Continuous |
| volatile acidity | Acetic acid concentration (g/dm³) | Continuous |
| citric acid | Citric acid concentration (g/dm³) | Continuous |
| residual sugar | Sugar remaining after fermentation (g/dm³) | Continuous |
| chlorides | Sodium chloride concentration (g/dm³) | Continuous |
| free sulfur dioxide | Free form of SO₂ (mg/dm³) | Continuous |
| total sulfur dioxide | Total SO₂ (mg/dm³) | Continuous |
| density | Density of wine (g/cm³) | Continuous |
| pH | pH value (0-14 scale) | Continuous |
| sulphates | Potassium sulphate concentration (g/dm³) | Continuous |
| alcohol | Alcohol percentage (% vol) | Continuous |

### Class Distribution:
- **Low Quality (0):** ~855 samples (53.5%)
- **High Quality (1):** ~744 samples (46.5%)

---

## Models Used

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.7406 | 0.8242 | 0.7419 | 0.7406 | 0.7409 | 0.4808 |
| Decision Tree | 0.7531 | 0.7513 | 0.7529 | 0.7531 | 0.7529 | 0.5034 |
| K-Nearest Neighbor | 0.7406 | 0.8117 | 0.7407 | 0.7406 | 0.7407 | 0.4790 |
| Naive Bayes | 0.7219 | 0.7884 | 0.7282 | 0.7219 | 0.7219 | 0.4500 |
| Random Forest (Ensemble) | 0.8031 | 0.9020 | 0.8043 | 0.8031 | 0.8033 | 0.6062 |
| XGBoost (Ensemble) | 0.8250 | 0.8963 | 0.8259 | 0.8250 | 0.8252 | 0.6497 |

---

## Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Provides a solid baseline with 74.06% accuracy. As a linear model, it performs well on linearly separable patterns and offers high interpretability. Fast training time and efficient for production deployment. Limited by its linear decision boundary assumption. |
| **Decision Tree** | Achieves 75.31% accuracy with excellent interpretability through tree visualization. Captures non-linear relationships in the data. However, prone to overfitting without proper pruning. High variance between different data splits. |
| **K-Nearest Neighbor** | Achieves 74.06% accuracy. Performance is sensitive to the choice of K and feature scaling. Computationally expensive during prediction for large datasets. Works well when similar wines have similar quality. |
| **Naive Bayes** | Achieves 72.19% accuracy due to the independence assumption between features. Wine characteristics are often correlated (e.g., pH and acidity), partially violating this assumption. However, extremely fast training and handles small datasets well. |
| **Random Forest (Ensemble)** | Strong performer with 80.31% accuracy and highest AUC of 0.902. Ensemble approach reduces overfitting compared to single Decision Tree. Provides feature importance rankings showing alcohol content as most predictive. Robust to outliers. |
| **XGBoost (Ensemble)** | Best performer with 82.50% accuracy and AUC of 0.8963. Gradient boosting captures complex patterns effectively. Built-in regularization prevents overfitting. Highest MCC score (0.6497) indicates best balanced classification performance. |

### Key Observations:

1. **Best Overall Model:** XGBoost achieves the highest accuracy (82.50%) and MCC (0.6497), making it the recommended model for this classification task.

2. **Ensemble Methods Outperform:** Both Random Forest and XGBoost significantly outperform individual models, demonstrating the power of ensemble learning.

3. **Trade-off: Interpretability vs Performance:** Logistic Regression and Decision Tree offer better interpretability but lower performance compared to ensemble methods.

4. **Feature Importance:** Alcohol content, volatile acidity, and sulphates are the most important features for predicting wine quality across all models.

5. **Class Imbalance:** The slight class imbalance (53.5% vs 46.5%) doesn't significantly impact model performance, as evidenced by balanced precision and recall scores.

---

## Project Structure

```
ML_Assignment_2/
│
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation (this file)
│
└── model/
    └── train_models.py         # Model training script
```

---

## How to Run Locally

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/ML_Assignment_2.git
cd ML_Assignment_2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

4. Open your browser and navigate to `http://localhost:8501`

---

## Streamlit App Features

1. **Dataset Upload Option (CSV):** Upload your own test data in CSV format
2. **Model Selection Dropdown:** Choose from 6 different classification models
3. **Display of Evaluation Metrics:** View Accuracy, AUC, Precision, Recall, F1, and MCC scores
4. **Confusion Matrix:** Visual representation of model predictions vs actual values
5. **Classification Report:** Detailed precision, recall, and F1 scores per class
6. **All Models Comparison:** Side-by-side comparison of all 6 models

---

## Deployment

The application is deployed on Streamlit Community Cloud.

**Live App Link:** [Your Streamlit App URL Here]

---

## Technologies Used

- **Python 3.8+**
- **Scikit-learn** - ML algorithms and metrics
- **XGBoost** - Gradient boosting classifier
- **Pandas & NumPy** - Data manipulation
- **Matplotlib & Seaborn** - Visualizations
- **Streamlit** - Web application framework

---

## Author

**Name:** Harikrishna M
**BITS ID:** 2024dc04289

---