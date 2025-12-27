# Fraud Detection for E-commerce and Bank Transactions
## Project Overview
This project focuses on building robust machine learning models to detect fraudulent transactions in both **e-commerce** and **banking (credit card)** datasets. Fraud detection presents unique challenges, particularly **severe class imbalance**, high-cardinality identifiers, and the need to balance **security** with **customer experience**. This project addresses these challenges using careful feature engineering, appropriate evaluation metrics, and model explainability techniques.
---
## Business Problem
Adey Innovations Inc., a fintech company specializing in e-commerce and banking solutions, aims to improve fraud detection accuracy while minimizing false positives.
Key objectives:
- Detect fraudulent transactions accurately
- Handle highly imbalanced datasets
- Reduce financial losses due to fraud
- Maintain good user experience
- Provide interpretable and explainable model outputs
---
## Datasets Used
### 1. Fraud_Data.csv (E-commerce Transactions)
- `user_id`
- `signup_time`
- `purchase_time`
- `purchase_value`
- `device_id`
- `source`
- `browser`
- `sex`
- `age`
- `ip_address`
- `class` (target: 1 = fraud, 0 = non-fraud)
**Challenge:** Highly imbalanced target variable.
### 2. IpAddress_to_Country.csv
- `lower_bound_ip_address`
- `upper_bound_ip_address`
- `country`
Used to enrich transactions with geolocation information.
### 3. creditcard.csv (Bank Transactions)
- `Time`
- `V1` – `V28` (PCA-transformed features)
- `Amount`
- `Class` (target)
**Challenge:** Extreme class imbalance.
---
## Used Structure
```
.
├── .vscode/
│   ├── settings.json
├── .github/
│   ├── workflows/
│       ├── unittests.yml
├── README.md
├── data/
│   ├── raw/            # Original datasets
│   ├── processed/      # Cleaned and feature-engineered data
├── models/              # Saved model artifacts
├── notebooks/
│   ├── eda-creditcard.ipynb
│   ├── eda-fraud-data.ipynb
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb
│   ├── shap-explainability.ipynb
│   ├── README.md
├── src/
│   ├── __init__.py
├── scripts/
│   ├── __init__.py
│   ├── README.md
├── .gitignore
├── requirements.txt
├── README.md

```

---

## Methodology

### 1. Data Cleaning & Preprocessing

- Removed duplicates

- Converted timestamps to datetime

### 2. Exploratory Data Analysis (EDA)

- Univariate and bivariate analysis

- Fraud vs non-fraud distribution

- Analysis of fraud patterns across countries, time, and transaction value

### 3. Feature Engineering

- Time-based features:

  - `hour_of_day`

  - `day_of_week`

  - `time_since_signup`

- Transaction behavior features

- IP-to-country mapping using range-based lookup

- One-hot encoding for low-cardinality categorical features

### 4. Handling Class Imbalance

- Used:

  - `class_weight='balanced'` for Logistic Regression

  - Class weighting for Random Forest

- SMOTE applied only on training data where appropriate

- Evaluated class distribution before and after resampling

---

## Modeling

### Baseline Model

- **Logistic Regression**

- Interpretable baseline

- Evaluated using:

  - F1-score

  - AUC-PR

  - Confusion Matrix

### Ensemble Model

- **Random Forest**

- Captures non-linear fraud patterns

- Basic hyperparameter tuning (`n_estimators`, `max_depth`)

- Stratified 5-fold cross-validation

## Evaluation Metrics

Given the imbalanced nature of fraud data:

- **AUC-PR (Precision-Recall AUC)** was prioritized over accuracy

- **F1-score** used to balance precision and recall

- **Confusion Matrix** analyzed for false positives and false negatives

