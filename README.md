# 💳 Credit Card Fraud Detection – Recall-Focused Pipeline (In Progress)

This project detects fraudulent transactions using the [Kaggle credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), which contains highly imbalanced data — with fraud accounting for only ~0.17% of all records.

---

## 🎯 Objective

To build a machine learning pipeline that **maximizes recall**, catching as many fraud cases as possible.

---

## ⚙️ Approach Summary

Rather than oversampling or generating synthetic data, this project uses:
- `class_weight='balanced'` in **Logistic Regression**
- `scale_pos_weight` in **XGBoost**

These approaches adjust model focus without introducing noise or overfitting risks.

---

## 🛠️ Features Engineered

| Feature           | Description                                         |
|------------------|-----------------------------------------------------|
| `Hour`           | Hour of transaction (0–23)                          |
| `TimeBucket`     | 6-hour time segments (e.g., 0–6 AM)                 |
| `IsNight`        | Binary flag for night hours (22:00–06:00)           |
| `TimeSinceLastTx`| Time delta since the previous transaction           |
| `TxInPastHour`   | Count of transactions in the past hour (rolling)    |
| `DayPart`        | Categorical: Morning, Afternoon, Evening, Night     |

PCA features `V1–V28`, `Time`, and `Amount` were also used as-is.

---

## 🧪 Modeling Pipeline

1. **EDA & Feature Engineering**
   - Visual patterns in fraud by time
   - Bimodal distribution in transaction frequency
   - Nighttime fraud concentration observed

2. **Train/Test Split**
   - Stratified 70/30 split
   - Scaled with `StandardScaler`

3. **Model Training**
   - ✅ Logistic Regression (`class_weight='balanced'`)
   - ✅ XGBoost (`scale_pos_weight`)
   - ✅ Isolation Forest (unsupervised)
   - ✅ Ensemble: Logistic Regression + XGBoost

4. **Evaluation Metrics**
   - Recall (priority)
   - Precision, F1 Score, ROC AUC
   - Confusion Matrices
   - ROC & Precision-Recall curves

---

## 📈 Key Results

| Model              | Precision | Recall | F1 Score | ROC AUC |
|-------------------|-----------|--------|----------|---------|
| Logistic Regression | 6.3%     | **87.8%** | 0.12     | 0.9666  |
| XGBoost             | **87.8%** | 77.7%  | **0.83** | 0.9654  |
| Isolation Forest    | 29.7%    | 29.7%  | 0.30     | 0.6480  |

---

## ✅ Project Milestones Completed

- [x] Data loading and exploration
- [x] Handling class imbalance (SMOTE, class weights)
- [x] Feature engineering (time-based and PCA features)
- [x] Correlation analysis and visualization
- [x] Model training (Logistic Regression, XGBoost, Isolation Forest)
- [x] Model evaluation (Confusion Matrix, ROC AUC, F1, Recall)
- [x] Threshold tuning using Precision-Recall curve
- [x] Ensemble model with soft voting (LR + XGBoost)
- [x] Interpretability with XGBoost feature importances


---

## 📁 Project Structure

credit-card-fraud-detection/
├── data/ # Input dataset (e.g., creditcard.csv)
├── notebooks/ # Jupyter notebooks (EDA, modeling)
├── outputs/ # Visualizations, metrics, exports
├── credit-card-fraud-detection.ipynb
├── README.md # Project overview
└── LICENSE # License file

---

## 📚 Dataset

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Type:** PCA-transformed numeric features + engineered time-based features
- **Imbalance:** 492 frauds out of 284,807 transactions (~0.17%)

---

## ✍️ Author

**Gleidy R.**  
[LinkedIn](https://www.linkedin.com/in/gleidyalonzo)

---

Feel free to fork, contribute, or share feedback!
