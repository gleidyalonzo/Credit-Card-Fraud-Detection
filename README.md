# 💳 Credit Card Fraud Detection – Recall-Focused Pipeline (In Progress)

This project focuses on detecting fraudulent transactions from the popular [Kaggle credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The dataset is highly imbalanced, with less than 0.2% fraud cases.

## 🎯 Objective

To develop a robust fraud detection model that **maximizes recall**, ensuring the system catches as many fraudulent transactions as possible — even at the expense of overall accuracy.

## ⚖️ Why Class Weights?

Instead of oversampling with SMOTE, this project leverages the `class_weight='balanced'` parameter in algorithms like **Random Forest** and **XGBoost** to directly address class imbalance. This avoids the risk of overfitting to synthetic data and helps the model focus on the rare fraud class.

---

## 🛠️ Features Engineered

- `Hour`: Hour of transaction derived from the `Time` feature
- `TxInPastHour`: Number of transactions in the past hour (rolling window)
- Additional PCA-transformed features (`V1`–`V28`) are included as-is

---

## 📊 Current Pipeline Steps

1. **Data Preprocessing**
   - One-hot encoding of any categorical features
   - Median/mode imputation of missing values
2. **Train/Test Split** (Stratified)
3. **Modeling with Class Weighting**
   - `RandomForestClassifier(class_weight='balanced')`
   - `XGBoost` with `scale_pos_weight`
4. **Evaluation Metrics**
   - **Recall** (primary)
   - Precision, F1-score
   - Confusion Matrix
   - ROC AUC

---

## 🚧 Status

- [x] Data cleaning and feature engineering
- [x] Modeling with Random Forest (in progress)
- [x] Modeling with XGBoost (in progress)
- [ ] Hyperparameter tuning (coming next)
- [ ] Cross-validation with StratifiedKFold
- [ ] Precision-Recall curve visualization

---

## 📁 File Structure

credit-card-fraud-detection/
├── data/ # Contains input dataset (e.g., creditcard.csv)
├── notebooks/ # Jupyter notebooks with experiments and EDA
├── outputs/ # Generated visualizations, model metrics, exports
├── .gitattributes # Git config
├── LICENSE # License file
├── README # This project documentation

## 📚 Dataset

- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## ✍️ Author

Gleidy R. | [https://www.linkedin.com/in/gleidyalonzo]


