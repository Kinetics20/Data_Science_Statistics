
# SGH x Mastercard Hackathon - May 2025

![Hackathon Logo](https://github.com/Kinetics20/Data_Science_Statistics/blob/main/master_card/hackaton_logo_2025.png)

## 📝 Project Summary

This project was developed as part of the **SGH x Mastercard Hackathon – May 2025**. The task was to **develop a fraud detection model** based on transactional data, including information about users and merchants. The challenge involved handling imbalanced data and optimizing classification performance.

Official Kaggle competition:
🔗 [SGH x Mastercard Hackathon – May 2025](https://www.kaggle.com/competitions/sgh-x-mastercard-hackathon-may-2025/overview)

### Key Objectives:

* Build a model that predicts the likelihood of fraud (`is_fraud`) based on transactional, user, and merchant data.
* Deal with significant data imbalance.
* Improve precision and recall for both fraud and non-fraud classes.

---

## 📁 Dataset Overview

The following data files were used:

* `transactions.json` – transaction records with timestamp and amount
* `users.csv` – user profile information
* `merchants.csv` – merchant profile information

Data processing included merging datasets, creating new features (e.g., `country_mismatch`, `transaction_hour`), and applying SMOTE for balancing classes.

---

## 🧪 Model Pipeline (`model_mc_3.py`)

### Tools & Libraries Used:

* `pandas`, `numpy`, `json` – data loading and manipulation
* `matplotlib`, `seaborn` – visualization
* `scikit-learn` – preprocessing, splitting, metrics
* `imblearn.SMOTE` – oversampling to balance classes
* `XGBoost (XGBClassifier)` – gradient boosting classifier

### Feature Engineering:

* Extracted hour from timestamps
* Created mismatch flags between user and merchant countries
* Encoded categorical variables using `OneHotEncoder`
* Balanced dataset using **SMOTE**
* Used `XGBoost` for final model training and evaluation

---

## 📊 Class Balance Before Training

```
is_fraud
0    0.914871
1    0.085129
```

---

## 🧠 Final Model Performance

```
              precision    recall  f1-score   support

           0       0.92      1.00      0.96     14731
           1       1.00      0.91      0.95     14731

    accuracy                           0.95     29462
   macro avg       0.96      0.95      0.95     29462
weighted avg       0.96      0.95      0.95     29462
```

---

## 📉 Confusion Matrix

![Confusion Matrix](https://github.com/Kinetics20/Data_Science_Statistics/blob/main/master_card/confusion_matrix_improved.png)

---

## 📌 Top 15 Feature Importances

![Feature Importance](https://github.com/Kinetics20/Data_Science_Statistics/blob/main/master_card/feature_importance.png)

---

## ✅ Summary

This project demonstrated an end-to-end pipeline for fraud detection using real-world-style transaction data. The model achieved strong performance, with an overall accuracy of **95%** and balanced precision/recall for both classes.

**Note:**
This project was created for testing and learning purposes in the context of the SGH x Mastercard Hackathon 2025.


