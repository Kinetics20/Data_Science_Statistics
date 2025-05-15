import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# 1. Load users and merchants data
users = pd.read_csv('users.csv')
merchants = pd.read_csv('merchants.csv')

# 2. Load part of the transactions JSON file
transactions = []
with open('transactions.json', 'r') as f:
    for i, line in enumerate(f):
        transactions.append(json.loads(line))
        if i >= 100_000:
            break

transactions_df = pd.json_normalize(transactions)

# 3. Merge datasets
df = transactions_df.merge(users, on='user_id', how='left')
df = df.merge(merchants, on='merchant_id', how='left')

print("\n📊 Class proportions:")
print(df['is_fraud'].value_counts(normalize=True).rename("proportion"))

# 4. Clean data
df = df.dropna()

# 5. Encode categorical columns
cat_cols = ['payment_method', 'country_x', 'country_y']
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# 6. Feature selection
feature_cols = [
    'amount', 'payment_method', 'age', 'trust_score',
    'risk_score', 'number_of_alerts_last_6_months',
    'avg_transaction_amount', 'account_age_months'
]
X = df[feature_cols]
y = df['is_fraud']

# 7. Train/test split BEFORE SMOTE (lepsza ewaluacja)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Pipeline with SMOTE, scaling, and XGBoost
pipeline = Pipeline(steps=[
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('xgb', XGBClassifier(eval_metric='logloss', random_state=42))
])

# 9. Parametry do GridSearch
param_grid = {
    'xgb__n_estimators': [100, 200],
    'xgb__max_depth': [3, 5, 7],
    'xgb__learning_rate': [0.05, 0.1, 0.2],
    'xgb__subsample': [0.8, 1.0]
}

grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring='f1', verbose=1)
grid.fit(X_train, y_train)

# 10. Evaluate best model
y_pred = grid.predict(X_test)
print("\n🔍 Best params:", grid.best_params_)
print("\n🧠 Classification Report:")
print(classification_report(y_test, y_pred))

# 11. Confusion Matrix
ConfusionMatrixDisplay.from_estimator(grid.best_estimator_, X_test, y_test)
plt.title("Confusion Matrix (Tuned XGBoost)")
plt.tight_layout()
plt.savefig("confusion_matrix_xgb_tuned.png", dpi=300)
plt.show()
