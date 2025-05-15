import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import numpy as np

# 1. Load users and merchants data
users = pd.read_csv('users.csv')
merchants = pd.read_csv('merchants.csv')

# 2. Load JSON transactions (simulate partial load)
transactions = []
with open('transactions.json', 'r') as f:
    for i, line in enumerate(f):
        transactions.append(json.loads(line))
        if i >= 100_000:
            break

transactions_df = pd.json_normalize(transactions)

# 3. Merge all data
df = transactions_df.merge(users, on='user_id', how='left')
df = df.merge(merchants, on='merchant_id', how='left')

print("\n📊 Class proportions:")
print(df['is_fraud'].value_counts(normalize=True).rename("proportion"))

# 4. Drop missing values (simplified)
df = df.dropna()

# 5. Feature engineering
# Add difference in countries as a flag feature
df['country_mismatch'] = (df['country_x'] != df['country_y']).astype(int)
# Hour of transaction
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['transaction_hour'] = df['timestamp'].dt.hour

# 6. Define features and target
cat_cols = ['payment_method', 'country_x', 'country_y']
num_cols = [
    'amount', 'age', 'trust_score', 'risk_score',
    'number_of_alerts_last_6_months',
    'avg_transaction_amount', 'account_age_months',
    'transaction_hour', 'country_mismatch'
]
feature_cols = cat_cols + num_cols
X = df[feature_cols]
y = df['is_fraud']

# 7. Define preprocessing and pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
], remainder='passthrough')

# 8. Balance data using SMOTE
X_processed = preprocessor.fit_transform(X)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_processed, y)

# 9. Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# 10. Train improved model with XGBoost
# clf = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
clf = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, eval_metric='logloss')

clf.fit(X_train, y_train)

# 11. Evaluate model
y_pred = clf.predict(X_test)
print("\n🧠 Classification Report:")
print(classification_report(y_test, y_pred))

# 12. Confusion matrix
ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_improved.png", dpi=300)
plt.show()

# 13. Feature importance plot (optional, if desired)
importances = clf.feature_importances_
feature_names = preprocessor.get_feature_names_out()
sorted_idx = np.argsort(importances)[::-1][:15]
sns.barplot(x=importances[sorted_idx], y=np.array(feature_names)[sorted_idx])
plt.title("Top 15 Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.show()
