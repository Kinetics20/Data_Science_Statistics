import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# 1. Load user and merchant data
users = pd.read_csv('users.csv')
merchants = pd.read_csv('merchants.csv')

# 2. Load transaction JSON file (simulate – read a subset of data)
transactions = []
with open('transactions.json', 'r') as f:
    for i, line in enumerate(f):
        transactions.append(json.loads(line))
        if i >= 100000:  # read only a portion of the data, e.g., 100k transactions
            break

transactions_df = pd.json_normalize(transactions)

# 3. Merge data
df = transactions_df.merge(users, left_on='user_id', right_on='user_id', how='left')
df = df.merge(merchants, left_on='merchant_id', right_on='merchant_id', how='left')

print(df['is_fraud'].value_counts(normalize=True))

# 4. Example cleaning / encoding
df = df.dropna()  # simplification: remove missing values

# Encode categorical columns (e.g., payment method, location)
# print(df.columns.tolist())
cat_cols = ['payment_method', 'country_x', 'country_y']
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# 5. Prepare features and target
feature_cols = ['amount', 'payment_method', 'age', 'trust_score']
X = df[feature_cols]
y = df['is_fraud']  # assume this column contains the target labels

# Balance the data
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 6. Train the model
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 7. Evaluation
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
