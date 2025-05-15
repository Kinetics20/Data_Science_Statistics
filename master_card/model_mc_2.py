import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# 1. Load user and merchant data
users = pd.read_csv('users.csv')
merchants = pd.read_csv('merchants.csv')

# 2. Load transaction JSON file (simulate – read part of the data)
transactions = []
with open('transactions.json', 'r') as f:
    for i, line in enumerate(f):
        transactions.append(json.loads(line))
        if i >= 100_000:  # load only 100k transactions
            break

transactions_df = pd.json_normalize(transactions)

# 3. Merge datasets
df = transactions_df.merge(users, on='user_id', how='left')
df = df.merge(merchants, on='merchant_id', how='left')

print("\n📊 Class proportions:")
print(df['is_fraud'].value_counts(normalize=True).rename("proportion"))

# 4. Data cleaning
df = df.dropna()  # simplification

# 5. Encode categorical variables
cat_cols = ['payment_method', 'country_x', 'country_y']
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# 6. Prepare features and label
feature_cols = [
    'amount', 'payment_method', 'age', 'trust_score',
    'risk_score', 'number_of_alerts_last_6_months',
    'avg_transaction_amount', 'account_age_months'
]
X = df[feature_cols]
y = df['is_fraud']

# 7. Balance classes using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 8. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# 9. Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 10. Evaluate the model
y_pred = clf.predict(X_test)
print("\n🧠 Classification Report:")
print(classification_report(y_test, y_pred))

# 11. Visualize confusion matrix
ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)  # Save the plot to a file
plt.show()
