import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# 1. Wczytanie danych użytkowników i sprzedawców
users = pd.read_csv('users.csv')
merchants = pd.read_csv('merchants.csv')

# 2. Wczytanie pliku JSON transakcji (symulacja – odczyt części danych)
transactions = []
with open('transactions.json', 'r') as f:
    for i, line in enumerate(f):
        transactions.append(json.loads(line))
        if i >= 100000:  # wczytaj tylko część danych, np. 100k transakcji
            break

transactions_df = pd.json_normalize(transactions)


# 3. Łączenie danych
df = transactions_df.merge(users, left_on='user_id', right_on='user_id', how='left')
df = df.merge(merchants, left_on='merchant_id', right_on='merchant_id', how='left')

print(df['is_fraud'].value_counts(normalize=True))

# 4. Przykładowe czyszczenie / kodowanie
df = df.dropna()  # uproszczenie: usuwamy brakujące wartości

# Kodowanie kategorii (np. metoda płatności, lokalizacja)
# print(df.columns.tolist())
cat_cols = ['payment_method', 'country_x', 'country_y']
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# 5. Przygotowanie cech i etykiety
feature_cols = ['amount', 'payment_method', 'age', 'trust_score']
X = df[feature_cols]
y = df['is_fraud']  # zakładamy, że etykieta znajduje się w tej kolumnie

# zbalansowanie danych

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 6. Trenowanie modelu
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 7. Ewaluacja
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
