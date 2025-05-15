import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report

from imblearn.over_sampling import SMOTE



# Wczytaj dane

df = pd.read_csv("master_cleared.csv")



# Usuń zbędne kolumny

df = df.drop(columns=['Unnamed: 0', 'transaction_id', 'timestamp', 'user_id', 'merchant_id', 'signup_date'])



# Zakoduj kolumny kategoryczne

categorical_cols = df.select_dtypes(include='object').columns

le = LabelEncoder()

for col in categorical_cols:

    df[col] = le.fit_transform(df[col])



# Podział na X i y

X = df.drop(columns='is_fraud')

y = df['is_fraud']



# Podział na zbiór treningowy i testowy

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Zbalansowanie danych za pomocą SMOTE

smote = SMOTE(random_state=42)

X_resampled, y_resampled = smote.fit_resample(X_train, y_train)



# Trening modelu z class_weight

model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

model.fit(X_resampled, y_resampled)



# Predykcja

y_pred = model.predict(X_test)



# Ocena modelu



print(classification_report(y_test, y_pred))