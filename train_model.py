import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
df.drop("Loan_ID", axis=1, inplace=True)

cat_cols = df.select_dtypes(include="object").columns
num_cols = df.select_dtypes(include=["int64", "float64"]).columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
print("✅ Accuracy:", accuracy_score(y_val, y_pred))
print("📋 Report:\n", classification_report(y_val, y_pred))

with open("model.pkl", "wb") as f:
    pickle.dump((model, encoders), f)