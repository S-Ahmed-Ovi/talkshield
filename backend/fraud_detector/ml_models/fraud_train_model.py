import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("🔹 Loading dataset...")
data_path = os.path.join(BASE_DIR, "creditcard.csv")

if not os.path.exists(data_path):
    print("❌ Dataset not found! Please place 'creditcard.csv' in:", data_path)
    exit()

df = pd.read_csv(data_path)
print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")

X = df.drop(columns=['Class'])
y = df['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("🔹 Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f"✅ Model trained with accuracy: {acc*100:.2f}%")

joblib.dump(model, os.path.join(BASE_DIR, "fraud_model.pkl"))
joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.pkl"))

print("💾 Model and scaler saved successfully!")
