import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cicids2018_sample.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "predictive")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv("D:/CU/Dissertation/Final DS Project/data/processed/cicids2018_processed.csv")

X = df.drop(columns=["Label", "is_attack"])
y = df["is_attack"]

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
)

model.fit(X_train, y_train)

train_pred = model.predict(X_train)
pred = model.predict(X_test)

proba = model.predict_proba(X_test)[:, 1]

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, pred)

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)
train_f1 = f1_score(y_train, train_pred)
test_f1 = f1_score(y_test, pred)

print("Train F1:", train_f1)
print("Test F1:", test_f1)

roc = roc_auc_score(y_test, proba)
cm = confusion_matrix(y_test, pred)
report_txt = classification_report(y_test, pred)

print("\n=== XGBoost Results ===")
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", report_txt)
print("ROC-AUC:", roc)

with open(os.path.join(OUTPUT_DIR, "xgboost results.txt"), "w") as f:
    f.write("XGBoost Results\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")
    f.write("Classification Report:\n")
    f.write(report_txt + "\n")
    f.write("ROC-AUC:\n")
    f.write(str(roc) + "\n")

joblib.dump(model, os.path.join(OUTPUT_DIR, "xgboost model.pkl"))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "xgboost scaler.pkl"))