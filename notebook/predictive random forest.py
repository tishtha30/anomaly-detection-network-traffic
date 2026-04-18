import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score, f1_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cicids2018_sample.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "predictive")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv("D:/CU/Dissertation/Final DS Project/data/processed/cicids2018_processed.csv")

X = df.drop(columns=["Label", "is_attack"])
y = df["is_attack"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)

train_pred = model.predict(X_train)
pred = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, pred)

train_f1 = f1_score(y_train, train_pred)
test_f1 = f1_score(y_test, pred)

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)
print("Train F1:", train_f1)
print("Test F1:", test_f1)

cm = confusion_matrix(y_test, pred)
report = classification_report(y_test, pred)
roc = roc_auc_score(y_test, probs)

print("\n=== Random Forest Results ===")
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)

print("ROC-AUC:", roc)

with open(os.path.join(OUTPUT_DIR, "random forest results.txt"), "w") as f:
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")
    f.write("ROC-AUC:\n")
    f.write(str(roc))

joblib.dump(model, os.path.join(OUTPUT_DIR, "random_forest_model.pkl"))