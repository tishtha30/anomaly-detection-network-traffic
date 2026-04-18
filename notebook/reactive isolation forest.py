import os
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import RobustScaler


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cicids2018_sample.csv")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "reactive")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv("D:/CU/Dissertation/Final DS Project/data/processed/cicids2018_processed.csv")

X = df.drop(columns=["Label", "is_attack"])
y = df["is_attack"]

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

X_train = X_scaled[y == 0]
X_test = X_scaled
y_test = y

contamination = 0.20

model = IsolationForest(
    n_estimators=200,
    contamination=contamination,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train)

pred = model.predict(X_test)
pred = np.where(pred == -1, 1, 0)

scores = -model.decision_function(X_test)

roc = roc_auc_score(y_test, scores)
cm = confusion_matrix(y_test, pred)
report_txt = classification_report(y_test, pred)

print("\n=== Isolation Forest (contamination=0.20) ===")
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", report_txt)
print("ROC-AUC:", roc)

with open(os.path.join(OUTPUT_DIR, "isolation forest results.txt"), "w") as f:
    f.write("Isolation Forest (contamination=0.20)\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")
    f.write("Classification Report:\n")
    f.write(report_txt + "\n")
    f.write("ROC-AUC:\n")
    f.write(str(roc) + "\n")

joblib.dump(model, os.path.join(OUTPUT_DIR, "isolation forest baseline model.pkl"))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "robust_scaler.pkl"))