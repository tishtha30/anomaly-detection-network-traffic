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

contamination_values = [0.05, 0.10, 0.20]

results = []

for c in contamination_values:
    model = IsolationForest(
        n_estimators=200,
        contamination=c,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train)

    pred = model.predict(X_test)
    pred = np.where(pred == -1, 1, 0)
    
    scores = -model.decision_function(X_test)

    roc = roc_auc_score(y_test, scores)
    cm = confusion_matrix(y_test, pred)
    report = classification_report(y_test, pred, output_dict=True)

    results.append({
        "contamination": c,
        "roc_auc": roc,
        "precision_attack": report["1"]["precision"],
        "recall_attack": report["1"]["recall"],
        "f1_attack": report["1"]["f1-score"]
    })

    print("\n=== Contamination:", c, "===")
    print("Confusion Matrix:\n", cm)
    print("ROC-AUC:", roc)
    print("Precision (attack):", report["1"]["precision"])
    print("Recall (attack):", report["1"]["recall"])
    print("F1 (attack):", report["1"]["f1-score"])

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "if_contamination_experiment.csv"), index=False)

#joblib.dump(model, os.path.join(OUTPUT_DIR, "isolation forest model.pkl"))