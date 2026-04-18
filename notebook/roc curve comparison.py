import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cicids2018_sample.csv")
FIG_DIR = os.path.join(PROJECT_ROOT, "output", "figures")

os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv("D:/CU/Dissertation/Final DS Project/data/processed/cicids2018_processed.csv")

X = df.drop(columns=["Label", "is_attack"])
y = df["is_attack"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -------------------
# Isolation Forest
# -------------------

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

if_model = IsolationForest(
    n_estimators=200,
    contamination=0.2,
    random_state=42
)

if_model.fit(X_scaled[y == 0])

if_scores = -if_model.decision_function(X_scaled)

fpr_if, tpr_if, _ = roc_curve(y, if_scores)
auc_if = roc_auc_score(y, if_scores)

# -------------------
# Random Forest
# -------------------

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

rf.fit(X_train, y_train)

rf_prob = rf.predict_proba(X_test)[:,1]

fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
auc_rf = roc_auc_score(y_test, rf_prob)

# -------------------
# XGBoost
# -------------------

scaler_xgb = RobustScaler()
X_train_xgb = scaler_xgb.fit_transform(X_train)
X_test_xgb = scaler_xgb.transform(X_test)

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

xgb.fit(X_train_xgb, y_train)

xgb_prob = xgb.predict_proba(X_test_xgb)[:,1]

fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_prob)
auc_xgb = roc_auc_score(y_test, xgb_prob)

# -------------------
# Deep Learning
# -------------------

scaler_ann = StandardScaler()
X_train_ann = scaler_ann.fit_transform(X_train)
X_test_ann = scaler_ann.transform(X_test)

ann = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=100,
    random_state=42
)

ann.fit(X_train_ann, y_train)

ann_prob = ann.predict_proba(X_test_ann)[:, 1]

fpr_ann, tpr_ann, _ = roc_curve(y_test, ann_prob)
auc_ann = roc_auc_score(y_test, ann_prob)

# -------------------
# Plot ROC
# -------------------

plt.figure(figsize=(7,6))

plt.plot(fpr_if, tpr_if, label=f'Isolation Forest (AUC={auc_if:.3f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={auc_rf:.3f})')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC={auc_xgb:.3f})')
plt.plot(fpr_ann, tpr_ann, label=f'Deep Learning ANN (AUC={auc_ann:.3f})')


plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")

plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(FIG_DIR,"roc curve comparison.png"))

print("ROC curve saved")