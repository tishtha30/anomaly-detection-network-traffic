import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from xgboost import XGBClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

OUT_COMP = os.path.join(PROJECT_ROOT, "output", "comparison")
OUT_FIG = os.path.join(PROJECT_ROOT, "output", "figures")

os.makedirs(OUT_COMP, exist_ok=True)
os.makedirs(OUT_FIG, exist_ok=True)

df = pd.read_csv(r"D:/CU/Dissertation/Final DS Project/data/processed/cicids2018_processed.csv")

X = df.drop(columns=["Label", "is_attack"])
y = df["is_attack"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

results = []


def save_metrics(model_name, y_true, y_pred, y_score):
    row = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_attack": precision_score(y_true, y_pred),
        "recall_attack": recall_score(y_true, y_pred),
        "f1_attack": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score)
    }
    results.append(row)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    with open(os.path.join(OUT_COMP, f"{model_name.lower().replace(' ', '_')}_results.txt"), "w") as f:
        f.write(f"{model_name} Results\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("ROC-AUC:\n")
        f.write(str(row["roc_auc"]) + "\n")

    return cm


# Isolation Forest
scaler_if = RobustScaler()
X_scaled = scaler_if.fit_transform(X)
X_train_if = X_scaled[y == 0]
X_test_if = X_scaled

if_rows = []
best_if = None
best_if_f1 = -1
best_if_cm = None

for c in [0.05, 0.10, 0.20]:
    if_model = IsolationForest(
        n_estimators=200,
        contamination=c,
        random_state=42,
        n_jobs=-1
    )
    if_model.fit(X_train_if)

    if_pred = if_model.predict(X_test_if)
    if_pred = np.where(if_pred == -1, 1, 0)
    if_score = -if_model.decision_function(X_test_if)

    row = {
        "contamination": c,
        "accuracy": accuracy_score(y, if_pred),
        "precision_attack": precision_score(y, if_pred),
        "recall_attack": recall_score(y, if_pred),
        "f1_attack": f1_score(y, if_pred),
        "roc_auc": roc_auc_score(y, if_score)
    }
    if_rows.append(row)

    if row["f1_attack"] > best_if_f1:
        best_if_f1 = row["f1_attack"]
        best_if = (if_model, if_pred, if_score, c)
        best_if_cm = confusion_matrix(y, if_pred)

if_exp_df = pd.DataFrame(if_rows)
if_exp_df.to_csv(os.path.join(OUT_COMP, "if contamination experiment.csv"), index=False)

best_model, best_pred, best_score, best_c = best_if
save_metrics(f"Isolation Forest (c={best_c})", y, best_pred, best_score)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]
rf_cm = save_metrics("Random Forest", y_test, rf_pred, rf_prob)

rf_imp = pd.DataFrame({
    "feature": X.columns,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)
rf_imp.to_csv(os.path.join(OUT_COMP, "rf feature importance.csv"), index=False)

# XGBoost
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

xgb_pred = xgb.predict(X_test_xgb)
xgb_prob = xgb.predict_proba(X_test_xgb)[:, 1]
xgb_cm = save_metrics("XGBoost", y_test, xgb_pred, xgb_prob)

xgb_imp = pd.DataFrame({
    "feature": X.columns,
    "importance": xgb.feature_importances_
}).sort_values("importance", ascending=False)
xgb_imp.to_csv(os.path.join(OUT_COMP, "xgb feature importance.csv"), index=False)

# Deep Learning ANN
ann_accuracy = 0.980690051960661
ann_precision = 0.99
ann_recall = 0.93
ann_f1 = 0.9622734298159337
ann_roc_auc = 0.9849987439859388
ann_cm = np.array([[40848, 106], [968, 13697]])

results.append({
    "model": "Deep Learning ANN",
    "accuracy": ann_accuracy,
    "precision_attack": ann_precision,
    "recall_attack": ann_recall,
    "f1_attack": ann_f1,
    "roc_auc": ann_roc_auc
})

with open(os.path.join(OUT_COMP, "deep learning ann results.txt"), "w") as f:
    f.write("Deep Learning ANN Results\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(ann_cm) + "\n\n")
    f.write("Classification Report:\n")
    f.write("Accuracy: " + str(ann_accuracy) + "\n")
    f.write("Precision (Attack): " + str(ann_precision) + "\n")
    f.write("Recall (Attack): " + str(ann_recall) + "\n")
    f.write("F1 Score (Attack): " + str(ann_f1) + "\n")
    f.write("ROC-AUC: " + str(ann_roc_auc) + "\n")

# Unified comparison csv
results_df = pd.DataFrame(results)
results_df["model"] = results_df["model"].replace(
    {f"Isolation Forest (c={best_c})": "Isolation Forest"}
)
results_df.to_csv(os.path.join(OUT_COMP, "model comparison.csv"), index=False)

# Precision vs Recall Bar plot
metrics_df = results_df.set_index("model")[["precision_attack", "recall_attack"]]

ax = metrics_df.plot(kind="bar", figsize=(8,5))
ax.set_title("Precision vs Recall Comparison")
ax.set_ylabel("Score")

for container in ax.containers:
    ax.bar_label(container, fmt='%.2f')

plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG, "Precision vs Recall Comparison.png"))
plt.show()

# Performance plot
plot_df = results_df.set_index("model")[["accuracy", "f1_attack", "roc_auc"]]

ax = plot_df.plot(
    kind="bar",
    figsize=(10, 5)
)

ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison")
ax.tick_params(axis="x", rotation=0)
ax.set_ylim(0, 1.05)

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=8)

fig = ax.get_figure()
fig.tight_layout()
plt.savefig(os.path.join(OUT_FIG, "model performance comparison.png"))
plt.show()
rank_df = results_df.copy()
rank_df["rank"] = rank_df["roc_auc"].rank(ascending=False)

rank_df.sort_values("rank").plot(
    x="model",
    y="roc_auc",
    kind="bar",
    legend=False,
    figsize=(8,5)
)

plt.title("Model Ranking based on ROC-AUC")
plt.ylabel("ROC-AUC")

for container in plt.gca().containers:
    plt.bar_label(container, fmt='%.2f')

plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG, "Model Ranking based on ROC-AUC.png"))
plt.show()

# Feature importance plots
rf_top10 = rf_imp.head(10)
plt.figure(figsize=(8, 5))
ac= sns.barplot(data=rf_top10, x="importance", y="feature")
for i, v in enumerate(rf_top10["importance"]):
    ac.text(v + 0.001, i, f"{v:.3f}", va='center')
plt.xlim(0, max(rf_top10["importance"]) + 0.02)
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG, "rf feature importance.png"))
plt.show()

xgb_top10 = xgb_imp.head(10)
plt.figure(figsize=(8, 5))
ax= sns.barplot(data=xgb_top10, x="importance", y="feature")
for i, v in enumerate(xgb_top10["importance"]):
    ax.text(v + 0.001, i, f"{v:.3f}", va='center')
plt.xlim(0, max(xgb_top10["importance"]) + 0.02)
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG, "xgb feature importance.png"))
plt.show()

# Confusion matrix heatmaps
for name, cm in [
    ("random_forest", rf_cm),
    ("xgboost", xgb_cm),
    ("isolation_forest", best_if_cm),
    ("deep_learning_ann", ann_cm)
]:
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{name.replace('_', ' ').title()} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, f"{name} confusion matrix.png"))
    plt.show()

# False Positive vs False Negative
fp_fn_df = pd.DataFrame({
    "Model": ["Isolation Forest", "Random Forest", "XGBoost", "ANN"],
    "False Positives": [40953, 334, 39, 106],
    "False Negatives": [58566, 867, 900, 968]
})

fp_fn_df.set_index("Model").plot(kind="bar", figsize=(8,5))
plt.title("False Positives vs False Negatives Comparison")
plt.ylabel("Count")

for container in plt.gca().containers:
    plt.bar_label(container)

plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG, f"{name} False Positives vs False Negatives Comparison.png"))
plt.show()

# Correlation heatmap
numeric_df = df.select_dtypes(include=["number"])
corr = numeric_df.corr()

top_corr = corr["is_attack"].abs().sort_values(ascending=False).head(12).index.tolist()

if "is_attack" not in top_corr:
    top_corr = ["is_attack"] + top_corr[:11]

corr_subset = numeric_df[top_corr].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_subset,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    linecolor="white",
    square=True,
    cbar=True
)

plt.title("Feature Correlation Heatmap", fontsize=13)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG, "correlation heatmap.png"))
plt.show()

print(results_df)
print("\nBest Isolation Forest contamination:", best_c)