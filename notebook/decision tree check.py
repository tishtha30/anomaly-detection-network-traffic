import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_PATH = r"D:/CU/Dissertation/Final DS Project/data/processed/cicids2018_processed.csv"
OUT_DIR = os.path.join(PROJECT_ROOT, "output", "comparison")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Label", "is_attack"])
y = df["is_attack"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_pred))

print("\nClassification Report:")
print(classification_report(y_test, test_pred))

plt.figure(figsize=(16, 8))
plot_tree(
    model,
    feature_names=X.columns.tolist(),
    class_names=["Benign", "Attack"],
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title("Decision Tree for Leakage and Feature Logic Check")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "decision_tree_check.png"))
plt.show()