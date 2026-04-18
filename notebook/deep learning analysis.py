import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Paths
DATA_PATH = r"D:/CU/Dissertation/Final DS Project/data/processed/cicids2018_processed.csv"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Label", "is_attack"])
y = df["is_attack"]

# Scaling (IMPORTANT for DL)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Model
model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=256,
    validation_split=0.1,
    verbose=1
)

# Predictions
probs = model.predict(X_test).ravel()
pred = (probs > 0.5).astype(int)

# Metrics
acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred)
roc = roc_auc_score(y_test, probs)

print("\n=== Deep Learning ANN Results ===")
print("Accuracy:", acc)
print("F1 Score:", f1)
print("ROC-AUC:", roc)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred))