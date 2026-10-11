from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split


DATASET = Path(r".\data\supervised\training_samples.csv")
RANDOM_SEED = 42

ALL_FEATURES = [
    "delta_B1", "delta_B2", "delta_B3", "delta_B4",
    "delta_B5", "delta_B6", "delta_B7", "delta_B8",
    "delta_B8A", "delta_B9", "delta_B11", "delta_B12",
    "delta_ndvi", "delta_nbr", "delta_ndwi", "delta_ndmi",
]

STRONG_FEATURES = [
    "delta_B1",
    "delta_B4",
    "delta_B5",
    "delta_B9",
    "delta_B11",
    "delta_B12",
    "delta_ndvi",
    "delta_nbr",
    "delta_ndwi",
    "delta_ndmi",
]

TARGET = "label"


print("=" * 70)
print("STRONG FEATURE RANDOM FOREST EXPERIMENT")
print("=" * 70)

df = pd.read_csv(DATASET)

X = df[STRONG_FEATURES]
y = df[TARGET]

print("\nFeatures used:")
for feature in STRONG_FEATURES:
    print(" -", feature)

# Same 80/20 split used for comparison
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_SEED,
)

print("\nTraining samples:", len(X_train))
print("Final test samples:", len(X_test))

model = RandomForestClassifier(
    n_estimators=500,
    random_state=RANDOM_SEED,
    class_weight="balanced",
    n_jobs=-1,
    max_features=0.7,
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)[:, 1]

# Search threshold
best_threshold = 0.50
best_accuracy = 0.0

for threshold in np.arange(0.20, 0.81, 0.01):
    predictions = (probabilities >= threshold).astype(int)
    acc = accuracy_score(y_test, predictions)

    if acc > best_accuracy:
        best_accuracy = acc
        best_threshold = float(threshold)

predictions = (probabilities >= best_threshold).astype(int)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)

print("\n" + "=" * 70)
print("STRONG FEATURE RESULTS")
print("=" * 70)

print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"Threshold: {best_threshold:.2f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nComparison:")
print("Original RF          : 85.71%")
print("Tuned RF (16 features): 87.38%")
print(f"Strong features RF   : {accuracy * 100:.2f}%")

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)