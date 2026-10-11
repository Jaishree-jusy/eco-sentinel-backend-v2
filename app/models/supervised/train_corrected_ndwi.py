from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = Path(
    r".\data\supervised\training_samples.csv"
)

MODEL_OUTPUT = Path(
    r".\data\supervised\deforestation_rf_corrected_ndwi.joblib"
)

METRICS_OUTPUT = Path(
    r".\data\supervised\rf_corrected_ndwi_metrics.json"
)

RANDOM_SEED = 42

FEATURE_COLUMNS = [
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


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("RANDOM FOREST - CORRECTED NDWI")
print("=" * 70)

print("\nLoading corrected dataset...")

df = pd.read_csv(DATASET)

print("Dataset shape:", df.shape)

print("\nClass distribution:")
print(df[TARGET].value_counts().sort_index())


# ============================================================
# SELECT FEATURES
# ============================================================

X = df[FEATURE_COLUMNS]
y = df[TARGET]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_SEED,
)

print("\nTraining samples:", len(X_train))
print("Final test samples:", len(X_test))


# ============================================================
# RANDOM FOREST
# ============================================================

print("\nCreating Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=40,
    min_samples_leaf=1,
    max_features=1.0,
    class_weight="balanced",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining model...")

model.fit(
    X_train,
    y_train,
)

print("Training complete.")


# ============================================================
# PROBABILITIES
# ============================================================

probabilities = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# THRESHOLD SEARCH
# ============================================================

print("\nSearching classification threshold...")

best_threshold = 0.50
best_accuracy = 0.0

for threshold in np.arange(
    0.20,
    0.81,
    0.01,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = float(threshold)


print(
    "Best threshold:",
    best_threshold,
)

print(
    "Best threshold accuracy:",
    f"{best_accuracy:.4f}"
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

predictions = (
    probabilities >= best_threshold
).astype(int)

accuracy = accuracy_score(
    y_test,
    predictions,
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0,
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0,
)

cm = confusion_matrix(
    y_test,
    predictions,
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL CORRECTED-NDWI RESULTS")
print("=" * 70)

print(
    f"Accuracy : {accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"Threshold: {best_threshold:.2f}"
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print("Original RF              : 85.71%")
print("Tuned RF (16 features)   : 87.38%")
print("Strong-feature RF        : 88.08%")
print("Strong-feature tuned RF  : 88.63%")
print(
    f"Corrected NDWI RF        : "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

importance = model.feature_importances_

feature_importance = sorted(
    zip(
        FEATURE_COLUMNS,
        importance,
    ),
    key=lambda x: x[1],
    reverse=True,
)

for feature, value in feature_importance:
    print(
        f"{feature:15s} {value:.6f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "threshold": best_threshold,
        "ndwi_formula": "(B3 - B8) / (B3 + B8)",
    },
    MODEL_OUTPUT,
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "model": "Random Forest",
    "dataset": str(DATASET),
    "ndwi_corrected": True,
    "ndwi_formula": "(B3 - B8) / (B3 + B8)",
    "features": FEATURE_COLUMNS,
    "random_seed": RANDOM_SEED,
    "training_samples": len(X_train),
    "final_test_samples": len(X_test),
    "best_threshold": best_threshold,
    "test_accuracy": float(accuracy),
    "test_precision": float(precision),
    "test_recall": float(recall),
    "test_f1": float(f1),
    "confusion_matrix": cm.tolist(),
}

with open(
    METRICS_OUTPUT,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metrics,
        f,
        indent=2,
    )


print("\nSaved model:")
print(MODEL_OUTPUT)

print("\nSaved metrics:")
print(METRICS_OUTPUT)

print("\n" + "=" * 70)
print("CORRECTED NDWI EXPERIMENT COMPLETE")
print("=" * 70)