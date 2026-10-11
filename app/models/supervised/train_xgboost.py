import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = Path(r".\data\supervised\training_samples.csv")
MODEL_OUTPUT = Path(r".\data\supervised\deforestation_xgboost.joblib")
METRICS_OUTPUT = Path(r".\data\supervised\xgboost_model_metrics.json")

RANDOM_SEED = 42

FEATURE_COLUMNS = [
    "delta_B1",
    "delta_B2",
    "delta_B3",
    "delta_B4",
    "delta_B5",
    "delta_B6",
    "delta_B7",
    "delta_B8",
    "delta_B8A",
    "delta_B9",
    "delta_B11",
    "delta_B12",
    "delta_ndvi",
    "delta_nbr",
    "delta_ndwi",
    "delta_ndmi",
]

TARGET_COLUMN = "label"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("XGBOOST DEFORESTATION MODEL TRAINING")
print("=" * 70)

print("\nLoading dataset...")
df = pd.read_csv(DATASET)

print("Dataset shape:", df.shape)

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

print("\nClass distribution:")
print(y.value_counts().sort_index())


# ============================================================
# TRAIN / VALIDATION / FINAL TEST SPLIT
# ============================================================

# First: 80% development, 20% untouched final test
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_SEED,
)

# Second: 80% training, 20% validation from development set
X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.20,
    stratify=y_dev,
    random_state=RANDOM_SEED,
)

print("\nDataset split:")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Final test:", len(X_test))


# ============================================================
# MODEL
# ============================================================

print("\nCreating XGBoost model...")

model = XGBClassifier(
    n_estimators=1500,
    learning_rate=0.03,
    max_depth=8,
    min_child_weight=2,
    subsample=0.85,
    colsample_bytree=0.85,
    gamma=0.05,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_SEED,
    n_jobs=-1,
    tree_method="hist",
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining XGBoost...")
print("This may take some time.\n")

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

print("Training complete.")


# ============================================================
# VALIDATION THRESHOLD SEARCH
# ============================================================

print("\nSearching for best classification threshold...")

val_probabilities = model.predict_proba(X_val)[:, 1]

best_threshold = 0.50
best_accuracy = 0.0

for threshold in np.arange(0.20, 0.81, 0.01):
    val_predictions = (val_probabilities >= threshold).astype(int)

    accuracy = accuracy_score(y_val, val_predictions)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = float(threshold)

print("Best validation threshold:", best_threshold)
print("Best validation accuracy:", round(best_accuracy, 4))


# ============================================================
# FINAL TEST
# ============================================================

print("\nEvaluating on untouched final test set...")

test_probabilities = model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

accuracy = accuracy_score(y_test, test_predictions)
precision = precision_score(y_test, test_predictions, zero_division=0)
recall = recall_score(y_test, test_predictions, zero_division=0)
f1 = f1_score(y_test, test_predictions, zero_division=0)

cm = confusion_matrix(y_test, test_predictions)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL XGBOOST RESULTS")
print("=" * 70)

print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification threshold:", best_threshold)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\nFeature importance:")

importance = model.feature_importances_

feature_importance = sorted(
    zip(FEATURE_COLUMNS, importance),
    key=lambda x: x[1],
    reverse=True,
)

for feature, value in feature_importance:
    print(f"{feature:15s} {value:.6f}")


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "threshold": best_threshold,
    },
    MODEL_OUTPUT,
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "model": "XGBoost",
    "dataset": str(DATASET),
    "random_seed": RANDOM_SEED,
    "features": FEATURE_COLUMNS,
    "training_samples": len(X_train),
    "validation_samples": len(X_val),
    "final_test_samples": len(X_test),
    "best_validation_threshold": best_threshold,
    "validation_accuracy": float(best_accuracy),
    "test_accuracy": float(accuracy),
    "test_precision": float(precision),
    "test_recall": float(recall),
    "test_f1": float(f1),
    "confusion_matrix": cm.tolist(),
}

with open(METRICS_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)


print("\nSaved model:")
print(MODEL_OUTPUT)

print("\nSaved metrics:")
print(METRICS_OUTPUT)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)