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
    r".\data\supervised\deforestation_rf_hard_negative.joblib"
)

METRICS_OUTPUT = Path(
    r".\data\supervised\rf_hard_negative_metrics.json"
)

ERROR_OUTPUT = Path(
    r".\data\supervised\hard_negative_error_analysis.csv"
)

RANDOM_SEED = 42

FEATURE_COLUMNS = [
    "delta_B12",
    "delta_nbr",
    "delta_B1",
    "delta_B9",
    "nbr_local_median_2023",
    "delta_snbr",
    "delta_B11",
    "nbr_local_median_2025",
    "delta_B5",
    "snbr_2025",
    "delta_ndmi",
    "nbr_change_zscore",
]

TARGET = "label"

# Conservative hard-negative weighting.
# Normal samples = 1.0
# Hard negative samples = 2.0
HARD_NEGATIVE_WEIGHT = 2.0

# Exact configuration of the current 90.51% champion.
RF_CONFIG = {
    "n_estimators": 300,
    "max_depth": 40,
    "min_samples_leaf": 1,
    "max_features": 1.0,
}


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("HARD-NEGATIVE RANDOM FOREST EXPERIMENT")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATASET)

print("Dataset shape:", df.shape)

print("\nClass distribution:")
print(df[TARGET].value_counts().sort_index())

X = df[FEATURE_COLUMNS]
y = df[TARGET]


# ============================================================
# STEP 1: CREATE UNTOUCHED FINAL TEST SET
# ============================================================

X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_SEED,
)

print("\nDevelopment samples:", len(X_dev))
print("Untouched final test samples:", len(X_test))


# ============================================================
# STEP 2: SPLIT DEVELOPMENT DATA
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.20,
    stratify=y_dev,
    random_state=RANDOM_SEED,
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))


# ============================================================
# STEP 3: CREATE HARD-NEGATIVE WEIGHTS
# ============================================================

print("\n" + "=" * 70)
print("CREATING HARD-NEGATIVE WEIGHTS")
print("=" * 70)

train_weights = np.ones(len(X_train))

hard_negative = (
    (y_train.to_numpy() == 0)
    &
    (
        (X_train["delta_nbr"] < -0.07)
        |
        (X_train["delta_ndmi"] < -0.05)
        |
        (X_train["nbr_change_zscore"] < -0.5)
        |
        (X_train["delta_B11"] > 250)
        |
        (X_train["delta_B12"] > 200)
    )
)

train_weights[hard_negative] = HARD_NEGATIVE_WEIGHT

print("Hard-negative weight:", HARD_NEGATIVE_WEIGHT)
print("Hard negatives in training:", int(hard_negative.sum()))
print("Normal training samples:", int((~hard_negative).sum()))


# ============================================================
# STEP 4: TRAIN HARD-NEGATIVE RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING HARD-NEGATIVE RANDOM FOREST")
print("=" * 70)

print("Configuration:")
print(RF_CONFIG)

model = RandomForestClassifier(
    n_estimators=RF_CONFIG["n_estimators"],
    max_depth=RF_CONFIG["max_depth"],
    min_samples_leaf=RF_CONFIG["min_samples_leaf"],
    max_features=RF_CONFIG["max_features"],
    class_weight="balanced",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
    sample_weight=train_weights,
)

print("\nTraining complete.")


# ============================================================
# STEP 5: VALIDATION THRESHOLD SEARCH
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION THRESHOLD SEARCH")
print("=" * 70)

val_probability = model.predict_proba(X_val)[:, 1]

best_threshold = 0.50
best_validation_accuracy = -1.0

for threshold in np.arange(0.30, 0.71, 0.01):

    val_prediction = (
        val_probability >= threshold
    ).astype(int)

    val_accuracy = accuracy_score(
        y_val,
        val_prediction,
    )

    if val_accuracy > best_validation_accuracy:
        best_validation_accuracy = val_accuracy
        best_threshold = float(round(threshold, 2))

print(
    f"Best validation accuracy: "
    f"{best_validation_accuracy * 100:.2f}%"
)

print(
    f"Selected threshold: "
    f"{best_threshold:.2f}"
)


# ============================================================
# STEP 6: RETRAIN ON ALL DEVELOPMENT DATA
# ============================================================

print("\n" + "=" * 70)
print("RETRAINING ON ALL DEVELOPMENT DATA")
print("=" * 70)

dev_weights = np.ones(len(X_dev))

dev_hard_negative = (
    (y_dev.to_numpy() == 0)
    &
    (
        (X_dev["delta_nbr"] < -0.07)
        |
        (X_dev["delta_ndmi"] < -0.05)
        |
        (X_dev["nbr_change_zscore"] < -0.5)
        |
        (X_dev["delta_B11"] > 250)
        |
        (X_dev["delta_B12"] > 200)
    )
)

dev_weights[dev_hard_negative] = HARD_NEGATIVE_WEIGHT

print(
    "Hard negatives in development data:",
    int(dev_hard_negative.sum())
)

final_model = RandomForestClassifier(
    n_estimators=RF_CONFIG["n_estimators"],
    max_depth=RF_CONFIG["max_depth"],
    min_samples_leaf=RF_CONFIG["min_samples_leaf"],
    max_features=RF_CONFIG["max_features"],
    class_weight="balanced",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

final_model.fit(
    X_dev,
    y_dev,
    sample_weight=dev_weights,
)

print("Final development training complete.")


# ============================================================
# STEP 7: FINAL UNTOUCHED TEST
# ============================================================

print("\n" + "=" * 70)
print("FINAL UNTOUCHED TEST")
print("=" * 70)

test_probability = final_model.predict_proba(
    X_test
)[:, 1]

test_prediction = (
    test_probability >= best_threshold
).astype(int)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    test_prediction,
)

precision = precision_score(
    y_test,
    test_prediction,
    zero_division=0,
)

recall = recall_score(
    y_test,
    test_prediction,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    test_prediction,
    zero_division=0,
)

cm = confusion_matrix(
    y_test,
    test_prediction,
)


# ============================================================
# ERROR ANALYSIS
# ============================================================

error_analysis = pd.DataFrame({
    "actual": y_test.to_numpy(),
    "predicted": test_prediction,
    "probability": test_probability,
})

error_analysis["error_type"] = "correct"

error_analysis.loc[
    (error_analysis["actual"] == 0)
    &
    (error_analysis["predicted"] == 1),
    "error_type"
] = "false_positive"

error_analysis.loc[
    (error_analysis["actual"] == 1)
    &
    (error_analysis["predicted"] == 0),
    "error_type"
] = "false_negative"

error_analysis.to_csv(
    ERROR_OUTPUT,
    index=False,
)

print("\nError analysis saved:")
print(ERROR_OUTPUT)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("HARD-NEGATIVE RANDOM FOREST RESULTS")
print("=" * 70)

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall   : {recall * 100:.2f}%"
)

print(
    f"F1 Score : {f1 * 100:.2f}%"
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

print("Current Champion RF       : 90.51%")
print(
    f"Hard-Negative RF         : "
    f"{accuracy * 100:.2f}%"
)

difference = (accuracy - 0.9051) * 100

print(
    f"Difference               : "
    f"{difference:+.2f} percentage points"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

importance = final_model.feature_importances_

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
        f"{feature:25s} {value:.6f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    {
        "model": final_model,
        "feature_columns": FEATURE_COLUMNS,
        "threshold": best_threshold,
        "ndwi_formula": "(B3 - B8) / (B3 + B8)",
        "best_config": RF_CONFIG,
        "hard_negative_weight": HARD_NEGATIVE_WEIGHT,
        "hard_negative_rules": [
            "delta_nbr < -0.07",
            "delta_ndmi < -0.05",
            "nbr_change_zscore < -0.5",
            "delta_B11 > 250",
            "delta_B12 > 200",
        ],
    },
    MODEL_OUTPUT,
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "model": "Random Forest - Hard Negative Weighted",
    "dataset": str(DATASET),
    "features": FEATURE_COLUMNS,
    "random_seed": RANDOM_SEED,

    "development_samples": len(X_dev),
    "training_samples": len(X_train),
    "validation_samples": len(X_val),
    "final_test_samples": len(X_test),

    "best_config": RF_CONFIG,

    "hard_negative_weight": HARD_NEGATIVE_WEIGHT,

    "hard_negative_rules": [
        "delta_nbr < -0.07",
        "delta_ndmi < -0.05",
        "nbr_change_zscore < -0.5",
        "delta_B11 > 250",
        "delta_B12 > 200",
    ],

    "training_hard_negatives": int(
        hard_negative.sum()
    ),

    "development_hard_negatives": int(
        dev_hard_negative.sum()
    ),

    "validation_accuracy": float(
        best_validation_accuracy
    ),

    "selected_threshold": float(
        best_threshold
    ),

    "test_accuracy": float(
        accuracy
    ),

    "test_precision": float(
        precision
    ),

    "test_recall": float(
        recall
    ),

    "test_f1": float(
        f1
    ),

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


# ============================================================
# COMPLETE
# ============================================================

print("\nSaved model:")
print(MODEL_OUTPUT)

print("\nSaved metrics:")
print(METRICS_OUTPUT)

print("\n" + "=" * 70)
print("HARD-NEGATIVE EXPERIMENT COMPLETE")
print("=" * 70)