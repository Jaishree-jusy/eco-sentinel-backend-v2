from pathlib import Path

import json
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

DATASET = Path(
    r".\data\supervised\training_samples.csv"
)

MODEL_OUTPUT = Path(
    r".\data\supervised\xgboost_spatial_selected.joblib"
)

METRICS_OUTPUT = Path(
    r".\data\supervised\xgboost_spatial_selected_metrics.json"
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


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SPATIAL-SELECTED XGBOOST EXPERIMENT")
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
# STEP 2: DEVELOPMENT TRAIN + VALIDATION
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
# STEP 3: XGBOOST CONFIGURATIONS
# ============================================================

configs = [
    {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1,
    },
    {
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1,
    },
    {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1,
    },
    {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1,
    },
    {
        "n_estimators": 800,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1,
    },
    {
        "n_estimators": 500,
        "max_depth": 8,
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1,
    },
]


# ============================================================
# STEP 4: VALIDATION SEARCH
# ============================================================

best_config = None
best_threshold = 0.50
best_validation_accuracy = -1.0

results = []

print("\nTotal configurations:", len(configs))
print("\nStarting XGBoost validation search...")
print("=" * 70)


for i, config in enumerate(configs, start=1):

    print(
        f"\n[{i}/{len(configs)}] Testing: {config}"
    )

    model = XGBClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        min_child_weight=config["min_child_weight"],
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(
        X_train,
        y_train,
    )

    val_probability = model.predict_proba(
        X_val
    )[:, 1]

    local_best_threshold = 0.50
    local_best_accuracy = -1.0

    for threshold in np.arange(
        0.30,
        0.71,
        0.01,
    ):

        val_prediction = (
            val_probability >= threshold
        ).astype(int)

        val_accuracy = accuracy_score(
            y_val,
            val_prediction,
        )

        if val_accuracy > local_best_accuracy:
            local_best_accuracy = val_accuracy
            local_best_threshold = float(
                round(threshold, 2)
            )

    results.append({
        "config": config,
        "validation_accuracy": float(
            local_best_accuracy
        ),
        "threshold": local_best_threshold,
    })

    print(
        f"Validation accuracy: "
        f"{local_best_accuracy * 100:.2f}%"
    )

    print(
        f"Validation threshold: "
        f"{local_best_threshold:.2f}"
    )

    if local_best_accuracy > best_validation_accuracy:

        best_validation_accuracy = (
            local_best_accuracy
        )

        best_config = config.copy()
        best_threshold = local_best_threshold

        print(">>> NEW BEST CONFIGURATION <<<")


# ============================================================
# STEP 5: BEST VALIDATION RESULT
# ============================================================

print("\n" + "=" * 70)
print("BEST XGBOOST VALIDATION CONFIGURATION")
print("=" * 70)

print("Best configuration:")
print(best_config)

print(
    "Best validation accuracy:",
    f"{best_validation_accuracy * 100:.2f}%"
)

print(
    "Selected threshold:",
    best_threshold,
)


# ============================================================
# STEP 6: RETRAIN BEST MODEL ON ALL DEVELOPMENT DATA
# ============================================================

print("\n" + "=" * 70)
print("RETRAINING BEST XGBOOST ON DEVELOPMENT DATA")
print("=" * 70)

final_model = XGBClassifier(
    n_estimators=best_config["n_estimators"],
    max_depth=best_config["max_depth"],
    learning_rate=best_config["learning_rate"],
    subsample=best_config["subsample"],
    colsample_bytree=best_config["colsample_bytree"],
    min_child_weight=best_config["min_child_weight"],
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_SEED,
    n_jobs=-1,
    tree_method="hist",
)

final_model.fit(
    X_dev,
    y_dev,
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
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL SPATIAL-SELECTED XGBOOST RESULTS")
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

print("Current RF champion : 90.51%")
print(
    f"XGBoost              : "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("XGBOOST FEATURE IMPORTANCE")
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
        "best_config": best_config,
    },
    MODEL_OUTPUT,
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "model": "XGBoost - Spatial Selected",
    "dataset": str(DATASET),
    "features": FEATURE_COLUMNS,
    "random_seed": RANDOM_SEED,

    "development_samples": len(X_dev),
    "training_samples": len(X_train),
    "validation_samples": len(X_val),
    "final_test_samples": len(X_test),

    "best_config": best_config,

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

    "validation_results": results,
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
print("XGBOOST EXPERIMENT COMPLETE")
print("=" * 70)