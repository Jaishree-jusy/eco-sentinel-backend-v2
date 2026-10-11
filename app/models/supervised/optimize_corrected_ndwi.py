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
    r".\data\supervised\deforestation_rf_corrected_optimized.joblib"
)

METRICS_OUTPUT = Path(
    r".\data\supervised\rf_corrected_optimized_metrics.json"
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
print("CORRECTED-NDWI RANDOM FOREST OPTIMIZATION")
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
# STEP 2: SPLIT DEVELOPMENT DATA INTO TRAIN + VALIDATION
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
# STEP 3: HYPERPARAMETER CONFIGURATIONS
# ============================================================

configs = []

for n_estimators in [300, 500, 800]:
    for max_depth in [None, 20, 30, 40]:
        for min_samples_leaf in [1, 2]:
            for max_features in ["sqrt", 0.7, 1.0]:

                configs.append({
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                    "max_features": max_features,
                })


print("\nTotal configurations:", len(configs))


# ============================================================
# STEP 4: VALIDATION SEARCH
# ============================================================

best_config = None
best_threshold = 0.50
best_validation_accuracy = -1.0

results = []

print("\nStarting validation search...")
print("=" * 70)

for i, config in enumerate(configs, start=1):

    print(
        f"\n[{i}/{len(configs)}] "
        f"Testing: {config}"
    )

    model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features=config["max_features"],
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
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
# STEP 5: SHOW BEST VALIDATION RESULT
# ============================================================

print("\n" + "=" * 70)
print("BEST VALIDATION CONFIGURATION")
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
print("RETRAINING BEST MODEL ON DEVELOPMENT DATA")
print("=" * 70)

final_model = RandomForestClassifier(
    n_estimators=best_config["n_estimators"],
    max_depth=best_config["max_depth"],
    min_samples_leaf=best_config["min_samples_leaf"],
    max_features=best_config["max_features"],
    class_weight="balanced",
    random_state=RANDOM_SEED,
    n_jobs=-1,
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
print("FINAL OPTIMIZED CORRECTED-NDWI RESULTS")
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

print("Original RF              : 85.71%")
print("Tuned RF (16 features)   : 87.38%")
print("Strong-feature RF        : 88.08%")
print("Strong-feature tuned RF  : 88.63%")
print("Corrected NDWI RF        : 88.76%")
print(
    f"Optimized corrected RF   : "
    f"{accuracy * 100:.2f}%"
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
        f"{feature:15s} {value:.6f}"
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
        "best_config": best_config,
    },
    MODEL_OUTPUT,
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "model": "Random Forest - Corrected NDWI - Optimized",
    "dataset": str(DATASET),
    "ndwi_corrected": True,
    "ndwi_formula": "(B3 - B8) / (B3 + B8)",
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
print("OPTIMIZATION COMPLETE")
print("=" * 70)