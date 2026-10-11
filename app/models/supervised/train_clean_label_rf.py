from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = r".\data\supervised\training_samples.csv"

MODEL_DIR = r".\data\supervised"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "deforestation_rf_clean_labels.joblib",
)

METRICS_PATH = os.path.join(
    MODEL_DIR,
    "rf_clean_labels_metrics.json",
)


# ============================================================
# BEST 10 FEATURES
# ============================================================

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


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def find_best_threshold(
    y_true,
    probabilities,
):

    best_threshold = 0.50
    best_accuracy = -1.0

    for threshold in np.arange(
        0.20,
        0.81,
        0.005,
    ):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            y_true,
            predictions,
        )

        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_threshold = float(
                threshold
            )

    return (
        best_threshold,
        best_accuracy,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "ECO-SENTINEL CLEAN-LABEL RANDOM FOREST"
    )
    print("=" * 75)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(
        DATASET_PATH
    )

    print(
        f"\nDataset: {len(df):,} samples"
    )

    X = df[
        FEATURE_COLUMNS
    ]

    y = df[
        "label"
    ].astype(int)

    # ========================================================
    # FIXED 80 / 20 DEVELOPMENT / TEST SPLIT
    # ========================================================

    (
        X_dev,
        X_test,
        y_dev,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # ========================================================
    # DEVELOPMENT → TRAIN / VALIDATION
    # ========================================================

    (
        X_train,
        X_val,
        y_train,
        y_val,
    ) = train_test_split(
        X_dev,
        y_dev,
        test_size=0.20,
        random_state=42,
        stratify=y_dev,
    )

    print("\nSplit:")
    print(
        f"Training   : {len(X_train):,}"
    )
    print(
        f"Validation : {len(X_val):,}"
    )
    print(
        f"Final test : {len(X_test):,}"
    )

    # ========================================================
    # STEP 1 — BASE MODEL
    # ========================================================

    print("\n" + "=" * 75)
    print(
        "STEP 1 — TRAIN BASE MODEL"
    )
    print("=" * 75)

    base_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=40,
        min_samples_leaf=1,
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
    )

    base_model.fit(
        X_train,
        y_train,
    )

    # OOB model gives a more honest estimate of
    # training-sample difficulty.

    print(
        "\nTraining OOB model..."
    )

    oob_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=40,
        min_samples_leaf=1,
        max_features=1.0,
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )

    oob_model.fit(
        X_train,
        y_train,
    )

    oob_probability = (
        oob_model.oob_decision_function_[:, 1]
    )

    # ========================================================
    # STEP 2 — IDENTIFY AMBIGUOUS TRAINING SAMPLES
    # ========================================================

    print("\n" + "=" * 75)
    print(
        "STEP 2 — IDENTIFY AMBIGUOUS LABELS"
    )
    print("=" * 75)

    # Distance from 0.5.
    # Small distance = difficult/ambiguous.
    confidence = np.abs(
        oob_probability - 0.5
    )

    # Try several conservative cleaning levels.
    removal_rates = [
        0.00,
        0.02,
        0.05,
        0.08,
        0.10,
    ]

    best_rate = 0.00
    best_val_accuracy = -1.0
    best_threshold = 0.50

    # Convert pandas to arrays for indexing.
    X_train_array = X_train.to_numpy()
    y_train_array = y_train.to_numpy()

    # ========================================================
    # STEP 3 — TEST CLEANING LEVELS
    # ========================================================

    print("\n" + "=" * 75)
    print(
        "STEP 3 — TEST LABEL CLEANING LEVELS"
    )
    print("=" * 75)

    for removal_rate in removal_rates:

        if removal_rate == 0:

            keep_mask = np.ones(
                len(X_train_array),
                dtype=bool,
            )

        else:

            number_remove = int(
                len(X_train_array)
                * removal_rate
            )

            # Remove the least confident samples.
            remove_indices = np.argsort(
                confidence
            )[:number_remove]

            keep_mask = np.ones(
                len(X_train_array),
                dtype=bool,
            )

            keep_mask[
                remove_indices
            ] = False

        X_clean = X_train_array[
            keep_mask
        ]

        y_clean = y_train_array[
            keep_mask
        ]

        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=40,
            min_samples_leaf=1,
            max_features=1.0,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            X_clean,
            y_clean,
        )

        val_probability = (
            model.predict_proba(
                X_val
            )[:, 1]
        )

        threshold, accuracy = (
            find_best_threshold(
                y_val,
                val_probability,
            )
        )

        print(
            f"Removal={removal_rate * 100:>5.1f}% "
            f"| Samples={len(X_clean):>6,} "
            f"| Validation="
            f"{accuracy:.4f} "
            f"| Threshold="
            f"{threshold:.3f}"
        )

        if accuracy > best_val_accuracy:

            best_val_accuracy = accuracy
            best_rate = removal_rate
            best_threshold = threshold

    # ========================================================
    # BEST CONFIGURATION
    # ========================================================

    print("\n" + "=" * 75)
    print(
        "BEST CLEANING CONFIGURATION"
    )
    print("=" * 75)

    print(
        f"Removal rate : "
        f"{best_rate * 100:.1f}%"
    )

    print(
        f"Validation accuracy : "
        f"{best_val_accuracy:.4f}"
    )

    print(
        f"Threshold : "
        f"{best_threshold:.3f}"
    )

    # ========================================================
    # STEP 4 — CLEAN COMPLETE DEVELOPMENT SET
    # ========================================================

    print("\n" + "=" * 75)
    print(
        "STEP 4 — CLEAN DEVELOPMENT SET"
    )
    print("=" * 75)

    X_dev_array = X_dev.to_numpy()
    y_dev_array = y_dev.to_numpy()

    # Get OOB probabilities for the complete development set.
    dev_oob_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=40,
        min_samples_leaf=1,
        max_features=1.0,
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )

    dev_oob_model.fit(
        X_dev_array,
        y_dev_array,
    )

    dev_confidence = np.abs(
        dev_oob_model.oob_decision_function_[:, 1]
        - 0.5
    )

    if best_rate == 0:

        dev_keep_mask = np.ones(
            len(X_dev_array),
            dtype=bool,
        )

    else:

        number_remove = int(
            len(X_dev_array)
            * best_rate
        )

        remove_indices = np.argsort(
            dev_confidence
        )[:number_remove]

        dev_keep_mask = np.ones(
            len(X_dev_array),
            dtype=bool,
        )

        dev_keep_mask[
            remove_indices
        ] = False

    X_clean_dev = X_dev_array[
        dev_keep_mask
    ]

    y_clean_dev = y_dev_array[
        dev_keep_mask
    ]

    print(
        f"Original development samples: "
        f"{len(X_dev_array):,}"
    )

    print(
        f"Clean development samples: "
        f"{len(X_clean_dev):,}"
    )

    print(
        f"Removed samples: "
        f"{len(X_dev_array) - len(X_clean_dev):,}"
    )

    # ========================================================
    # STEP 5 — FINAL CLEAN MODEL
    # ========================================================

    print("\n" + "=" * 75)
    print(
        "STEP 5 — TRAIN FINAL CLEAN MODEL"
    )
    print("=" * 75)

    final_model = RandomForestClassifier(
        n_estimators=800,
        max_depth=40,
        min_samples_leaf=1,
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
    )

    final_model.fit(
        X_clean_dev,
        y_clean_dev,
    )

    print(
        "Final model training complete."
    )

    # ========================================================
    # STEP 6 — FINAL TEST
    # ========================================================

    print("\n" + "=" * 75)
    print(
        "🔥 FINAL UNTOUCHED TEST 🔥"
    )
    print("=" * 75)

    test_probability = (
        final_model.predict_proba(
            X_test
        )[:, 1]
    )

    predictions = (
        test_probability
        >= best_threshold
    ).astype(int)

    # ========================================================
    # METRICS
    # ========================================================

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

    roc_auc = roc_auc_score(
        y_test,
        test_probability,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    print(
        f"\nAccuracy  : "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print(
        f"Threshold : {best_threshold:.3f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(matrix)

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "No Deforestation",
                "Deforestation",
            ],
            zero_division=0,
        )
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": final_model,
            "threshold": best_threshold,
            "features": FEATURE_COLUMNS,
            "label_cleaning_rate": best_rate,
        },
        MODEL_PATH,
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics = {
        "model": "Clean-Label Random Forest",
        "random_state": 42,
        "features": FEATURE_COLUMNS,
        "feature_count": len(FEATURE_COLUMNS),
        "development_samples": len(X_dev),
        "clean_development_samples": len(
            X_clean_dev
        ),
        "testing_samples": len(X_test),
        "label_cleaning_rate": float(
            best_rate
        ),
        "decision_threshold": float(
            best_threshold
        ),
        "validation_accuracy": float(
            best_val_accuracy
        ),
        "accuracy": float(
            accuracy
        ),
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "f1_score": float(
            f1
        ),
        "roc_auc": float(
            roc_auc
        ),
        "confusion_matrix": matrix.tolist(),
        "dataset": DATASET_PATH,
        "model_path": MODEL_PATH,
    }

    with open(
        METRICS_PATH,
        "w",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )

    print("\n" + "=" * 75)
    print(
        "CLEAN-LABEL MODEL SAVED"
    )
    print("=" * 75)

    print(
        f"Model  : {MODEL_PATH}"
    )

    print(
        f"Metrics: {METRICS_PATH}"
    )


if __name__ == "__main__":
    main()