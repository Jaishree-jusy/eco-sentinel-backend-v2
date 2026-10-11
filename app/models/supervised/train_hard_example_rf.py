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
    "deforestation_rf_hard_example.joblib",
)

METRICS_PATH = os.path.join(
    MODEL_DIR,
    "rf_hard_example_metrics.json",
)


# ============================================================
# CURRENT BEST 10 FEATURES
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

def find_best_threshold(y_true, probabilities):

    best_threshold = 0.50
    best_accuracy = -1.0

    for threshold in np.arange(0.20, 0.81, 0.005):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            y_true,
            predictions,
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)

    return best_threshold, best_accuracy


# ============================================================
# HARD EXAMPLE WEIGHTS
# ============================================================

def create_sample_weights(probabilities, strength):

    difficulty = (
        1.0
        - np.abs(probabilities - 0.5) * 2.0
    )

    difficulty = np.clip(
        difficulty,
        0.0,
        1.0,
    )

    weights = (
        1.0 + strength * difficulty
    )

    return weights


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "ECO-SENTINEL HARD-EXAMPLE WEIGHTED RANDOM FOREST"
    )
    print("=" * 75)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(DATASET_PATH)

    print(
        f"\nDataset: {len(df):,} samples"
    )

    X = df[FEATURE_COLUMNS]

    y = df["label"].astype(int)

    # ========================================================
    # FIXED SPLIT
    # ========================================================

    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
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
        "STEP 1 — TRAIN BASE RANDOM FOREST"
    )
    print("=" * 75)

    base_model = RandomForestClassifier(
        n_estimators=300,
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

    train_probability = (
        base_model.predict_proba(
            X_train
        )[:, 1]
    )

    val_probability = (
        base_model.predict_proba(
            X_val
        )[:, 1]
    )

    base_threshold, base_val_accuracy = (
        find_best_threshold(
            y_val,
            val_probability,
        )
    )

    print(
        f"\nBase validation accuracy: "
        f"{base_val_accuracy:.4f}"
    )

    print(
        f"Base threshold: "
        f"{base_threshold:.3f}"
    )

    # ========================================================
    # STEP 2 — HARD EXAMPLES
    # ========================================================

    print("\n" + "=" * 75)
    print(
        "STEP 2 — FIND HARD TRAINING EXAMPLES"
    )
    print("=" * 75)

    print(
        "\nCreating weights from training predictions..."
    )

    # Important:
    # Random Forest training predictions can be too confident.
    # Use out-of-bag probabilities to estimate difficulty.

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

    print(
        f"Average OOB probability: "
        f"{oob_probability.mean():.4f}"
    )

    # ========================================================
    # STEP 3 — SEARCH HARD-EXAMPLE STRENGTH
    # ========================================================

    print("\n" + "=" * 75)
    print(
        "STEP 3 — TEST HARD-EXAMPLE STRENGTH"
    )
    print("=" * 75)

    strengths = [
        0.25,
        0.50,
        0.75,
        1.00,
        1.50,
        2.00,
        3.00,
    ]

    best_strength = 0.0
    best_validation_accuracy = base_val_accuracy
    best_validation_threshold = base_threshold

    for strength in strengths:

        sample_weights = create_sample_weights(
            probabilities=oob_probability,
            strength=strength,
        )

        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=40,
            min_samples_leaf=1,
            max_features=1.0,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
        )

        probability = (
            model.predict_proba(
                X_val
            )[:, 1]
        )

        threshold, accuracy = (
            find_best_threshold(
                y_val,
                probability,
            )
        )

        print(
            f"Strength={strength:<4} "
            f"Validation accuracy="
            f"{accuracy:.4f} "
            f"Threshold={threshold:.3f}"
        )

        if accuracy > best_validation_accuracy:

            best_validation_accuracy = accuracy
            best_strength = strength
            best_validation_threshold = threshold

    print("\n" + "=" * 75)
    print(
        "BEST HARD-EXAMPLE CONFIGURATION"
    )
    print("=" * 75)

    print(
        f"Strength  : {best_strength}"
    )

    print(
        f"Threshold : "
        f"{best_validation_threshold:.3f}"
    )

    print(
        f"Validation accuracy : "
        f"{best_validation_accuracy:.4f}"
    )

    # ========================================================
    # STEP 4 — OOB WEIGHTS FOR COMPLETE DEVELOPMENT SET
    # ========================================================

    print("\n" + "=" * 75)
    print(
        "STEP 4 — BUILD DEVELOPMENT WEIGHTS"
    )
    print("=" * 75)

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
        X_dev,
        y_dev,
    )

    dev_oob_probability = (
        dev_oob_model.oob_decision_function_[:, 1]
    )

    dev_weights = create_sample_weights(
        probabilities=dev_oob_probability,
        strength=best_strength,
    )

    print(
        f"Average weight : "
        f"{dev_weights.mean():.4f}"
    )

    print(
        f"Minimum weight : "
        f"{dev_weights.min():.4f}"
    )

    print(
        f"Maximum weight : "
        f"{dev_weights.max():.4f}"
    )

    # ========================================================
    # STEP 5 — FINAL MODEL
    # ========================================================

    print("\n" + "=" * 75)
    print(
        "STEP 5 — TRAIN FINAL HARD-EXAMPLE RF"
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
        X_dev,
        y_dev,
        sample_weight=dev_weights,
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
        >= best_validation_threshold
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
        f"Threshold : "
        f"{best_validation_threshold:.3f}"
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
    # SAVE
    # ========================================================

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": final_model,
            "threshold": best_validation_threshold,
            "features": FEATURE_COLUMNS,
            "hard_example_strength": best_strength,
        },
        MODEL_PATH,
    )

    metrics = {
        "model": "Hard-Example Weighted Random Forest",
        "random_state": 42,
        "features": FEATURE_COLUMNS,
        "feature_count": len(FEATURE_COLUMNS),
        "development_samples": len(X_dev),
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "testing_samples": len(X_test),
        "hard_example_strength": float(
            best_strength
        ),
        "decision_threshold": float(
            best_validation_threshold
        ),
        "validation_accuracy": float(
            best_validation_accuracy
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
        "model_path": MODEL_PATH,
        "dataset": DATASET_PATH,
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
        "HARD-EXAMPLE MODEL SAVED"
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