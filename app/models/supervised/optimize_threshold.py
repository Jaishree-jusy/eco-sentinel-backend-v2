"""
ECO-SENTINEL
Final Threshold Optimization Experiment

Uses the existing training_samples.csv.
No new dataset is created.

Protocol:
    1. 80% development / 20% untouched test
    2. 80% of development for training / 20% validation
    3. Optimize threshold ONLY on validation
    4. Retrain on complete development set
    5. Evaluate untouched test exactly once

Purpose:
    Determine whether the current Random Forest probability
    output can achieve >= 90% accuracy through a better
    decision threshold.
"""

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
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT / "data"
SUP_DIR = DATA_DIR / "supervised"

CSV_PATH = SUP_DIR / "training_samples.csv"

MODEL_PATH = (
    SUP_DIR /
    "deforestation_rf_threshold_optimized.joblib"
)

METRICS_PATH = (
    SUP_DIR /
    "rf_threshold_optimized_metrics.json"
)


# ============================================================
# CURRENT BEST FEATURE SET
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
    """
    Select threshold using validation data only.

    Uses a fine-grained search from 0.20 to 0.80.
    """

    best_threshold = 0.50
    best_accuracy = -1.0

    results = []

    thresholds = np.arange(
        0.20,
        0.801,
        0.005,
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            y_true,
            predictions,
        )

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        results.append(
            {
                "threshold": float(threshold),
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_threshold = float(
                threshold
            )

    return (
        best_threshold,
        best_accuracy,
        results,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "ECO-SENTINEL — FINAL THRESHOLD OPTIMIZATION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading existing dataset...")

    df = pd.read_csv(
        CSV_PATH
    )

    print(
        "Dataset shape:",
        df.shape,
    )

    print(
        "Labels:",
        df["label"].value_counts().to_dict(),
    )

    # --------------------------------------------------------
    # Verify features
    # --------------------------------------------------------

    missing = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Missing required features: "
            f"{missing}"
        )

    X = df[
        FEATURE_COLUMNS
    ].copy()

    y = df[
        "label"
    ].astype(int)

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X = X.fillna(0)

    # --------------------------------------------------------
    # First split
    #
    # 80% development
    # 20% untouched test
    # --------------------------------------------------------

    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # --------------------------------------------------------
    # Second split
    #
    # 80% train
    # 20% validation
    # --------------------------------------------------------

    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=0.20,
        random_state=42,
        stratify=y_dev,
    )

    print(
        "\nDevelopment samples:",
        len(X_dev),
    )

    print(
        "Training samples:   ",
        len(X_train),
    )

    print(
        "Validation samples: ",
        len(X_val),
    )

    print(
        "Untouched test:      ",
        len(X_test),
    )

    # --------------------------------------------------------
    # Train current best RF
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "TRAINING CURRENT BEST RANDOM FOREST"
    )
    print("=" * 70)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=40,
        min_samples_leaf=1,
        max_features=1.0,
        class_weight=None,
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining...")

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Validation probabilities
    # --------------------------------------------------------

    print(
        "\nGenerating validation probabilities..."
    )

    val_probabilities = model.predict_proba(
        X_val
    )[:, 1]

    # --------------------------------------------------------
    # Optimize threshold
    # --------------------------------------------------------

    print(
        "\nSearching thresholds from "
        "0.20 to 0.80..."
    )

    (
        best_threshold,
        best_validation_accuracy,
        threshold_results,
    ) = find_best_threshold(
        y_val,
        val_probabilities,
    )

    val_predictions = (
        val_probabilities
        >= best_threshold
    ).astype(int)

    val_precision = precision_score(
        y_val,
        val_predictions,
        zero_division=0,
    )

    val_recall = recall_score(
        y_val,
        val_predictions,
        zero_division=0,
    )

    val_f1 = f1_score(
        y_val,
        val_predictions,
        zero_division=0,
    )

    val_auc = roc_auc_score(
        y_val,
        val_probabilities,
    )

    print("\n" + "=" * 70)
    print(
        "BEST VALIDATION THRESHOLD"
    )
    print("=" * 70)

    print(
        f"\nThreshold:  {best_threshold:.3f}"
    )

    print(
        f"Accuracy:   {best_validation_accuracy:.4f}"
    )

    print(
        f"Precision:  {val_precision:.4f}"
    )

    print(
        f"Recall:     {val_recall:.4f}"
    )

    print(
        f"F1:         {val_f1:.4f}"
    )

    print(
        f"ROC-AUC:    {val_auc:.4f}"
    )

    # --------------------------------------------------------
    # Show top threshold candidates
    # --------------------------------------------------------

    ranked_thresholds = sorted(
        threshold_results,
        key=lambda item: item["accuracy"],
        reverse=True,
    )

    print(
        "\nTop 10 validation thresholds:"
    )

    for item in ranked_thresholds[:10]:

        print(
            f"  {item['threshold']:.3f}"
            f" -> accuracy "
            f"{item['accuracy']:.4f}"
            f", precision "
            f"{item['precision']:.4f}"
            f", recall "
            f"{item['recall']:.4f}"
        )

    # --------------------------------------------------------
    # Retrain using ALL development data
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "RETRAINING ON COMPLETE DEVELOPMENT DATA"
    )
    print("=" * 70)

    final_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=40,
        min_samples_leaf=1,
        max_features=1.0,
        class_weight=None,
        random_state=42,
        n_jobs=-1,
    )

    final_model.fit(
        X_dev,
        y_dev,
    )

    # --------------------------------------------------------
    # FINAL UNTOUCHED TEST
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "FINAL UNTOUCHED TEST"
    )
    print("=" * 70)

    test_probabilities = final_model.predict_proba(
        X_test
    )[:, 1]

    test_predictions = (
        test_probabilities
        >= best_threshold
    ).astype(int)

    test_accuracy = accuracy_score(
        y_test,
        test_predictions,
    )

    test_precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    test_recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    test_f1 = f1_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    test_auc = roc_auc_score(
        y_test,
        test_probabilities,
    )

    test_cm = confusion_matrix(
        y_test,
        test_predictions,
    )

    print(
        f"\nThreshold:  {best_threshold:.3f}"
    )

    print(
        f"\nAccuracy:   {test_accuracy:.4f}"
    )

    print(
        f"Precision:  {test_precision:.4f}"
    )

    print(
        f"Recall:     {test_recall:.4f}"
    )

    print(
        f"F1 Score:   {test_f1:.4f}"
    )

    print(
        f"ROC-AUC:    {test_auc:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        test_cm
    )

    # --------------------------------------------------------
    # Compare target
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "90% TARGET CHECK"
    )
    print("=" * 70)

    if test_accuracy >= 0.90:

        print(
            "\n🎯 TARGET ACHIEVED!"
        )

        print(
            f"Final accuracy: "
            f"{test_accuracy * 100:.2f}%"
        )

    else:

        print(
            "\nTarget not reached in this experiment."
        )

        print(
            f"Final accuracy: "
            f"{test_accuracy * 100:.2f}%"
        )

        print(
            f"Gap to 90%: "
            f"{(0.90 - test_accuracy) * 100:.2f} percentage points"
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(
        {
            "model": final_model,
            "features": FEATURE_COLUMNS,
            "threshold": best_threshold,
        },
        MODEL_PATH,
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics = {
        "model": "RandomForest Threshold Optimized",

        "features": FEATURE_COLUMNS,

        "n_features":
            len(FEATURE_COLUMNS),

        "n_estimators":
            300,

        "max_depth":
            40,

        "min_samples_leaf":
            1,

        "max_features":
            1.0,

        "threshold":
            best_threshold,

        "validation_accuracy":
            float(best_validation_accuracy),

        "validation_precision":
            float(val_precision),

        "validation_recall":
            float(val_recall),

        "validation_f1":
            float(val_f1),

        "validation_roc_auc":
            float(val_auc),

        "test_accuracy":
            float(test_accuracy),

        "test_precision":
            float(test_precision),

        "test_recall":
            float(test_recall),

        "test_f1":
            float(test_f1),

        "test_roc_auc":
            float(test_auc),

        "confusion_matrix":
            test_cm.tolist(),

        "target_accuracy":
            0.90,

        "target_achieved":
            bool(test_accuracy >= 0.90),
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print(
        "EXPERIMENT COMPLETE"
    )
    print("=" * 70)

    print(
        "\nModel saved:",
        MODEL_PATH,
    )

    print(
        "Metrics saved:",
        METRICS_PATH,
    )


if __name__ == "__main__":
    main()