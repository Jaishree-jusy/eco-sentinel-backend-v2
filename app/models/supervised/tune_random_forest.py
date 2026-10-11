"""
ECO-SENTINEL Random Forest Hyperparameter Tuning

Purpose:
    Find a better Random Forest configuration than the current
    baseline model while keeping the same 80/20 stratified
    train/test evaluation methodology.

IMPORTANT:
    The final test set is NOT used during tuning.
"""

from __future__ import annotations

import json
import os

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split, ParameterGrid

from app.models.supervised.random_forest import FEATURE_COLUMNS


DATASET_PATH = r".\data\supervised\training_samples.csv"
MODEL_DIR = r".\data\supervised"

TUNED_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "deforestation_random_forest_tuned.joblib",
)

TUNING_RESULTS_PATH = os.path.join(
    MODEL_DIR,
    "rf_tuning_results.json",
)


def main() -> None:

    print("=" * 70)
    print("ECO-SENTINEL RANDOM FOREST HYPERPARAMETER TUNING")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    df = pd.read_csv(DATASET_PATH)

    X = df[FEATURE_COLUMNS]
    y = df["label"].astype(int)

    print(f"Dataset samples : {len(df):,}")
    print(f"Features         : {len(FEATURE_COLUMNS)}")

    # ---------------------------------------------------------
    # Keep a completely untouched final test set.
    #
    # 80% = development data
    # 20% = final test data
    # ---------------------------------------------------------

    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # ---------------------------------------------------------
    # Split development data again:
    #
    # 80% development -> training
    # 20% development -> validation
    #
    # This prevents the final test set from influencing tuning.
    # ---------------------------------------------------------

    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=0.20,
        random_state=42,
        stratify=y_dev,
    )

    print(f"Training samples : {len(X_train):,}")
    print(f"Validation       : {len(X_val):,}")
    print(f"Final test       : {len(X_test):,}")

    # ---------------------------------------------------------
    # Hyperparameter search space
    # ---------------------------------------------------------

    parameter_grid = {
        "n_estimators": [300, 500],
        "max_depth": [None, 20, 30],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", 0.7],
    }

    combinations = list(ParameterGrid(parameter_grid))

    print(f"\nConfigurations to test: {len(combinations)}")
    print("=" * 70)

    results = []

    # ---------------------------------------------------------
    # Tune using validation set
    # ---------------------------------------------------------

    for i, params in enumerate(combinations, start=1):

        print(
            f"\n[{i}/{len(combinations)}] Testing: {params}"
        )

        model = RandomForestClassifier(
            **params,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(X_val)

        accuracy = accuracy_score(
            y_val,
            predictions,
        )

        precision = precision_score(
            y_val,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_val,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_val,
            predictions,
            zero_division=0,
        )

        print(
            f"Accuracy={accuracy:.4f} | "
            f"Precision={precision:.4f} | "
            f"Recall={recall:.4f} | "
            f"F1={f1:.4f}"
        )

        results.append(
            {
                "params": params,
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
            }
        )

    # ---------------------------------------------------------
    # Find best configuration using validation accuracy
    # ---------------------------------------------------------

    results.sort(
        key=lambda x: x["accuracy"],
        reverse=True,
    )

    best = results[0]

    print("\n" + "=" * 70)
    print("BEST VALIDATION CONFIGURATION")
    print("=" * 70)

    print(f"Parameters: {best['params']}")
    print(f"Accuracy  : {best['accuracy']:.4f}")
    print(f"Precision : {best['precision']:.4f}")
    print(f"Recall    : {best['recall']:.4f}")
    print(f"F1 Score  : {best['f1_score']:.4f}")

    # ---------------------------------------------------------
    # Retrain best configuration on ALL development data.
    #
    # Final test set is still untouched.
    # ---------------------------------------------------------

    print("\nRetraining best configuration on development data...")

    best_model = RandomForestClassifier(
        **best["params"],
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    best_model.fit(
        X_dev,
        y_dev,
    )

    # ---------------------------------------------------------
    # FINAL TEST EVALUATION
    # ---------------------------------------------------------

    print("\nEvaluating on untouched final test set...")

    test_predictions = best_model.predict(X_test)

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

    test_matrix = confusion_matrix(
        y_test,
        test_predictions,
    )

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(f"Accuracy : {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall   : {test_recall:.4f}")
    print(f"F1 Score : {test_f1:.4f}")

    print("\nConfusion Matrix:")
    print(test_matrix)

    # ---------------------------------------------------------
    # Save tuned model
    # ---------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    joblib.dump(
        best_model,
        TUNED_MODEL_PATH,
    )

    # ---------------------------------------------------------
    # Save tuning results
    # ---------------------------------------------------------

    output = {
        "dataset": DATASET_PATH,
        "features": FEATURE_COLUMNS,
        "development_samples": int(len(X_dev)),
        "training_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "final_test_samples": int(len(X_test)),
        "best_parameters": best["params"],
        "best_validation_metrics": {
            "accuracy": best["accuracy"],
            "precision": best["precision"],
            "recall": best["recall"],
            "f1_score": best["f1_score"],
        },
        "final_test_metrics": {
            "accuracy": float(test_accuracy),
            "precision": float(test_precision),
            "recall": float(test_recall),
            "f1_score": float(test_f1),
            "confusion_matrix": test_matrix.tolist(),
        },
        "all_validation_results": results,
        "model_path": TUNED_MODEL_PATH,
    }

    with open(
        TUNING_RESULTS_PATH,
        "w",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("TUNED MODEL SAVED")
    print("=" * 70)

    print(f"Model   : {TUNED_MODEL_PATH}")
    print(f"Results : {TUNING_RESULTS_PATH}")


if __name__ == "__main__":
    main()