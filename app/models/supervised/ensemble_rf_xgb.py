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

from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = r".\data\supervised\training_samples.csv"

MODEL_DIR = r".\data\supervised"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "deforestation_rf_xgb_ensemble.joblib",
)

METRICS_PATH = os.path.join(
    MODEL_DIR,
    "rf_xgb_ensemble_metrics.json",
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
# FIND BEST WEIGHT + THRESHOLD
# VALIDATION ONLY
# ============================================================

def find_best_ensemble(
    y_true,
    rf_probability,
    xgb_probability,
):
    best_accuracy = -1.0
    best_weight = 0.5
    best_threshold = 0.5

    # RF weight.
    # XGBoost weight = 1 - RF weight.
    weights = np.arange(0.0, 1.01, 0.05)

    thresholds = np.arange(0.20, 0.81, 0.005)

    for rf_weight in weights:

        xgb_weight = 1.0 - rf_weight

        combined_probability = (
            rf_weight * rf_probability
            + xgb_weight * xgb_probability
        )

        for threshold in thresholds:

            predictions = (
                combined_probability >= threshold
            ).astype(int)

            accuracy = accuracy_score(
                y_true,
                predictions,
            )

            if accuracy > best_accuracy:

                best_accuracy = accuracy
                best_weight = float(rf_weight)
                best_threshold = float(threshold)

    return (
        best_weight,
        1.0 - best_weight,
        best_threshold,
        best_accuracy,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("ECO-SENTINEL RF + XGBOOST ENSEMBLE")
    print("=" * 75)

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    df = pd.read_csv(DATASET_PATH)

    print(
        f"\nDataset: {df.shape[0]:,} samples"
    )

    print(
        f"Features: {len(FEATURE_COLUMNS)}"
    )

    X = df[FEATURE_COLUMNS]

    y = df["label"].astype(int)

    print(
        f"Class distribution: "
        f"{y.value_counts().to_dict()}"
    )

    # --------------------------------------------------------
    # 80% DEVELOPMENT / 20% FINAL TEST
    # --------------------------------------------------------

    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # --------------------------------------------------------
    # 80% TRAIN / 20% VALIDATION
    # --------------------------------------------------------

    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=0.20,
        random_state=42,
        stratify=y_dev,
    )

    print("\nDataset split:")
    print(
        f"Training:   {len(X_train):,}"
    )
    print(
        f"Validation: {len(X_val):,}"
    )
    print(
        f"Final test: {len(X_test):,}"
    )

    # ========================================================
    # MODEL 1 — RANDOM FOREST
    # ========================================================

    print("\n" + "=" * 75)
    print("MODEL 1 — RANDOM FOREST")
    print("=" * 75)

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=40,
        min_samples_leaf=1,
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining Random Forest...")

    rf_model.fit(
        X_train,
        y_train,
    )

    print("Random Forest training complete.")

    # ========================================================
    # MODEL 2 — XGBOOST
    # ========================================================

    print("\n" + "=" * 75)
    print("MODEL 2 — XGBOOST")
    print("=" * 75)

    xgb_model = XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining XGBoost...")

    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_val, y_val)
        ],
        verbose=False,
    )

    print("XGBoost training complete.")

    # ========================================================
    # VALIDATION PROBABILITIES
    # ========================================================

    print("\n" + "=" * 75)
    print("ENSEMBLE OPTIMIZATION")
    print("=" * 75)

    rf_val_probability = rf_model.predict_proba(
        X_val
    )[:, 1]

    xgb_val_probability = xgb_model.predict_proba(
        X_val
    )[:, 1]

    (
        best_rf_weight,
        best_xgb_weight,
        best_threshold,
        validation_accuracy,
    ) = find_best_ensemble(
        y_val,
        rf_val_probability,
        xgb_val_probability,
    )

    print(
        f"\nBest RF weight      : "
        f"{best_rf_weight:.2f}"
    )

    print(
        f"Best XGBoost weight : "
        f"{best_xgb_weight:.2f}"
    )

    print(
        f"Best threshold      : "
        f"{best_threshold:.3f}"
    )

    print(
        f"Validation accuracy : "
        f"{validation_accuracy:.4f}"
    )

    # ========================================================
    # RETRAIN BOTH MODELS ON COMPLETE DEVELOPMENT SET
    # ========================================================

    print("\n" + "=" * 75)
    print("RETRAINING ON DEVELOPMENT DATA")
    print("=" * 75)

    final_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=40,
        min_samples_leaf=1,
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
    )

    final_xgb = XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining final Random Forest...")

    final_rf.fit(
        X_dev,
        y_dev,
    )

    print("Training final XGBoost...")

    final_xgb.fit(
        X_dev,
        y_dev,
    )

    print("Final models trained.")

    # ========================================================
    # FINAL TEST
    # ========================================================

    print("\n" + "=" * 75)
    print("FINAL TEST — COMPLETELY UNTOUCHED")
    print("=" * 75)

    rf_test_probability = final_rf.predict_proba(
        X_test
    )[:, 1]

    xgb_test_probability = final_xgb.predict_proba(
        X_test
    )[:, 1]

    ensemble_probability = (
        best_rf_weight * rf_test_probability
        + best_xgb_weight * xgb_test_probability
    )

    predictions = (
        ensemble_probability >= best_threshold
    ).astype(int)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

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
        ensemble_probability,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "=" * 75)
    print("🔥 RF + XGBOOST ENSEMBLE RESULTS 🔥")
    print("=" * 75)

    print(
        f"\nAccuracy  : {accuracy:.4f}"
        f"  ({accuracy * 100:.2f}%)"
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
        f"\nRF weight      : {best_rf_weight:.2f}"
    )

    print(
        f"XGBoost weight : {best_xgb_weight:.2f}"
    )

    print(
        f"Threshold      : {best_threshold:.3f}"
    )

    print("\nConfusion Matrix:")

    print(matrix)

    print("\nClassification Report:")

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
    # SAVE MODELS
    # ========================================================

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    joblib.dump(
        {
            "random_forest": final_rf,
            "xgboost": final_xgb,
            "rf_weight": best_rf_weight,
            "xgb_weight": best_xgb_weight,
            "threshold": best_threshold,
            "features": FEATURE_COLUMNS,
        },
        MODEL_PATH,
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics = {
        "model": "Random Forest + XGBoost Ensemble",
        "random_state": 42,
        "features": FEATURE_COLUMNS,
        "training_samples": int(len(X_dev)),
        "validation_samples": int(len(X_val)),
        "testing_samples": int(len(X_test)),
        "rf_weight": float(best_rf_weight),
        "xgb_weight": float(best_xgb_weight),
        "decision_threshold": float(best_threshold),
        "validation_accuracy": float(validation_accuracy),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
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
    print("ENSEMBLE SAVED")
    print("=" * 75)

    print(
        f"\nModel  : {MODEL_PATH}"
    )

    print(
        f"Metrics: {METRICS_PATH}"
    )


if __name__ == "__main__":
    main()