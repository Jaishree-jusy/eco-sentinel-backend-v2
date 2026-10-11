from pathlib import Path
import json

import numpy as np
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

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data" / "supervised"

DATASET = DATA_DIR / "training_samples.csv"

MODEL_OUT = DATA_DIR / "deforestation_rf_reduced_features.joblib"
METRICS_OUT = DATA_DIR / "rf_reduced_features_metrics.json"


# ============================================================
# FEATURE SETS
# ============================================================

FEATURE_SETS = {

    # Current best model
    "current_10": [
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
    ],

    # Strongest spectral/index features
    "strong_6": [
        "delta_B11",
        "delta_B12",
        "delta_ndvi",
        "delta_nbr",
        "delta_ndwi",
        "delta_ndmi",
    ],

    # Strongest indices only
    "indices_4": [
        "delta_ndvi",
        "delta_nbr",
        "delta_ndwi",
        "delta_ndmi",
    ],

    # Strongest SWIR + indices
    "swir_indices_6": [
        "delta_B11",
        "delta_B12",
        "delta_ndvi",
        "delta_nbr",
        "delta_ndwi",
        "delta_ndmi",
    ],

    # Strongest bands + most useful indices
    "bands_indices_8": [
        "delta_B11",
        "delta_B12",
        "delta_B5",
        "delta_B9",
        "delta_ndvi",
        "delta_nbr",
        "delta_ndwi",
        "delta_ndmi",
    ],
}


# ============================================================
# RF CONFIGURATION
# ============================================================

RF_CONFIG = {
    "n_estimators": 300,
    "max_depth": 40,
    "min_samples_leaf": 1,
    "max_features": 1.0,
    "class_weight": None,
    "random_state": 42,
    "n_jobs": -1,
}


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

def find_best_threshold(y_true, probabilities):

    best_threshold = 0.50
    best_accuracy = -1.0

    for threshold in np.arange(
        0.30,
        0.71,
        0.01
    ):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            y_true,
            predictions
        )

        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_threshold = float(threshold)

    return best_threshold


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ECO-SENTINEL — REDUCED FEATURE RF EXPERIMENT")
    print("=" * 70)

    print()
    print("Loading dataset...")

    df = __import__("pandas").read_csv(DATASET)

    print(
        f"Dataset shape: {df.shape}"
    )

    y = df["label"].astype(int).values

    print(
        "Labels:",
        dict(
            zip(
                *np.unique(
                    y,
                    return_counts=True
                )
            )
        )
    )

    # --------------------------------------------------------
    # SAME DEVELOPMENT / TEST SPLIT USED PREVIOUSLY
    # --------------------------------------------------------

    indices = np.arange(
        len(df)
    )

    dev_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    train_idx, val_idx = train_test_split(
        dev_idx,
        test_size=0.20,
        random_state=42,
        stratify=y[dev_idx]
    )

    print()
    print(
        f"Development samples: {len(dev_idx):,}"
    )

    print(
        f"Training samples:    {len(train_idx):,}"
    )

    print(
        f"Validation samples:  {len(val_idx):,}"
    )

    print(
        f"Test samples:        {len(test_idx):,}"
    )

    results = {}

    best_name = None
    best_val_accuracy = -1.0
    best_threshold = 0.50
    best_features = None
    best_model = None

    # ========================================================
    # TEST EACH FEATURE SET
    # ========================================================

    for name, features in FEATURE_SETS.items():

        print()
        print("=" * 70)
        print(
            f"FEATURE SET: {name}"
        )
        print("=" * 70)

        print(
            f"Number of features: {len(features)}"
        )

        print(
            features
        )

        X = df[
            features
        ].astype(
            np.float32
        ).values

        X_train = X[train_idx]
        y_train = y[train_idx]

        X_val = X[val_idx]
        y_val = y[val_idx]

        print()
        print("Training RF...")

        model = RandomForestClassifier(
            **RF_CONFIG
        )

        model.fit(
            X_train,
            y_train
        )

        val_probabilities = (
            model.predict_proba(
                X_val
            )[:, 1]
        )

        threshold = find_best_threshold(
            y_val,
            val_probabilities
        )

        val_predictions = (
            val_probabilities
            >= threshold
        ).astype(int)

        val_accuracy = accuracy_score(
            y_val,
            val_predictions
        )

        val_precision = precision_score(
            y_val,
            val_predictions,
            zero_division=0
        )

        val_recall = recall_score(
            y_val,
            val_predictions,
            zero_division=0
        )

        val_f1 = f1_score(
            y_val,
            val_predictions,
            zero_division=0
        )

        val_auc = roc_auc_score(
            y_val,
            val_probabilities
        )

        print()
        print(
            f"Validation threshold: {threshold:.2f}"
        )

        print(
            f"Validation accuracy:  "
            f"{val_accuracy:.4f}"
        )

        print(
            f"Validation precision: "
            f"{val_precision:.4f}"
        )

        print(
            f"Validation recall:    "
            f"{val_recall:.4f}"
        )

        print(
            f"Validation F1:        "
            f"{val_f1:.4f}"
        )

        print(
            f"Validation ROC-AUC:   "
            f"{val_auc:.4f}"
        )

        results[name] = {

            "features": features,

            "num_features":
                len(features),

            "validation_accuracy":
                float(val_accuracy),

            "validation_precision":
                float(val_precision),

            "validation_recall":
                float(val_recall),

            "validation_f1":
                float(val_f1),

            "validation_roc_auc":
                float(val_auc),

            "threshold":
                float(threshold),
        }

        # ----------------------------------------------------
        # Select using VALIDATION only
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            best_name = name

            best_threshold = threshold

            best_features = features

            best_model = model

    # ========================================================
    # BEST FEATURE SET
    # ========================================================

    print()
    print("=" * 70)
    print("BEST FEATURE SET")
    print("=" * 70)

    print(
        f"Selected: {best_name}"
    )

    print(
        f"Validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print(
        f"Threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        "Features:"
    )

    for feature in best_features:
        print(
            f"  - {feature}"
        )

    # ========================================================
    # RETRAIN BEST FEATURE SET ON ALL DEVELOPMENT DATA
    # ========================================================

    print()
    print("=" * 70)
    print(
        "RETRAINING BEST MODEL ON DEVELOPMENT DATA"
    )
    print("=" * 70)

    X_best = df[
        best_features
    ].astype(
        np.float32
    ).values

    X_dev = X_best[dev_idx]
    y_dev = y[dev_idx]

    X_test = X_best[test_idx]
    y_test = y[test_idx]

    final_model = RandomForestClassifier(
        **RF_CONFIG
    )

    final_model.fit(
        X_dev,
        y_dev
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL UNTOUCHED TEST")
    print("=" * 70)

    test_probabilities = (
        final_model.predict_proba(
            X_test
        )[:, 1]
    )

    test_predictions = (
        test_probabilities
        >= best_threshold
    ).astype(int)

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    test_precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    test_recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    test_f1 = f1_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    test_auc = roc_auc_score(
        y_test,
        test_probabilities
    )

    cm = confusion_matrix(
        y_test,
        test_predictions
    )

    print()
    print(
        f"Feature set: {best_name}"
    )

    print(
        f"Number of features: "
        f"{len(best_features)}"
    )

    print(
        f"Threshold: "
        f"{best_threshold:.2f}"
    )

    print()
    print(
        f"Accuracy:  {test_accuracy:.4f}"
    )

    print(
        f"Precision: {test_precision:.4f}"
    )

    print(
        f"Recall:    {test_recall:.4f}"
    )

    print(
        f"F1 Score:  {test_f1:.4f}"
    )

    print(
        f"ROC-AUC:   {test_auc:.4f}"
    )

    print()
    print("Confusion Matrix:")

    print(cm)

    # ========================================================
    # SAVE MODEL
    # ========================================================

    import joblib

    package = {

        "model": final_model,

        "features": best_features,

        "threshold": best_threshold,

        "feature_set_name": best_name
    }

    joblib.dump(
        package,
        MODEL_OUT
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    output = {

        "experiment":
            "Reduced feature Random Forest",

        "best_feature_set":
            best_name,

        "best_features":
            best_features,

        "num_features":
            len(best_features),

        "rf_config":
            RF_CONFIG,

        "threshold":
            float(best_threshold),

        "validation_accuracy":
            float(best_val_accuracy),

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
            cm.tolist(),

        "all_feature_set_results":
            results,

        "previous_best_accuracy":
            0.8873
    }

    with open(
        METRICS_OUT,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2
        )

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    print(
        f"Model saved: {MODEL_OUT}"
    )

    print(
        f"Metrics saved: {METRICS_OUT}"
    )


if __name__ == "__main__":
    main()