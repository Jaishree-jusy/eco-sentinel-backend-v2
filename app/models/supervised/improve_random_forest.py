from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# ============================================================
# ECO-SENTINEL
# TARGETED RANDOM FOREST IMPROVEMENT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data" / "supervised"

DATASET_PATH = DATA_DIR / "training_samples.csv"

MODEL_PATH = DATA_DIR / "deforestation_random_forest_improved.joblib"
METRICS_PATH = DATA_DIR / "rf_improved_metrics.json"

RANDOM_SEED = 42


# ============================================================
# FEATURES
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

TARGET_COLUMN = "label"


# ============================================================
# TARGETED CONFIGURATIONS
# ============================================================

CONFIGURATIONS = [

    # Strong baseline neighborhood
    {
        "n_estimators": 300,
        "max_depth": 40,
        "min_samples_leaf": 1,
        "max_features": 1.0,
        "class_weight": None,
    },

    {
        "n_estimators": 500,
        "max_depth": 40,
        "min_samples_leaf": 1,
        "max_features": 1.0,
        "class_weight": None,
    },

    {
        "n_estimators": 800,
        "max_depth": 40,
        "min_samples_leaf": 1,
        "max_features": 1.0,
        "class_weight": None,
    },

    # Slightly shallower
    {
        "n_estimators": 500,
        "max_depth": 30,
        "min_samples_leaf": 1,
        "max_features": 1.0,
        "class_weight": None,
    },

    {
        "n_estimators": 500,
        "max_depth": 50,
        "min_samples_leaf": 1,
        "max_features": 1.0,
        "class_weight": None,
    },

    # Feature subsampling
    {
        "n_estimators": 500,
        "max_depth": 40,
        "min_samples_leaf": 1,
        "max_features": 0.7,
        "class_weight": None,
    },

    {
        "n_estimators": 800,
        "max_depth": 40,
        "min_samples_leaf": 1,
        "max_features": 0.7,
        "class_weight": None,
    },

    # Slight regularization
    {
        "n_estimators": 500,
        "max_depth": 40,
        "min_samples_leaf": 2,
        "max_features": 1.0,
        "class_weight": None,
    },

    {
        "n_estimators": 800,
        "max_depth": 40,
        "min_samples_leaf": 2,
        "max_features": 1.0,
        "class_weight": None,
    },

    # Balanced class weighting
    {
        "n_estimators": 500,
        "max_depth": 40,
        "min_samples_leaf": 1,
        "max_features": 1.0,
        "class_weight": "balanced",
    },

    {
        "n_estimators": 800,
        "max_depth": 40,
        "min_samples_leaf": 1,
        "max_features": 0.7,
        "class_weight": "balanced",
    },

    # Deeper forest
    {
        "n_estimators": 800,
        "max_depth": 50,
        "min_samples_leaf": 1,
        "max_features": 1.0,
        "class_weight": None,
    },
]


# ============================================================
# THRESHOLDS
# ============================================================

THRESHOLDS = np.arange(
    0.30,
    0.61,
    0.01,
)


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    print("\nLoading dataset...")

    df = pd.read_csv(DATASET_PATH)

    print("Dataset shape:", df.shape)

    required = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    X = df[FEATURE_COLUMNS].astype(np.float32)
    y = df[TARGET_COLUMN].astype(int)

    print("\nClass distribution:")
    print(y.value_counts().sort_index())

    return X, y


# ============================================================
# SPLIT
# ============================================================

def create_splits(X, y):

    print("\nCreating development/test split...")

    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    print(
        "Development:",
        len(X_dev),
        "| Test:",
        len(X_test),
    )

    print("\nCreating train/validation split...")

    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=0.20,
        stratify=y_dev,
        random_state=RANDOM_SEED,
    )

    print(
        "Train:",
        len(X_train),
        "| Validation:",
        len(X_val),
        "| Test:",
        len(X_test),
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        X_dev,
        y_dev,
    )


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

def optimize_threshold(y_true, probabilities):

    best_threshold = 0.50
    best_accuracy = -1.0
    best_result = None

    all_results = []

    for threshold in THRESHOLDS:

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

        result = {
            "threshold": float(threshold),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

        all_results.append(result)

        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_threshold = float(threshold)
            best_result = result

    return (
        best_threshold,
        best_result,
        all_results,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("=" * 70)
    print("ECO-SENTINEL TARGETED RANDOM FOREST IMPROVEMENT")
    print("=" * 70)

    print(
        "\nConfigurations to test:",
        len(CONFIGURATIONS),
    )

    print("\nFeatures:")

    for feature in FEATURE_COLUMNS:
        print(" ", feature)

    X, y = load_dataset()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        X_dev,
        y_dev,
    ) = create_splits(X, y)


    # ========================================================
    # HYPERPARAMETER SEARCH
    # ========================================================

    print("\n" + "=" * 70)
    print("TESTING RANDOM FOREST CONFIGURATIONS")
    print("=" * 70)

    best_model = None
    best_config = None

    best_val_accuracy = -1.0
    best_val_f1 = -1.0
    best_val_auc = -1.0

    results = []

    for number, config in enumerate(
        CONFIGURATIONS,
        start=1,
    ):

        print("\n" + "-" * 70)

        print(
            f"Configuration {number}/"
            f"{len(CONFIGURATIONS)}"
        )

        print(config)

        model = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_leaf=config["min_samples_leaf"],
            max_features=config["max_features"],
            class_weight=config["class_weight"],
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
        )

        probabilities = model.predict_proba(
            X_val
        )[:, 1]

        predictions = (
            probabilities >= 0.50
        ).astype(int)

        accuracy = accuracy_score(
            y_val,
            predictions,
        )

        f1 = f1_score(
            y_val,
            predictions,
            zero_division=0,
        )

        auc = roc_auc_score(
            y_val,
            probabilities,
        )

        print(
            f"Validation Accuracy: {accuracy:.4f}"
        )

        print(
            f"Validation F1:       {f1:.4f}"
        )

        print(
            f"Validation ROC-AUC:  {auc:.4f}"
        )

        results.append(
            {
                "configuration": config,
                "validation_accuracy": float(
                    accuracy
                ),
                "validation_f1": float(f1),
                "validation_auc": float(auc),
            }
        )

        if accuracy > best_val_accuracy:

            best_val_accuracy = accuracy
            best_val_f1 = f1
            best_val_auc = auc

            best_model = model
            best_config = config


    # ========================================================
    # BEST MODEL
    # ========================================================

    print("\n" + "=" * 70)
    print("BEST VALIDATION MODEL")
    print("=" * 70)

    print("\nConfiguration:")

    print(
        json.dumps(
            best_config,
            indent=2,
        )
    )

    print(
        f"\nValidation Accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print(
        f"Validation F1: "
        f"{best_val_f1:.4f}"
    )

    print(
        f"Validation ROC-AUC: "
        f"{best_val_auc:.4f}"
    )


    # ========================================================
    # THRESHOLD OPTIMIZATION
    # ========================================================

    print("\n" + "=" * 70)
    print("VALIDATION THRESHOLD OPTIMIZATION")
    print("=" * 70)

    val_probabilities = best_model.predict_proba(
        X_val
    )[:, 1]

    (
        best_threshold,
        threshold_result,
        threshold_results,
    ) = optimize_threshold(
        y_val,
        val_probabilities,
    )

    print(
        f"\nBest threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Validation Accuracy: "
        f"{threshold_result['accuracy']:.4f}"
    )

    print(
        f"Validation Precision: "
        f"{threshold_result['precision']:.4f}"
    )

    print(
        f"Validation Recall: "
        f"{threshold_result['recall']:.4f}"
    )

    print(
        f"Validation F1: "
        f"{threshold_result['f1']:.4f}"
    )


    # ========================================================
    # FINAL RETRAIN
    # ========================================================

    print("\n" + "=" * 70)
    print("RETRAINING BEST MODEL")
    print("=" * 70)

    final_model = RandomForestClassifier(
        n_estimators=best_config["n_estimators"],
        max_depth=best_config["max_depth"],
        min_samples_leaf=best_config["min_samples_leaf"],
        max_features=best_config["max_features"],
        class_weight=best_config["class_weight"],
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    final_model.fit(
        X_dev,
        y_dev,
    )

    print(
        "\nRetrained using",
        len(X_dev),
        "development samples.",
    )


    # ========================================================
    # FINAL TEST
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL UNTOUCHED TEST")
    print("=" * 70)

    test_probabilities = final_model.predict_proba(
        X_test
    )[:, 1]

    test_predictions = (
        test_probabilities >= best_threshold
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
        f"\nAccuracy : {test_accuracy:.4f}"
    )

    print(
        f"Precision: {test_precision:.4f}"
    )

    print(
        f"Recall   : {test_recall:.4f}"
    )

    print(
        f"F1       : {test_f1:.4f}"
    )

    print(
        f"ROC-AUC  : {test_auc:.4f}"
    )

    print("\nConfusion matrix:")

    print(test_cm)


    # ========================================================
    # SAVE MODEL
    # ========================================================

    joblib.dump(
        final_model,
        MODEL_PATH,
    )


    # ========================================================
    # SAVE METRICS
    # ========================================================

    elapsed_minutes = (
        time.time() - start_time
    ) / 60

    output = {

        "model":
            "Targeted Improved Random Forest",

        "features":
            FEATURE_COLUMNS,

        "random_seed":
            RANDOM_SEED,

        "best_configuration":
            best_config,

        "best_validation_accuracy":
            float(best_val_accuracy),

        "best_validation_f1":
            float(best_val_f1),

        "best_validation_auc":
            float(best_val_auc),

        "selected_threshold":
            float(best_threshold),

        "threshold_selected_using":
            "validation_set_only",

        "development_samples":
            int(len(X_dev)),

        "test_samples":
            int(len(X_test)),

        "final_test_metrics": {

            "accuracy":
                float(test_accuracy),

            "precision":
                float(test_precision),

            "recall":
                float(test_recall),

            "f1":
                float(test_f1),

            "roc_auc":
                float(test_auc),

            "confusion_matrix":
                test_cm.tolist(),
        },

        "training_time_minutes":
            float(elapsed_minutes),

        "configuration_results":
            results,

        "threshold_results":
            threshold_results,
    }


    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )


    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    print(
        "\nImproved model saved to:"
    )

    print(MODEL_PATH)

    print(
        "\nMetrics saved to:"
    )

    print(METRICS_PATH)

    print(
        f"\nTotal time: "
        f"{elapsed_minutes:.2f} minutes"
    )


if __name__ == "__main__":
    main()