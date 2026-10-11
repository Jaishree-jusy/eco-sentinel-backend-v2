from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
import rasterio

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

SENTINEL_2023 = (
    r".\data\raw_images\forest_samples\Forest_Sample_2023.tif"
)

SENTINEL_2025 = (
    r".\data\raw_images\forest_samples\Forest_Sample_2025.tif"
)

MODEL_DIR = r".\data\supervised"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "deforestation_rf_context.joblib",
)

METRICS_PATH = os.path.join(
    MODEL_DIR,
    "rf_context_metrics.json",
)


# ============================================================
# EXISTING 10 BEST FEATURES
# ============================================================

BASE_FEATURES = [
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
# HELPERS
# ============================================================

EPS = 1e-6


def safe_divide(a, b):
    return np.divide(
        a,
        b,
        out=np.zeros_like(a, dtype=np.float32),
        where=np.abs(b) > EPS,
    )


def normalized_change(a23, a25):
    """
    Symmetric normalized change:

        (2025 - 2023) / (|2025| + |2023|)

    This reduces sensitivity to absolute magnitude.
    """
    denominator = (
        np.abs(a25)
        + np.abs(a23)
        + EPS
    )

    return (
        (a25 - a23)
        / denominator
    ).astype(np.float32)


def calculate_indices(bands):
    """
    Sentinel-2 bands used:

    B1  = index 0
    B2  = index 1
    B3  = index 2
    B4  = index 3
    B5  = index 4
    B6  = index 5
    B7  = index 6
    B8  = index 7
    B8A = index 8
    B9  = index 9
    B11 = index 10
    B12 = index 11
    """

    b3 = bands[2].astype(np.float32)
    b4 = bands[3].astype(np.float32)
    b5 = bands[4].astype(np.float32)
    b8 = bands[7].astype(np.float32)
    b8a = bands[8].astype(np.float32)
    b11 = bands[10].astype(np.float32)
    b12 = bands[11].astype(np.float32)

    ndvi = safe_divide(
        b8 - b4,
        b8 + b4,
    )

    nbr = safe_divide(
        b8 - b12,
        b8 + b12,
    )

    ndwi = safe_divide(
        b3 - b8,
        b3 + b8,
    )

    ndmi = safe_divide(
        b8 - b11,
        b8 + b11,
    )

    return {
        "ndvi": ndvi,
        "nbr": nbr,
        "ndwi": ndwi,
        "ndmi": ndmi,
    }


# ============================================================
# BUILD CONTEXT FEATURES
# ============================================================

def build_features(df):

    print("\nLoading Sentinel-2 rasters...")

    with rasterio.open(SENTINEL_2023) as src23:
        raster23 = src23.read().astype(
            np.float32
        )

    with rasterio.open(SENTINEL_2025) as src25:
        raster25 = src25.read().astype(
            np.float32
        )

    print(
        f"2023 raster shape: {raster23.shape}"
    )

    print(
        f"2025 raster shape: {raster25.shape}"
    )

    rows = df["row"].to_numpy(
        dtype=np.int64
    )

    cols = df["col"].to_numpy(
        dtype=np.int64
    )

    # --------------------------------------------------------
    # Extract the 12 spectral bands
    # --------------------------------------------------------

    spectral23 = raster23[
        :12,
        rows,
        cols,
    ]

    spectral25 = raster25[
        :12,
        rows,
        cols,
    ]

    features = {}

    # --------------------------------------------------------
    # Keep the existing 10 strongest features
    # --------------------------------------------------------

    for feature in BASE_FEATURES:
        features[feature] = df[
            feature
        ].to_numpy(
            dtype=np.float32
        )

    # --------------------------------------------------------
    # BASELINE / ABSOLUTE VEGETATION CONTEXT
    # --------------------------------------------------------

    indices23 = calculate_indices(
        spectral23
    )

    indices25 = calculate_indices(
        spectral25
    )

    print(
        "\nAdding baseline vegetation context..."
    )

    for name in [
        "ndvi",
        "nbr",
        "ndwi",
        "ndmi",
    ]:

        features[
            f"{name}_2023"
        ] = indices23[name]

        features[
            f"{name}_2025"
        ] = indices25[name]

    # --------------------------------------------------------
    # NORMALIZED SPECTRAL CHANGES
    # --------------------------------------------------------

    print(
        "Adding normalized spectral changes..."
    )

    selected_bands = {
        "B1": 0,
        "B4": 3,
        "B5": 4,
        "B9": 9,
        "B11": 10,
        "B12": 11,
    }

    for name, index in selected_bands.items():

        features[
            f"norm_{name}"
        ] = normalized_change(
            spectral23[index],
            spectral25[index],
        )

    # --------------------------------------------------------
    # NORMALIZED INDEX CHANGES
    # --------------------------------------------------------

    for name in [
        "ndvi",
        "nbr",
        "ndwi",
        "ndmi",
    ]:

        features[
            f"norm_{name}"
        ] = normalized_change(
            indices23[name],
            indices25[name],
        )

    # --------------------------------------------------------
    # IMPORTANT SPECTRAL RATIOS
    # --------------------------------------------------------

    print(
        "Adding spectral ratios..."
    )

    features["B12_B8_2023"] = safe_divide(
        spectral23[11],
        spectral23[7],
    )

    features["B12_B8_2025"] = safe_divide(
        spectral25[11],
        spectral25[7],
    )

    features["B11_B8_2023"] = safe_divide(
        spectral23[10],
        spectral23[7],
    )

    features["B11_B8_2025"] = safe_divide(
        spectral25[10],
        spectral25[7],
    )

    # --------------------------------------------------------
    # CREATE FINAL MATRIX
    # --------------------------------------------------------

    feature_names = list(
        features.keys()
    )

    X = np.column_stack(
        [
            features[name]
            for name in feature_names
        ]
    ).astype(np.float32)

    # Replace bad values
    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    print(
        f"\nFinal feature count: "
        f"{len(feature_names)}"
    )

    print(
        f"Final feature matrix: "
        f"{X.shape}"
    )

    return (
        X,
        feature_names,
    )


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

def find_best_threshold(
    y_true,
    probabilities,
):

    best_threshold = 0.50
    best_accuracy = -1.0

    thresholds = np.arange(
        0.20,
        0.81,
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
        "ECO-SENTINEL CONTEXT-AWARE RANDOM FOREST"
    )
    print("=" * 75)

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    df = pd.read_csv(
        DATASET_PATH
    )

    print(
        f"\nDataset: {len(df):,} samples"
    )

    if "row" not in df.columns or "col" not in df.columns:

        raise RuntimeError(
            "row/col columns are missing "
            "from training_samples.csv. "
            "Regenerate the dataset first."
        )

    # --------------------------------------------------------
    # BUILD FEATURES
    # --------------------------------------------------------

    X, feature_names = build_features(
        df
    )

    y = df[
        "label"
    ].to_numpy(
        dtype=np.int32
    )

    print(
        f"\nLabels: "
        f"{np.bincount(y).tolist()}"
    )

    # ========================================================
    # 80% DEVELOPMENT / 20% FINAL TEST
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
    # RANDOM FOREST
    # ========================================================

    print("\n" + "=" * 75)
    print("TRAINING CONTEXT-AWARE RANDOM FOREST")
    print("=" * 75)

    model = RandomForestClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight=None,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Training complete."
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    val_probability = model.predict_proba(
        X_val
    )[:, 1]

    (
        best_threshold,
        validation_accuracy,
    ) = find_best_threshold(
        y_val,
        val_probability,
    )

    validation_predictions = (
        val_probability
        >= best_threshold
    ).astype(int)

    validation_f1 = f1_score(
        y_val,
        validation_predictions,
        zero_division=0,
    )

    print("\n" + "=" * 75)
    print("VALIDATION")
    print("=" * 75)

    print(
        f"Threshold : "
        f"{best_threshold:.3f}"
    )

    print(
        f"Accuracy  : "
        f"{validation_accuracy:.4f}"
    )

    print(
        f"F1        : "
        f"{validation_f1:.4f}"
    )

    # ========================================================
    # RETRAIN ON ALL DEVELOPMENT DATA
    # ========================================================

    print("\n" + "=" * 75)
    print(
        "RETRAINING ON COMPLETE DEVELOPMENT SET"
    )
    print("=" * 75)

    final_model = RandomForestClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight=None,
        random_state=42,
        n_jobs=-1,
    )

    final_model.fit(
        X_dev,
        y_dev,
    )

    print(
        "Final training complete."
    )

    # ========================================================
    # FINAL TEST
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
    # FEATURE IMPORTANCE
    # ========================================================

    importance = dict(
        zip(
            feature_names,
            final_model.feature_importances_,
        )
    )

    importance = dict(
        sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )
    )

    print(
        "\nTop 20 Feature Importances:"
    )

    for name, value in list(
        importance.items()
    )[:20]:

        print(
            f"  {name:<25} "
            f"{value:.5f}"
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
            "features": feature_names,
        },
        MODEL_PATH,
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics = {
        "model": "Context-Aware Random Forest",
        "random_state": 42,
        "training_samples": int(
            len(X_train)
        ),
        "validation_samples": int(
            len(X_val)
        ),
        "development_samples": int(
            len(X_dev)
        ),
        "testing_samples": int(
            len(X_test)
        ),
        "feature_count": int(
            len(feature_names)
        ),
        "features": feature_names,
        "decision_threshold": float(
            best_threshold
        ),
        "validation_accuracy": float(
            validation_accuracy
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
    print("MODEL SAVED")
    print("=" * 75)

    print(
        f"Model  : {MODEL_PATH}"
    )

    print(
        f"Metrics: {METRICS_PATH}"
    )


if __name__ == "__main__":
    main()