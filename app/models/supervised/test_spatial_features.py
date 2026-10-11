"""
ECO-SENTINEL
Spatial / Contextual Feature Experiment

Adds neighborhood statistics to the strongest spectral change features.

IMPORTANT:
- Keeps the final test set untouched.
- Uses the existing training_samples.csv for labels.
- Uses Sentinel-2 aligned rasters to derive spatial context.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import rasterio

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
FOREST_DIR = DATA_DIR / "raw_images" / "forest_samples"

CSV_PATH = SUP_DIR / "training_samples.csv"

SENTINEL_2023 = FOREST_DIR / "Forest_Sample_2023.tif"
SENTINEL_2025 = FOREST_DIR / "Forest_Sample_2025.tif"

MODEL_PATH = SUP_DIR / "deforestation_rf_spatial.joblib"
METRICS_PATH = SUP_DIR / "rf_spatial_metrics.json"


# ============================================================
# FEATURES
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

def safe_ratio(a, b):
    return np.divide(
        a,
        b,
        out=np.zeros_like(a, dtype=np.float32),
        where=np.abs(b) > 1e-6,
    )


def calculate_indices(array):
    """
    Sentinel-2 bands used here:

    B4  = red
    B8  = NIR
    B12 = SWIR2
    B11 = SWIR1
    B3  = green
    """

    b3 = array[2].astype(np.float32)
    b4 = array[3].astype(np.float32)
    b8 = array[7].astype(np.float32)
    b11 = array[10].astype(np.float32)
    b12 = array[11].astype(np.float32)

    ndvi = safe_ratio(b8 - b4, b8 + b4)
    nbr = safe_ratio(b8 - b12, b8 + b12)
    ndwi = safe_ratio(b3 - b8, b3 + b8)
    ndmi = safe_ratio(b8 - b11, b8 + b11)

    return ndvi, nbr, ndwi, ndmi


def local_mean(arr, radius):
    """
    Calculate local mean using a square window.
    """
    from scipy.ndimage import uniform_filter

    size = radius * 2 + 1
    return uniform_filter(
        arr.astype(np.float32),
        size=size,
        mode="nearest",
    )


def local_std(arr, radius):
    """
    Calculate local standard deviation using a square window.
    """
    from scipy.ndimage import uniform_filter

    arr = arr.astype(np.float32)

    size = radius * 2 + 1

    mean = uniform_filter(
        arr,
        size=size,
        mode="nearest",
    )

    mean_sq = uniform_filter(
        arr * arr,
        size=size,
        mode="nearest",
    )

    variance = np.maximum(mean_sq - mean * mean, 0)

    return np.sqrt(variance)


def add_spatial_features(change_arrays):
    """
    Generate neighborhood statistics for the strongest
    temporal-change features.

    Uses 3x3 and 5x5 windows.
    """

    result = {}

    for name in [
        "delta_B11",
        "delta_B12",
        "delta_ndvi",
        "delta_nbr",
        "delta_ndwi",
        "delta_ndmi",
    ]:

        arr = change_arrays[name]

        # 3x3 neighborhood
        result[f"{name}_mean3"] = local_mean(arr, 1)
        result[f"{name}_std3"] = local_std(arr, 1)

        # 5x5 neighborhood
        result[f"{name}_mean5"] = local_mean(arr, 2)
        result[f"{name}_std5"] = local_std(arr, 2)

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ECO-SENTINEL — SPATIAL / CONTEXTUAL FEATURE EXPERIMENT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load training dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(CSV_PATH)

    print(f"Dataset shape: {df.shape}")
    print(f"Labels: {df['label'].value_counts().to_dict()}")

    # --------------------------------------------------------
    # Load Sentinel rasters
    # --------------------------------------------------------

    print("\nLoading Sentinel-2 rasters...")

    with rasterio.open(SENTINEL_2023) as src:
        image_2023 = src.read().astype(np.float32)

    with rasterio.open(SENTINEL_2025) as src:
        image_2025 = src.read().astype(np.float32)

    print("2023 shape:", image_2023.shape)
    print("2025 shape:", image_2025.shape)

    if image_2023.shape != image_2025.shape:
        raise RuntimeError(
            "2023 and 2025 rasters do not have identical dimensions."
        )

    # --------------------------------------------------------
    # Calculate temporal change features
    # --------------------------------------------------------

    print("\nCalculating spectral change features...")

    delta = image_2025 - image_2023

    ndvi23, nbr23, ndwi23, ndmi23 = calculate_indices(image_2023)
    ndvi25, nbr25, ndwi25, ndmi25 = calculate_indices(image_2025)

    change_arrays = {
        "delta_B1": delta[0],
        "delta_B4": delta[3],
        "delta_B5": delta[4],
        "delta_B9": delta[9],
        "delta_B11": delta[10],
        "delta_B12": delta[11],

        "delta_ndvi": ndvi25 - ndvi23,
        "delta_nbr": nbr25 - nbr23,
        "delta_ndwi": ndwi25 - ndwi23,
        "delta_ndmi": ndmi25 - ndmi23,
    }

    # --------------------------------------------------------
    # Spatial features
    # --------------------------------------------------------

    print("\nCalculating spatial neighborhood features...")
    print("Windows: 3x3 and 5x5")

    spatial_features = add_spatial_features(change_arrays)

    print(
        f"Generated {len(spatial_features)} spatial features."
    )

    # --------------------------------------------------------
    # Locate sample coordinates
    # --------------------------------------------------------

    required_columns = ["row", "col", "label"]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"training_samples.csv is missing required columns: {missing}"
        )

    rows = df["row"].to_numpy(dtype=np.int64)
    cols = df["col"].to_numpy(dtype=np.int64)

    height, width = image_2023.shape[1:]

    if (
        rows.min() < 0
        or rows.max() >= height
        or cols.min() < 0
        or cols.max() >= width
    ):
        raise RuntimeError(
            "Sample row/column coordinates fall outside Sentinel raster."
        )

    # --------------------------------------------------------
    # Build spatial feature dataframe
    # --------------------------------------------------------

    print("\nExtracting features at training sample locations...")

    feature_data = {}

    # Existing 10 spectral/change features
    for name in BASE_FEATURES:
        feature_data[name] = change_arrays[name][rows, cols]

    # Spatial features
    for name, arr in spatial_features.items():
        feature_data[name] = arr[rows, cols]

    X = pd.DataFrame(feature_data)
    y = df["label"].astype(int).to_numpy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X = X.fillna(0)

    print("Final feature matrix:", X.shape)

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

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

    print("\nDevelopment samples:", len(X_dev))
    print("Training samples:   ", len(X_train))
    print("Validation samples: ", len(X_val))
    print("Test samples:       ", len(X_test))

    # --------------------------------------------------------
    # Train RF
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING SPATIAL RF")
    print("=" * 70)

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=40,
        min_samples_leaf=1,
        max_features=0.7,
        class_weight=None,
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining Random Forest...")

    model.fit(X_train, y_train)

    # --------------------------------------------------------
    # Validation threshold
    # --------------------------------------------------------

    val_prob = model.predict_proba(X_val)[:, 1]

    best_threshold = 0.50
    best_accuracy = 0.0

    for threshold in np.arange(0.30, 0.701, 0.01):

        pred = (
            val_prob >= threshold
        ).astype(int)

        acc = accuracy_score(
            y_val,
            pred,
        )

        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = float(threshold)

    val_pred = (
        val_prob >= best_threshold
    ).astype(int)

    print("\nValidation threshold:", best_threshold)
    print(
        "Validation accuracy: ",
        f"{accuracy_score(y_val, val_pred):.4f}",
    )
    print(
        "Validation precision:",
        f"{precision_score(y_val, val_pred):.4f}",
    )
    print(
        "Validation recall:   ",
        f"{recall_score(y_val, val_pred):.4f}",
    )
    print(
        "Validation F1:       ",
        f"{f1_score(y_val, val_pred):.4f}",
    )
    print(
        "Validation ROC-AUC:  ",
        f"{roc_auc_score(y_val, val_prob):.4f}",
    )

    # --------------------------------------------------------
    # Retrain on complete development data
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RETRAINING ON DEVELOPMENT DATA")
    print("=" * 70)

    final_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=40,
        min_samples_leaf=1,
        max_features=0.7,
        class_weight=None,
        random_state=42,
        n_jobs=-1,
    )

    final_model.fit(
        X_dev,
        y_dev,
    )

    # --------------------------------------------------------
    # Final untouched test
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL UNTOUCHED TEST")
    print("=" * 70)

    test_prob = final_model.predict_proba(X_test)[:, 1]

    test_pred = (
        test_prob >= best_threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        test_pred,
    )

    precision = precision_score(
        y_test,
        test_pred,
    )

    recall = recall_score(
        y_test,
        test_pred,
    )

    f1 = f1_score(
        y_test,
        test_pred,
    )

    auc = roc_auc_score(
        y_test,
        test_prob,
    )

    cm = confusion_matrix(
        y_test,
        test_pred,
    )

    print("\nThreshold:", best_threshold)
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOP FEATURES")
    print("=" * 70)

    importances = pd.Series(
        final_model.feature_importances_,
        index=X.columns,
    ).sort_values(
        ascending=False
    )

    print(importances.head(20).to_string())

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    joblib.dump(
        {
            "model": final_model,
            "features": list(X.columns),
            "threshold": best_threshold,
            "feature_type": "spectral_change_plus_spatial_context",
        },
        MODEL_PATH,
    )

    metrics = {
        "model": "RandomForest Spatial Context",
        "n_estimators": 500,
        "max_depth": 40,
        "min_samples_leaf": 1,
        "max_features": 0.7,
        "n_features": len(X.columns),
        "features": list(X.columns),
        "threshold": best_threshold,
        "validation_accuracy": float(best_accuracy),
        "test_accuracy": float(accuracy),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
        "test_roc_auc": float(auc),
        "confusion_matrix": cm.tolist(),
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
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    print("Model saved:", MODEL_PATH)
    print("Metrics saved:", METRICS_PATH)


if __name__ == "__main__":
    main()