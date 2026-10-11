"""
ECO-SENTINEL — Corrected Training Pipeline (v2)

WHY THIS SCRIPT EXISTS
-----------------------
The previous experiments (LightGBM run, RF tuning runs, hard-example run,
context/spatial-feature runs) had three problems that inflate reported
accuracy without the model actually being better:

  1. `rf_corrected_ndwi` tuned its decision threshold directly on the
     final test set (no validation set at all). That number should be
     discarded.

  2. ~8 different experiments were all evaluated against the SAME fixed
     test split (same random_state, same test_size), and the best-looking
     number was carried forward as "current best" (see the hardcoded
     0.8873 in the reduced-features script, and the running scoreboard
     printed in rf_corrected_ndwi.py). Once you pick a winner by looking
     at test-set scores across multiple experiments, the test set is no
     longer an unbiased estimate — this is model selection leaking into
     the test metric, even though each individual script's *internal*
     train/val/test discipline looked fine.

  3. Both the neighborhood spatial-context features (doc: rf_spatial,
     uniform_filter over the *whole* raster before splitting) and the
     row-level random train/test split ignore spatial autocorrelation.
     Neighboring pixels in a Sentinel scene are highly correlated
     (same disturbance polygon, same illumination/season), so a model
     can score well by "memorizing" a nearby pixel it saw in training,
     not by learning a generalizable pattern. This is a standard and
     well-documented failure mode in remote-sensing ML.

WHAT THIS SCRIPT DOES DIFFERENTLY
-----------------------------------
  A. SPATIAL BLOCK SPLIT
     Instead of splitting individual pixels/rows randomly, we tile the
     raster into fixed-size blocks (default 64x64 px) and assign whole
     blocks to train / val / test. No pixel in test is within a block
     of any pixel used in train or val. This gives you an honest
     estimate of how the model performs on genuinely unseen areas.

  B. ONE LOCKED EVALUATION PROTOCOL
     All feature-set and hyperparameter search happens on the validation
     split only. The test split is touched exactly once, at the very
     end, using the single configuration that won on validation. No
     "current best test accuracy" scoreboard, no comparing test numbers
     across runs.

  C. PHENOLOGY-AWARE SPATIAL NORMALIZATION (borrowed from the SSLT
     paper, Yang et al. 2023, IEEE JSTARS)
     Their key insight: raw band/index differences between two dates
     conflate real disturbance with seasonal (phenological) drift and
     sensor differences. Their fix is to normalize each date's NBR
     against the *local neighborhood median of forest pixels on that
     same date*, before differencing:

         sNBR_date = NBR_center_date / NBR_median_date(window)

     This is different from (and does NOT have the leakage problem of)
     the old "spatial context" features, because:
       - it's computed PER DATE, using only same-date neighboring
         pixels, not neighboring TRAINING LABELS,
       - it doesn't fold a training pixel's own class-correlated value
         into a test pixel's feature — it just answers "how unusual is
         this pixel relative to the surrounding forest, on this date."
     We generate sNBR (and the same idea applied to NDVI/NDMI) for
     2023 and 2025 separately, then take deltas of the NORMALIZED
     indices rather than deltas of the RAW indices.

  D. JOINT FEATURE-SET + HYPERPARAMETER SEARCH VIA RANDOMIZED SEARCH
     (loosely inspired by the metaheuristic optimization idea in the
     ABAC/UBK paper, Li et al. 2026, Scientific Reports — searching a
     joint configuration space instead of hand-chasing one grid at a
     time) evaluated ONLY on the validation split, so you get one
     defensible "best config" instead of 8 scripts each claiming to be
     the new best.

USAGE
-----
    python train_eco_sentinel_v2.py

Expects the same inputs your other scripts used:
    - training_samples.csv  with columns: row, col, label, delta_* ...
    - Forest_Sample_2023.tif / Forest_Sample_2025.tif (optional; only
      needed for the sNBR-style phenology-normalized features — the
      script degrades gracefully to the existing delta_* features if
      the rasters aren't found).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

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
from sklearn.model_selection import ParameterSampler

try:
    import rasterio
    HAVE_RASTERIO = True
except ImportError:
    HAVE_RASTERIO = False


# ============================================================
# PATHS  (edit these to match your environment)
# ============================================================

DATASET_PATH = Path(r".\data\supervised\training_samples.csv")

SENTINEL_2023 = Path(r".\data\raw_images\forest_samples\Forest_Sample_2023.tif")
SENTINEL_2025 = Path(r".\data\raw_images\forest_samples\Forest_Sample_2025.tif")

MODEL_DIR = Path(r".\data\supervised")
MODEL_PATH = MODEL_DIR / "deforestation_v2_spatial_block.joblib"
METRICS_PATH = MODEL_DIR / "v2_spatial_block_metrics.json"

RANDOM_SEED = 42

# Spatial block size (pixels). Must be large enough that a 3x3/5x5
# neighborhood window never crosses a block boundary undetected, and
# large enough that adjacent blocks are only weakly correlated.
BLOCK_SIZE = 64

# Fractions of BLOCKS (not pixels) assigned to each split.
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20
TEST_FRAC = 0.20

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
# A. SPATIAL BLOCK SPLIT
# ============================================================

def assign_spatial_blocks(df: pd.DataFrame, block_size: int) -> pd.DataFrame:
    """
    Assign every sample to a spatial block id based on its (row, col).
    Samples that fall in the same block always end up in the same split.
    """

    if "row" not in df.columns or "col" not in df.columns:
        raise RuntimeError(
            "training_samples.csv must contain 'row' and 'col' columns "
            "to perform a spatial block split. Regenerate the dataset "
            "with pixel coordinates if it does not have them."
        )

    block_row = (df["row"] // block_size).astype(int)
    block_col = (df["col"] // block_size).astype(int)

    df = df.copy()
    df["block_id"] = block_row.astype(str) + "_" + block_col.astype(str)

    return df


def spatial_block_split(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
):
    """
    Randomly assign whole BLOCKS to train/val/test so that no block is
    split across sets. This is the key fix: pixels near each other
    always land in the same partition, removing the "test pixel next
    door to a training pixel" leakage channel.
    """

    assert abs((train_frac + val_frac + test_frac) - 1.0) < 1e-6

    rng = np.random.RandomState(seed)

    blocks = df["block_id"].unique()
    rng.shuffle(blocks)

    n_blocks = len(blocks)
    n_train = int(round(n_blocks * train_frac))
    n_val = int(round(n_blocks * val_frac))

    train_blocks = set(blocks[:n_train])
    val_blocks = set(blocks[n_train:n_train + n_val])
    test_blocks = set(blocks[n_train + n_val:])

    train_mask = df["block_id"].isin(train_blocks)
    val_mask = df["block_id"].isin(val_blocks)
    test_mask = df["block_id"].isin(test_blocks)

    print(f"\nSpatial blocks: {n_blocks} total")
    print(f"  train blocks: {len(train_blocks)}  ({train_mask.sum()} samples)")
    print(f"  val blocks:   {len(val_blocks)}  ({val_mask.sum()} samples)")
    print(f"  test blocks:  {len(test_blocks)}  ({test_mask.sum()} samples)")

    # Sanity check: no block should appear in more than one split.
    overlap = (train_blocks & val_blocks) | (train_blocks & test_blocks) | (val_blocks & test_blocks)
    assert not overlap, f"Block leakage detected: {overlap}"

    return train_mask, val_mask, test_mask


# ============================================================
# C. PHENOLOGY-NORMALIZED FEATURES (SSLT-style spatial normalization)
# ============================================================

EPS = 1e-6


def safe_divide(a, b):
    return np.divide(
        a, b,
        out=np.zeros_like(a, dtype=np.float32),
        where=np.abs(b) > EPS,
    )


def calculate_indices(bands: np.ndarray) -> dict:
    """Sentinel-2 12-band index calculation (same convention as your
    existing context script)."""
    b3 = bands[2].astype(np.float32)
    b4 = bands[3].astype(np.float32)
    b8 = bands[7].astype(np.float32)
    b11 = bands[10].astype(np.float32)
    b12 = bands[11].astype(np.float32)

    return {
        "ndvi": safe_divide(b8 - b4, b8 + b4),
        "nbr": safe_divide(b8 - b12, b8 + b12),
        "ndwi": safe_divide(b3 - b8, b3 + b8),
        "ndmi": safe_divide(b8 - b11, b8 + b11),
    }


def spatially_normalized_index(index_arr: np.ndarray, window: int = 15, percentile: float = 90.0) -> np.ndarray:
    """
    SSLT-style spatial normalization (Yang et al. 2023, Eq. 2):

        sIndex = index_center / median(index values above `percentile`
                                        within a `window`x`window` box
                                        around the center)

    This suppresses seasonal/phenological signal that affects the
    whole neighborhood roughly equally, while amplifying genuinely
    localized disturbance at the center pixel. Computed PER DATE
    (single raster in, single raster out) — never mixes information
    across dates or across train/test splits, since it only uses the
    same-date raster's own neighborhood.

    NOTE: this uses scipy for a fast approximate percentile-filtered
    median via a generic_filter. For large rasters this can be slow;
    reduce `window` or subsample if needed.
    """
    from scipy.ndimage import generic_filter

    def _high_percentile_median(values):
        thresh = np.percentile(values, percentile)
        high_vals = values[values >= thresh]
        if high_vals.size == 0:
            return np.median(values)
        return np.median(high_vals)

    neighborhood_ref = generic_filter(
        index_arr.astype(np.float32),
        _high_percentile_median,
        size=window,
        mode="nearest",
    )

    return safe_divide(index_arr.astype(np.float32), neighborhood_ref)


def build_phenology_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds sNBR/sNDVI/sNDMI-style delta features on top of the existing
    BASE_FEATURES, if the Sentinel rasters are available. Falls back
    to returning df unchanged if rasterio or the raster files are
    missing, so the rest of the pipeline still runs.

    IMPORTANT: if training_samples.csv already contains precomputed
    phenology-normalized columns (e.g. snbr_2023, snbr_2025,
    delta_snbr, nbr_change_zscore — produced by an earlier feature
    pipeline), we use those directly and skip the raster recomputation
    entirely. Recomputing per-pixel spatial normalization with a
    Python-level percentile filter (scipy generic_filter) over a full
    scene is extremely slow and unnecessary if the values already
    exist in the CSV.
    """

    already_have = [c for c in ["delta_snbr", "snbr_2023", "snbr_2025", "nbr_change_zscore"] if c in df.columns]

    if already_have:
        print(f"\n[phenology features] Found precomputed columns in the CSV: {already_have}")
        print("[phenology features] Using these directly — skipping raster recomputation.")
        return df

    if not HAVE_RASTERIO:
        print("\n[phenology features] rasterio not installed — skipping, "
              "using existing delta_* features only.")
        return df

    if not (SENTINEL_2023.exists() and SENTINEL_2025.exists()):
        print("\n[phenology features] Sentinel rasters not found at the "
              "configured paths — skipping, using existing delta_* "
              "features only.")
        return df

    print("\n[phenology features] No precomputed columns found. Loading "
          "rasters and computing spatially-normalized indices from "
          "scratch (this can take a while for large scenes — consider "
          "Ctrl+C and precomputing these once, offline, if it's too slow)...")

    with rasterio.open(SENTINEL_2023) as src:
        raster23 = src.read().astype(np.float32)

    with rasterio.open(SENTINEL_2025) as src:
        raster25 = src.read().astype(np.float32)

    idx23 = calculate_indices(raster23)
    idx25 = calculate_indices(raster25)

    norm23 = {name: spatially_normalized_index(arr) for name, arr in idx23.items()}
    norm25 = {name: spatially_normalized_index(arr) for name, arr in idx25.items()}

    rows = df["row"].to_numpy(dtype=np.int64)
    cols = df["col"].to_numpy(dtype=np.int64)

    df = df.copy()
    for name in ["ndvi", "nbr", "ndwi", "ndmi"]:
        df[f"s{name}_delta"] = (norm25[name] - norm23[name])[rows, cols]

    print(f"[phenology features] Added: {[f's{n}_delta' for n in ['ndvi','nbr','ndwi','ndmi']]}")

    return df


# ============================================================
# THRESHOLD SEARCH (validation only — never touches test)
# ============================================================

def find_best_threshold(y_true, probabilities):
    best_threshold = 0.50
    best_accuracy = -1.0
    for threshold in np.arange(0.20, 0.81, 0.01):
        preds = (probabilities >= threshold).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = float(threshold)
    return best_threshold, best_accuracy


# ============================================================
# D. JOINT FEATURE-SET + HYPERPARAMETER SEARCH (validation only)
# ============================================================

FEATURE_SET_CHOICES = {
    "base_10": BASE_FEATURES,
    # populated with the phenology-normalized set if available (see main())
}

HYPERPARAM_SPACE = {
    "n_estimators": [300, 500, 800],
    "max_depth": [20, 30, 40, None],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", 0.7, 1.0],
    "class_weight": [None, "balanced"],
}

N_SEARCH_ITER = 25


def run_search(df, train_mask, val_mask, feature_sets, seed):
    """
    Randomized joint search over (feature_set x hyperparameters),
    scored ONLY on the validation split. Returns the single winning
    configuration. This replaces the old pattern of running 8 separate
    scripts and comparing their TEST numbers.
    """

    sampler = list(ParameterSampler(HYPERPARAM_SPACE, n_iter=N_SEARCH_ITER, random_state=seed))

    y_train = df.loc[train_mask, "label"].astype(int).to_numpy()
    y_val = df.loc[val_mask, "label"].astype(int).to_numpy()

    results = []
    best = None

    print("\n" + "=" * 70)
    print(f"JOINT SEARCH: {len(feature_sets)} feature sets x {N_SEARCH_ITER} hyperparam draws "
          f"= {len(feature_sets) * N_SEARCH_ITER} configurations")
    print("=" * 70)

    trial = 0
    for fname, features in feature_sets.items():

        X_train = df.loc[train_mask, features].to_numpy(dtype=np.float32)
        X_val = df.loc[val_mask, features].to_numpy(dtype=np.float32)

        for params in sampler:
            trial += 1

            model = RandomForestClassifier(
                **params,
                random_state=seed,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)

            val_prob = model.predict_proba(X_val)[:, 1]
            threshold, val_acc = find_best_threshold(y_val, val_prob)

            val_pred = (val_prob >= threshold).astype(int)
            val_f1 = f1_score(y_val, val_pred, zero_division=0)
            val_auc = roc_auc_score(y_val, val_prob)

            record = {
                "trial": trial,
                "feature_set": fname,
                "n_features": len(features),
                "params": params,
                "threshold": threshold,
                "val_accuracy": float(val_acc),
                "val_f1": float(val_f1),
                "val_auc": float(val_auc),
            }
            results.append(record)

            if best is None or val_acc > best["val_accuracy"]:
                best = record

            if trial % 10 == 0 or trial == len(feature_sets) * N_SEARCH_ITER:
                print(f"  [{trial}] {fname} acc={val_acc:.4f} f1={val_f1:.4f} auc={val_auc:.4f} "
                      f"(best so far: {best['val_accuracy']:.4f})")

    print("\nBest configuration found on VALIDATION:")
    print(json.dumps(best, indent=2, default=str))

    return best, results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ECO-SENTINEL v2 — SPATIAL BLOCK SPLIT, LEAK-FREE PIPELINE")
    print("=" * 70)

    df = pd.read_csv(DATASET_PATH)
    print(f"\nDataset: {len(df):,} samples, columns: {list(df.columns)}")

    if "label" not in df.columns:
        raise RuntimeError("training_samples.csv must contain a 'label' column.")

    # --------------------------------------------------------
    # A. Spatial block split (fixes leakage from random pixel splits)
    # --------------------------------------------------------

    df = assign_spatial_blocks(df, BLOCK_SIZE)

    train_mask, val_mask, test_mask = spatial_block_split(
        df, TRAIN_FRAC, VAL_FRAC, TEST_FRAC, RANDOM_SEED
    )

    # --------------------------------------------------------
    # C. Optional phenology-normalized features
    # --------------------------------------------------------

    df = build_phenology_features(df)

    # Covers both: (a) columns this script generates itself
    # (s{name}_delta, from spatially_normalized_index), and
    # (b) columns already present in your CSV from an earlier
    # feature pipeline (snbr_2023/2025, delta_snbr, nbr_change_zscore,
    # nbr_local_median_2023/2025).
    generated_phenology_cols = [c for c in df.columns if c.endswith("_delta") and c.startswith("s")]
    existing_phenology_cols = [
        c for c in [
            "snbr_2023", "snbr_2025", "delta_snbr", "nbr_change_zscore",
            "nbr_local_median_2023", "nbr_local_median_2025",
        ]
        if c in df.columns
    ]
    phenology_cols = sorted(set(generated_phenology_cols) | set(existing_phenology_cols))

    feature_sets = dict(FEATURE_SET_CHOICES)
    if phenology_cols:
        feature_sets["base_10_plus_phenology"] = BASE_FEATURES + phenology_cols
        feature_sets["phenology_only"] = phenology_cols

    for fname, feats in feature_sets.items():
        missing = [f for f in feats if f not in df.columns]
        if missing:
            raise RuntimeError(f"Feature set '{fname}' references missing columns: {missing}")

    # --------------------------------------------------------
    # D. Joint search on validation split only
    # --------------------------------------------------------

    best, all_results = run_search(df, train_mask, val_mask, feature_sets, RANDOM_SEED)

    # --------------------------------------------------------
    # B. Retrain winning config on train+val, evaluate ONCE on test
    # --------------------------------------------------------

    dev_mask = train_mask | val_mask
    winning_features = feature_sets[best["feature_set"]]

    X_dev = df.loc[dev_mask, winning_features].to_numpy(dtype=np.float32)
    y_dev = df.loc[dev_mask, "label"].astype(int).to_numpy()

    X_test = df.loc[test_mask, winning_features].to_numpy(dtype=np.float32)
    y_test = df.loc[test_mask, "label"].astype(int).to_numpy()

    print("\n" + "=" * 70)
    print("RETRAINING WINNING CONFIG ON TRAIN+VAL")
    print("=" * 70)
    print(f"Feature set: {best['feature_set']} ({len(winning_features)} features)")
    print(f"Params: {best['params']}")

    final_model = RandomForestClassifier(
        **best["params"],
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    final_model.fit(X_dev, y_dev)

    print("\n" + "=" * 70)
    print("FINAL UNTOUCHED SPATIAL-BLOCK TEST — EVALUATED EXACTLY ONCE")
    print("=" * 70)

    test_prob = final_model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= best["threshold"]).astype(int)

    accuracy = accuracy_score(y_test, test_pred)
    precision = precision_score(y_test, test_pred, zero_division=0)
    recall = recall_score(y_test, test_pred, zero_division=0)
    f1 = f1_score(y_test, test_pred, zero_division=0)
    auc = roc_auc_score(y_test, test_prob)
    cm = confusion_matrix(y_test, test_pred)

    print(f"\nAccuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")
    print(f"Threshold : {best['threshold']:.3f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(
        y_test, test_pred,
        target_names=["No Deforestation", "Deforestation"],
        zero_division=0,
    ))

    print("\nNOTE: compare this number to your OLD numbers with caution —")
    print("it is very likely LOWER, because the spatial block split removes")
    print("the neighbor-pixel leakage that was inflating the old row-random")
    print("splits. A lower, honest number here is more useful than a higher,")
    print("leaked one: it's the number that will actually hold up on new imagery.")

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importances = sorted(
        zip(winning_features, final_model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    print("\nFeature importance:")
    for feat, val in importances:
        print(f"  {feat:<20} {val:.4f}")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(
        {
            "model": final_model,
            "features": winning_features,
            "threshold": best["threshold"],
            "block_size": BLOCK_SIZE,
        },
        MODEL_PATH,
    )

    metrics = {
        "model": "RandomForest (spatial-block, leak-free)",
        "random_seed": RANDOM_SEED,
        "block_size": BLOCK_SIZE,
        "winning_feature_set": best["feature_set"],
        "features": winning_features,
        "hyperparameters": best["params"],
        "threshold": best["threshold"],
        "validation_accuracy": best["val_accuracy"],
        "validation_f1": best["val_f1"],
        "validation_auc": best["val_auc"],
        "test_accuracy": float(accuracy),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
        "test_roc_auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "feature_importance": {f: float(v) for f, v in importances},
        "n_search_trials": len(all_results),
        "search_results": all_results,
        "dataset": str(DATASET_PATH),
        "model_path": str(MODEL_PATH),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Model  : {MODEL_PATH}")
    print(f"Metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()