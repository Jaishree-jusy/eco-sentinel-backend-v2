"""
ECO-SENTINEL Feature Set Comparison

Compares three feature sets using the same LightGBM model
and the same train/validation/test split.

The final test set is kept untouched until evaluation.
"""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split


DATASET_PATH = r".\data\supervised\training_samples.csv"


# ============================================================
# FEATURE SETS
# ============================================================

FEATURE_SET_A = [
    "delta_B1",
    "delta_B2",
    "delta_B3",
    "delta_B4",
    "delta_B5",
    "delta_B6",
    "delta_B7",
    "delta_B8",
    "delta_B8A",
    "delta_B9",
    "delta_B11",
    "delta_B12",
    "delta_ndvi",
    "delta_nbr",
    "delta_ndwi",
    "delta_ndmi",
]


FEATURE_SET_B = [
    "delta_B4",
    "delta_B8",
    "delta_ndvi",
    "delta_nbr",
    "delta_ndwi",
    "delta_ndmi",
]


FEATURE_SET_C = [
    "delta_B1",
    "delta_B4",
    "delta_B5",
    "delta_B8",
    "delta_B9",
    "delta_B11",
    "delta_B12",
    "delta_ndvi",
    "delta_nbr",
    "delta_ndwi",
    "delta_ndmi",
]


FEATURE_SETS = {
    "A - All 16 features": FEATURE_SET_A,
    "B - B4/B8 + indices": FEATURE_SET_B,
    "C - Selected spectral + indices": FEATURE_SET_C,
}


def build_model():
    return lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def main():

    print("=" * 70)
    print("ECO-SENTINEL FEATURE SET COMPARISON")
    print("=" * 70)

    df = pd.read_csv(DATASET_PATH)

    y = df["label"].astype(int)

    # --------------------------------------------------------
    # Create ONE common split.
    # Every feature set gets exactly the same samples.
    # --------------------------------------------------------

    indices = df.index

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(2 / 3),
        random_state=42,
        stratify=y.loc[temp_idx],
    )

    print(f"Training samples:   {len(train_idx):,}")
    print(f"Validation samples: {len(val_idx):,}")
    print(f"Testing samples:    {len(test_idx):,}")

    results = []

    # --------------------------------------------------------
    # Test each feature set
    # --------------------------------------------------------

    for name, features in FEATURE_SETS.items():

        print("\n" + "=" * 70)
        print(name)
        print("=" * 70)

        print("Features:")
        print(", ".join(features))

        X = df[features]

        X_train = X.loc[train_idx]
        X_val = X.loc[val_idx]
        X_test = X.loc[test_idx]

        y_train = y.loc[train_idx]
        y_val = y.loc[val_idx]
        y_test = y.loc[test_idx]

        model = build_model()

        print("\nTraining...")

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=75,
                    verbose=False,
                )
            ],
        )

        # ----------------------------------------------------
        # Default threshold
        # ----------------------------------------------------

        test_probabilities = model.predict_proba(
            X_test
        )[:, 1]

        predictions = (
            test_probabilities >= 0.50
        ).astype(int)

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

        print(
            f"\nAccuracy : {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall   : {recall:.4f}"
        )

        print(
            f"F1 Score : {f1:.4f}"
        )

        results.append(
            {
                "feature_set": name,
                "feature_count": len(features),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }
        )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "accuracy",
        ascending=False,
    )

    print("\n\n" + "=" * 70)
    print("FINAL FEATURE SET COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nBest feature set:")

    best = results_df.iloc[0]

    print(
        f"  {best['feature_set']}"
    )

    print(
        f"  Accuracy: {best['accuracy']:.4f}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()