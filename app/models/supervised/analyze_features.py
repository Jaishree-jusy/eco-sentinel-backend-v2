"""
ECO-SENTINEL Feature Distribution Diagnostic

Checks whether the 16 spectral-delta features provide good
separation between:

    0 = No Deforestation
    1 = Deforestation

This is a diagnostic only.
It does NOT modify the dataset or model.
"""

from __future__ import annotations

import pandas as pd


DATASET_PATH = r".\data\supervised\training_samples.csv"


FEATURES = [
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


def main():

    print("=" * 75)
    print("ECO-SENTINEL FEATURE DISTRIBUTION DIAGNOSTIC")
    print("=" * 75)

    df = pd.read_csv(DATASET_PATH)

    print(f"\nDataset: {len(df):,} samples")

    print("\nClass distribution:")
    print(df["label"].value_counts().sort_index())

    print("\n" + "=" * 75)
    print("FEATURE STATISTICS BY CLASS")
    print("=" * 75)

    for feature in FEATURES:

        no_loss = df.loc[
            df["label"] == 0,
            feature,
        ]

        loss = df.loc[
            df["label"] == 1,
            feature,
        ]

        mean_0 = no_loss.mean()
        mean_1 = loss.mean()

        median_0 = no_loss.median()
        median_1 = loss.median()

        std_0 = no_loss.std()
        std_1 = loss.std()

        difference = mean_1 - mean_0

        print("\n" + "-" * 75)
        print(f"FEATURE: {feature}")

        print(
            f"  No Deforestation:"
            f" mean={mean_0:.6f},"
            f" median={median_0:.6f},"
            f" std={std_0:.6f}"
        )

        print(
            f"  Deforestation:"
            f"     mean={mean_1:.6f},"
            f" median={median_1:.6f},"
            f" std={std_1:.6f}"
        )

        print(
            f"  Mean difference:"
            f" {difference:+.6f}"
        )

    # ---------------------------------------------------------
    # Correlation with label
    # ---------------------------------------------------------

    print("\n" + "=" * 75)
    print("FEATURE CORRELATION WITH DEFORESTATION LABEL")
    print("=" * 75)

    correlations = (
        df[FEATURES + ["label"]]
        .corr()["label"]
        .drop("label")
        .sort_values(
            key=lambda x: x.abs(),
            ascending=False,
        )
    )

    for feature, correlation in correlations.items():

        print(
            f"{feature:<20}"
            f" {correlation:+.4f}"
        )

    # ---------------------------------------------------------
    # Class means table
    # ---------------------------------------------------------

    print("\n" + "=" * 75)
    print("CLASS MEAN COMPARISON")
    print("=" * 75)

    means = (
        df.groupby("label")[FEATURES]
        .mean()
        .T
    )

    means.columns = [
        "No_Deforestation",
        "Deforestation",
    ]

    means["Difference"] = (
        means["Deforestation"]
        - means["No_Deforestation"]
    )

    means["Abs_Difference"] = (
        means["Difference"]
        .abs()
    )

    means = means.sort_values(
        "Abs_Difference",
        ascending=False,
    )

    print(
        means.to_string(
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print("\n" + "=" * 75)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()