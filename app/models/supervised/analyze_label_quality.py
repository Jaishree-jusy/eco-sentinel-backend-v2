from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


DATASET = Path(r".\data\supervised\training_samples.csv")

FEATURE_COLUMNS = [
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

TARGET_COLUMN = "label"
RANDOM_SEED = 42


print("=" * 70)
print("LABEL QUALITY DIAGNOSTIC")
print("=" * 70)

df = pd.read_csv(DATASET)

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

print("\nDataset:", df.shape)
print("Class counts:")
print(y.value_counts().sort_index())


# ------------------------------------------------------------
# Train/test split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_SEED,
)


# ------------------------------------------------------------
# Strong RF diagnostic model
# ------------------------------------------------------------

print("\nTraining diagnostic Random Forest...")

model = RandomForestClassifier(
    n_estimators=500,
    random_state=RANDOM_SEED,
    class_weight="balanced",
    n_jobs=-1,
    max_features=0.7,
)

model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.50).astype(int)

accuracy = accuracy_score(y_test, predictions)

print(f"Diagnostic accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")


# ------------------------------------------------------------
# Confidence analysis
# ------------------------------------------------------------

confidence = np.abs(probabilities - 0.50)

results = pd.DataFrame({
    "actual": y_test.to_numpy(),
    "predicted": predictions,
    "probability": probabilities,
    "confidence": confidence,
})

print("\n" + "=" * 70)
print("PREDICTION CONFIDENCE")
print("=" * 70)

bins = [
    (0.00, 0.05),
    (0.05, 0.10),
    (0.10, 0.20),
    (0.20, 0.30),
    (0.30, 0.40),
    (0.40, 0.50),
]

for low, high in bins:
    mask = (
        (results["confidence"] >= low)
        & (results["confidence"] < high)
    )

    count = int(mask.sum())

    if count == 0:
        continue

    correct = (
        results.loc[mask, "actual"]
        == results.loc[mask, "predicted"]
    ).mean()

    print(
        f"Confidence {low:.2f}-{high:.2f}: "
        f"{count} samples, "
        f"accuracy {correct:.4f}"
    )


# ------------------------------------------------------------
# Potentially ambiguous samples
# ------------------------------------------------------------

ambiguous = results[results["confidence"] < 0.10]

print("\n" + "=" * 70)
print("AMBIGUOUS SAMPLES")
print("=" * 70)

print(
    "Samples with model probability between "
    "0.40 and 0.60:",
    len(ambiguous),
)

if len(ambiguous) > 0:
    print(
        "Ambiguous sample error rate:",
        round(
            (
                ambiguous["actual"]
                != ambiguous["predicted"]
            ).mean(),
            4,
        ),
    )


# ------------------------------------------------------------
# Class probability statistics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PROBABILITY DISTRIBUTION BY ACTUAL CLASS")
print("=" * 70)

for label in [0, 1]:
    probs = results.loc[
        results["actual"] == label,
        "probability"
    ]

    print(f"\nClass {label}:")
    print(f"  Mean probability : {probs.mean():.4f}")
    print(f"  Median           : {probs.median():.4f}")
    print(f"  Std deviation    : {probs.std():.4f}")
    print(f"  10th percentile  : {probs.quantile(0.10):.4f}")
    print(f"  90th percentile  : {probs.quantile(0.90):.4f}")


# ------------------------------------------------------------
# Feature statistics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CLASS SEPARATION")
print("=" * 70)

for feature in FEATURE_COLUMNS:

    class0 = df.loc[df[TARGET_COLUMN] == 0, feature]
    class1 = df.loc[df[TARGET_COLUMN] == 1, feature]

    mean0 = class0.mean()
    mean1 = class1.mean()

    pooled_std = np.sqrt(
        (class0.var() + class1.var()) / 2
    )

    if pooled_std > 0:
        separation = abs(mean1 - mean0) / pooled_std
    else:
        separation = 0.0

    print(
        f"{feature:15s} "
        f"separation={separation:.4f}"
    )


print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)