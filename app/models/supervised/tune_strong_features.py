from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split


DATASET = Path(r".\data\supervised\training_samples.csv")
MODEL_OUTPUT = Path(r".\data\supervised\deforestation_random_forest_strong_tuned.joblib")
RESULTS_OUTPUT = Path(r".\data\supervised\rf_strong_tuning_results.json")

RANDOM_SEED = 42

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

TARGET = "label"


print("=" * 70)
print("TUNING RANDOM FOREST - STRONG FEATURES")
print("=" * 70)

df = pd.read_csv(DATASET)

X = df[FEATURE_COLUMNS]
y = df[TARGET]

# 80% development / 20% untouched final test
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_SEED,
)

# 80% training / 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.20,
    stratify=y_dev,
    random_state=RANDOM_SEED,
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Final test samples:", len(X_test))


# ------------------------------------------------------------
# Parameter combinations
# ------------------------------------------------------------

configs = []

for n_estimators in [300, 500, 800]:
    for max_depth in [None, 20, 30, 40]:
        for min_samples_leaf in [1, 2]:
            for max_features in ["sqrt", 0.7, 1.0]:

                configs.append({
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                    "max_features": max_features,
                })


print("\nConfigurations to test:", len(configs))


# ------------------------------------------------------------
# Validation tuning
# ------------------------------------------------------------

results = []

best_accuracy = -1
best_config = None

for i, config in enumerate(configs, start=1):

    print(
        f"\n[{i}/{len(configs)}] "
        f"trees={config['n_estimators']}, "
        f"depth={config['max_depth']}, "
        f"leaf={config['min_samples_leaf']}, "
        f"features={config['max_features']}"
    )

    model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features=config["max_features"],
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    val_probabilities = model.predict_proba(X_val)[:, 1]

    predictions = (val_probabilities >= 0.50).astype(int)

    accuracy = accuracy_score(y_val, predictions)

    results.append({
        **config,
        "validation_accuracy": float(accuracy),
    })

    print(f"Validation accuracy: {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_config = config.copy()


# ------------------------------------------------------------
# Best configuration
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BEST VALIDATION CONFIGURATION")
print("=" * 70)

print(best_config)
print(f"Validation accuracy: {best_accuracy:.4f}")


# ------------------------------------------------------------
# Retrain on complete development data
# ------------------------------------------------------------

print("\nRetraining best configuration on all development data...")

final_model = RandomForestClassifier(
    n_estimators=best_config["n_estimators"],
    max_depth=best_config["max_depth"],
    min_samples_leaf=best_config["min_samples_leaf"],
    max_features=best_config["max_features"],
    class_weight="balanced",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

final_model.fit(X_dev, y_dev)


# ------------------------------------------------------------
# Threshold optimization on validation set
# ------------------------------------------------------------

# Train a temporary model with the same configuration for threshold search
threshold_model = RandomForestClassifier(
    n_estimators=best_config["n_estimators"],
    max_depth=best_config["max_depth"],
    min_samples_leaf=best_config["min_samples_leaf"],
    max_features=best_config["max_features"],
    class_weight="balanced",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

threshold_model.fit(X_train, y_train)

val_probabilities = threshold_model.predict_proba(X_val)[:, 1]

best_threshold = 0.50
best_threshold_accuracy = 0.0

for threshold in np.arange(0.20, 0.81, 0.01):

    predictions = (
        val_probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_val,
        predictions
    )

    if accuracy > best_threshold_accuracy:
        best_threshold_accuracy = accuracy
        best_threshold = float(threshold)


print("\nBest validation threshold:", best_threshold)
print(
    "Validation accuracy with threshold:",
    f"{best_threshold_accuracy:.4f}"
)


# ------------------------------------------------------------
# Final untouched test evaluation
# ------------------------------------------------------------

print("\nEvaluating on untouched final test set...")

test_probabilities = final_model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
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

cm = confusion_matrix(
    y_test,
    test_predictions
)


# ------------------------------------------------------------
# Final results
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL TUNED STRONG-FEATURE RF")
print("=" * 70)

print(
    f"Accuracy : {test_accuracy:.4f} "
    f"({test_accuracy * 100:.2f}%)"
)

print(f"Precision: {test_precision:.4f}")
print(f"Recall   : {test_recall:.4f}")
print(f"F1 Score : {test_f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nBest configuration:")
print(best_config)

print("\nBest threshold:", best_threshold)


# ------------------------------------------------------------
# Save model
# ------------------------------------------------------------

import joblib

joblib.dump(
    {
        "model": final_model,
        "feature_columns": FEATURE_COLUMNS,
        "threshold": best_threshold,
    },
    MODEL_OUTPUT,
)


# ------------------------------------------------------------
# Save tuning results
# ------------------------------------------------------------

output = {
    "model": "Random Forest",
    "feature_set": FEATURE_COLUMNS,
    "best_configuration": best_config,
    "best_validation_accuracy": float(best_accuracy),
    "best_validation_threshold": float(best_threshold),
    "validation_accuracy_at_threshold": float(
        best_threshold_accuracy
    ),
    "final_test_accuracy": float(test_accuracy),
    "final_test_precision": float(test_precision),
    "final_test_recall": float(test_recall),
    "final_test_f1": float(test_f1),
    "confusion_matrix": cm.tolist(),
    "all_configurations": results,
}

with open(
    RESULTS_OUTPUT,
    "w",
    encoding="utf-8"
) as f:
    json.dump(output, f, indent=2)


print("\nSaved model:")
print(MODEL_OUTPUT)

print("\nSaved results:")
print(RESULTS_OUTPUT)

print("\n" + "=" * 70)
print("TUNING COMPLETE")
print("=" * 70)