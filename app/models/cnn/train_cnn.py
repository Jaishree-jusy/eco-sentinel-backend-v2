from pathlib import Path
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


# ============================================================
# ECO-SENTINEL
# CNN TRAINING
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CNN_DIR = PROJECT_ROOT / "data" / "cnn"

X_TRAIN = CNN_DIR / "X_train.npy"
Y_TRAIN = CNN_DIR / "y_train.npy"

X_VAL = CNN_DIR / "X_val.npy"
Y_VAL = CNN_DIR / "y_val.npy"

X_TEST = CNN_DIR / "X_test.npy"
Y_TEST = CNN_DIR / "y_test.npy"

MODEL_PATH = CNN_DIR / "ecosentinel_cnn_best.pt"
METRICS_PATH = CNN_DIR / "cnn_metrics.json"


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 64
EPOCHS = 30

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

RANDOM_SEED = 42
PATIENCE = 7

INPUT_CHANNELS = 24
PATCH_SIZE = 15


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# DATASET
# ============================================================

class NumpyPatchDataset(Dataset):

    def __init__(self, x_path, y_path):

        self.X = np.load(
            x_path,
            mmap_mode="r"
        )

        self.y = np.load(
            y_path,
            mmap_mode="r"
        )

        if len(self.X) != len(self.y):
            raise ValueError(
                f"X/y length mismatch: "
                f"{len(self.X)} vs {len(self.y)}"
            )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        x = np.asarray(
            self.X[index],
            dtype=np.float32
        )

        y = np.float32(
            self.y[index]
        )

        return (
            torch.from_numpy(x),
            torch.tensor(y)
        )


# ============================================================
# CNN ARCHITECTURE
# ============================================================

class EcoSentinelCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            # ------------------------------------------------
            # BLOCK 1
            # ------------------------------------------------

            nn.Conv2d(
                in_channels=INPUT_CHANNELS,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=2
            ),

            nn.Dropout2d(
                p=0.10
            ),

            # ------------------------------------------------
            # BLOCK 2
            # ------------------------------------------------

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=2
            ),

            nn.Dropout2d(
                p=0.15
            ),

            # ------------------------------------------------
            # BLOCK 3
            # ------------------------------------------------

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(
                p=0.30
            ),

            nn.Linear(
                64,
                1
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.global_pool(x)

        x = self.classifier(x)

        return x.squeeze(1)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    loader,
    device
):

    model.eval()

    all_probabilities = []
    all_labels = []

    total_loss = 0.0
    total_samples = 0

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():

        for X, y in loader:

            X = X.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device,
                non_blocking=True
            )

            logits = model(X)

            loss = criterion(
                logits,
                y
            )

            batch_size = X.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

            probabilities = torch.sigmoid(
                logits
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

            all_labels.extend(
                y.cpu().numpy()
            )

    probabilities = np.asarray(
        all_probabilities
    )

    labels = np.asarray(
        all_labels
    ).astype(int)

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {

        "loss": float(
            total_loss / total_samples
        ),

        "accuracy": float(
            accuracy_score(
                labels,
                predictions
            )
        ),

        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                labels,
                probabilities
            )
        ),

        "confusion_matrix":
            confusion_matrix(
                labels,
                predictions
            ).tolist()
    }

    return metrics


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    print("=" * 70)
    print("ECO-SENTINEL CNN TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

        print("\nGPU detected:")
        print(
            " ",
            torch.cuda.get_device_name(0)
        )

    else:

        device = torch.device(
            "cpu"
        )

        print("\nNo CUDA GPU detected.")
        print("Training will use CPU.")

    print(
        "PyTorch version:",
        torch.__version__
    )

    print(
        "Device:",
        device
    )

    # --------------------------------------------------------
    # VERIFY DATASET FILES
    # --------------------------------------------------------

    required_files = [
        X_TRAIN,
        Y_TRAIN,
        X_VAL,
        Y_VAL,
        X_TEST,
        Y_TEST,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing dataset file:\n{path}"
            )

    # --------------------------------------------------------
    # LOAD DATASETS
    # --------------------------------------------------------

    print("\nLoading datasets...")

    train_dataset = NumpyPatchDataset(
        X_TRAIN,
        Y_TRAIN
    )

    val_dataset = NumpyPatchDataset(
        X_VAL,
        Y_VAL
    )

    test_dataset = NumpyPatchDataset(
        X_TEST,
        Y_TEST
    )

    print(
        "Training samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(val_dataset)
    )

    print(
        "Test samples:",
        len(test_dataset)
    )

    # --------------------------------------------------------
    # VERIFY SAMPLE
    # --------------------------------------------------------

    sample_x, sample_y = train_dataset[0]

    print("\nSample:")
    print(
        "  Input shape:",
        tuple(sample_x.shape)
    )

    print(
        "  Label:",
        float(sample_y)
    )

    print(
        "  Input min:",
        float(sample_x.min())
    )

    print(
        "  Input max:",
        float(sample_x.max())
    )

    if not torch.isfinite(
        sample_x
    ).all():

        raise ValueError(
            "Training sample contains NaN or Inf."
        )

    # --------------------------------------------------------
    # DATA LOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    model = EcoSentinelCNN().to(
        device
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "\nTrainable parameters:",
        parameter_count
    )

    # --------------------------------------------------------
    # LOSS / OPTIMIZER
    # --------------------------------------------------------

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    best_val_f1 = -1.0
    best_epoch = 0

    patience_counter = 0

    history = []

    start_time = time.time()

    print("\nStarting training...")
    print("-" * 70)

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0
        total_samples = 0
        correct = 0

        for X, y in train_loader:

            X = X.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(X)

            loss = criterion(
                logits,
                y
            )

            loss.backward()

            optimizer.step()

            batch_size = X.size(0)

            running_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

            predictions = (
                torch.sigmoid(logits)
                >= 0.5
            ).float()

            correct += int(
                (predictions == y)
                .sum()
                .item()
            )

        train_loss = (
            running_loss
            / total_samples
        )

        train_accuracy = (
            correct
            / total_samples
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_metrics = evaluate_model(
            model,
            val_loader,
            device
        )

        val_f1 = val_metrics["f1"]

        scheduler.step(
            val_f1
        )

        current_lr = optimizer.param_groups[0]["lr"]

        epoch_record = {

            "epoch": epoch,

            "train_loss":
                train_loss,

            "train_accuracy":
                train_accuracy,

            "val_loss":
                val_metrics["loss"],

            "val_accuracy":
                val_metrics["accuracy"],

            "val_precision":
                val_metrics["precision"],

            "val_recall":
                val_metrics["recall"],

            "val_f1":
                val_metrics["f1"],

            "val_roc_auc":
                val_metrics["roc_auc"],

            "learning_rate":
                current_lr,
        }

        history.append(
            epoch_record
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val AUC: {val_metrics['roc_auc']:.4f}"
        )

        # ----------------------------------------------------
        # SAVE BEST MODEL
        #
        # Selection uses validation F1 only.
        # Test data is not used here.
        # ----------------------------------------------------

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            best_epoch = epoch

            patience_counter = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "epoch":
                        epoch,

                    "best_val_f1":
                        best_val_f1,

                    "input_channels":
                        INPUT_CHANNELS,

                    "patch_size":
                        PATCH_SIZE,
                },
                MODEL_PATH
            )

            print(
                "  -> Best model saved."
            )

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

            print(
                "\nEarly stopping triggered."
            )

            break

    elapsed = (
        time.time()
        - start_time
    )

    print("-" * 70)

    print(
        f"Training time: "
        f"{elapsed / 60:.2f} minutes"
    )

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        "Best validation F1:",
        f"{best_val_f1:.4f}"
    )

    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------

    print(
        "\nLoading best validation model..."
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    print(
        "\nFinal validation evaluation..."
    )

    final_val_metrics = evaluate_model(
        model,
        val_loader,
        device
    )

    # --------------------------------------------------------
    # FINAL TEST
    #
    # Test set is evaluated only after model selection.
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL TEST EVALUATION"
    )

    print(
        "=" * 70
    )

    test_metrics = evaluate_model(
        model,
        test_loader,
        device
    )

    print(
        f"\nAccuracy : "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"F1       : "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print(
        "\nConfusion matrix:"
    )

    print(
        np.asarray(
            test_metrics[
                "confusion_matrix"
            ]
        )
    )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    results = {

        "model":
            "EcoSentinelCNN",

        "input_channels":
            INPUT_CHANNELS,

        "patch_size":
            PATCH_SIZE,

        "batch_size":
            BATCH_SIZE,

        "epochs_requested":
            EPOCHS,

        "best_epoch":
            best_epoch,

        "best_validation_f1":
            best_val_f1,

        "training_time_minutes":
            elapsed / 60,

        "train_samples":
            len(train_dataset),

        "validation_samples":
            len(val_dataset),

        "test_samples":
            len(test_dataset),

        "final_validation_metrics":
            final_val_metrics,

        "final_test_metrics":
            test_metrics,

        "history":
            history,
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print(
        "\nMetrics saved to:"
    )

    print(
        METRICS_PATH
    )

    print(
        "\nBest CNN model saved to:"
    )

    print(
        MODEL_PATH
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "CNN TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()