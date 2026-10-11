from pathlib import Path
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights


# ============================================================
# ECO-SENTINEL — PRETRAINED RESNET-50
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data" / "cnn"

X_TRAIN = DATA_DIR / "X_train.npy"
Y_TRAIN = DATA_DIR / "y_train.npy"

X_VAL = DATA_DIR / "X_val.npy"
Y_VAL = DATA_DIR / "y_val.npy"

X_TEST = DATA_DIR / "X_test.npy"
Y_TEST = DATA_DIR / "y_test.npy"

FEATURE_TRAIN = DATA_DIR / "resnet50_train_features.npy"
FEATURE_VAL = DATA_DIR / "resnet50_val_features.npy"
FEATURE_TEST = DATA_DIR / "resnet50_test_features.npy"

MODEL_OUT = DATA_DIR / "ecosentinel_resnet50_best.pt"
METRICS_OUT = DATA_DIR / "resnet50_metrics.json"


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CHANNELS = 24

BATCH_SIZE = 64
FEATURE_BATCH_SIZE = 32

HEAD_EPOCHS = 15

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DATASET
# ============================================================

class NPYDataset(Dataset):

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
                f"X/y mismatch: "
                f"{len(self.X)} vs {len(self.y)}"
            )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        x = np.array(
            self.X[index],
            dtype=np.float32,
            copy=True
        )

        y = np.float32(
            self.y[index]
        )

        return (
            torch.from_numpy(x),
            torch.tensor(y)
        )


# ============================================================
# BUILD RESNET-50
# ============================================================

def build_resnet50():

    print()
    print("Loading ImageNet-pretrained ResNet-50...")

    model = resnet50(
        weights=ResNet50_Weights.DEFAULT
    )

    old_conv = model.conv1

    print("Adapting first convolution:")
    print("  Original input channels: 3")
    print("  New input channels:     24")

    new_conv = nn.Conv2d(
        in_channels=INPUT_CHANNELS,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )

    # Initialize the 24-channel layer from pretrained
    # RGB filters.

    with torch.no_grad():

        rgb_weights = old_conv.weight.data

        mean_rgb = rgb_weights.mean(
            dim=1,
            keepdim=True
        )

        repeated_weights = mean_rgb.repeat(
            1,
            INPUT_CHANNELS,
            1,
            1
        )

        repeated_weights *= (
            3.0 / INPUT_CHANNELS
        )

        # IMPORTANT:
        # PyTorch uses copy_(), not copy().
        new_conv.weight.data.copy_(
            repeated_weights
        )

    model.conv1 = new_conv

    # Remove ImageNet classifier.
    model.fc = nn.Identity()

    # Freeze the complete ResNet backbone.
    for parameter in model.parameters():
        parameter.requires_grad = False

    model = model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# FEATURE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_features(
    model,
    dataset,
    output_path,
    split_name
):

    print()
    print("=" * 70)
    print(
        f"EXTRACTING {split_name.upper()} FEATURES"
    )
    print("=" * 70)

    loader = DataLoader(
        dataset,
        batch_size=FEATURE_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    all_features = []
    all_labels = []

    start_time = time.time()

    total_batches = len(loader)

    for batch_index, (X, y) in enumerate(
        loader,
        start=1
    ):

        X = X.to(
            DEVICE,
            non_blocking=True
        )

        features = model(X)

        all_features.append(
            features.cpu().numpy()
        )

        all_labels.append(
            y.numpy()
        )

        if (
            batch_index == 1
            or batch_index % 25 == 0
            or batch_index == total_batches
        ):

            elapsed = (
                time.time() - start_time
            )

            print(
                f"Batch "
                f"{batch_index}/{total_batches} | "
                f"Elapsed "
                f"{elapsed / 60:.1f} min"
            )

    features = np.concatenate(
        all_features,
        axis=0
    ).astype(np.float32)

    labels = np.concatenate(
        all_labels,
        axis=0
    ).astype(np.float32)

    np.save(
        output_path,
        features
    )

    print()
    print(
        f"{split_name} features shape: "
        f"{features.shape}"
    )

    print(
        f"Saved: {output_path}"
    )

    return features, labels


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.5
):

    y_true = np.asarray(
        y_true
    ).astype(np.int64)

    probabilities = np.asarray(
        probabilities
    )

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    tp = int(
        np.sum(
            (predictions == 1)
            & (y_true == 1)
        )
    )

    tn = int(
        np.sum(
            (predictions == 0)
            & (y_true == 0)
        )
    )

    fp = int(
        np.sum(
            (predictions == 1)
            & (y_true == 0)
        )
    )

    fn = int(
        np.sum(
            (predictions == 0)
            & (y_true == 1)
        )
    )

    accuracy = (
        tp + tn
    ) / max(
        len(y_true),
        1
    )

    precision = (
        tp / max(tp + fp, 1)
    )

    recall = (
        tp / max(tp + fn, 1)
    )

    f1 = (
        2 * precision * recall
        / max(
            precision + recall,
            1e-12
        )
    )

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    order = np.argsort(
        -probabilities
    )

    sorted_true = y_true[order]

    positives = np.sum(
        y_true == 1
    )

    negatives = np.sum(
        y_true == 0
    )

    if (
        positives > 0
        and negatives > 0
    ):

        cumulative_positive = np.cumsum(
            sorted_true == 1
        )

        cumulative_negative = np.cumsum(
            sorted_true == 0
        )

        tpr = (
            cumulative_positive
            / positives
        )

        fpr = (
            cumulative_negative
            / negatives
        )

        # Compatible with older NumPy versions.
        auc = np.trapz(
            tpr,
            fpr
        )

    else:

        auc = float("nan")

    return {

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "roc_auc":
            float(auc),

        "confusion_matrix":
            [
                [tn, fp],
                [fn, tp]
            ]
    }


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def find_best_threshold(
    y_true,
    probabilities
):

    best_threshold = 0.50
    best_accuracy = -1.0

    for threshold in np.arange(
        0.30,
        0.71,
        0.01
    ):

        metrics = calculate_metrics(
            y_true,
            probabilities,
            threshold=float(threshold)
        )

        if (
            metrics["accuracy"]
            > best_accuracy
        ):

            best_accuracy = (
                metrics["accuracy"]
            )

            best_threshold = float(
                threshold
            )

    return best_threshold


# ============================================================
# CLASSIFICATION HEAD
# ============================================================

class ClassificationHead(nn.Module):

    def __init__(
        self,
        input_features
    ):

        super().__init__()

        self.classifier = nn.Sequential(

            nn.Linear(
                input_features,
                256
            ),

            nn.ReLU(),

            nn.Dropout(
                0.30
            ),

            nn.Linear(
                256,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.20
            ),

            nn.Linear(
                64,
                1
            )
        )

    def forward(self, x):

        return self.classifier(x)


# ============================================================
# TRAIN CLASSIFICATION HEAD
# ============================================================

def train_head(
    train_features,
    train_labels,
    val_features,
    val_labels
):

    print()
    print("=" * 70)
    print(
        "TRAINING RESNET-50 CLASSIFICATION HEAD"
    )
    print("=" * 70)

    train_X = torch.tensor(
        train_features,
        dtype=torch.float32
    )

    train_y = torch.tensor(
        train_labels,
        dtype=torch.float32
    ).view(
        -1,
        1
    )

    val_X = torch.tensor(
        val_features,
        dtype=torch.float32
    )

    train_dataset = torch.utils.data.TensorDataset(
        train_X,
        train_y
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    input_features = (
        train_features.shape[1]
    )

    head = ClassificationHead(
        input_features
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_val_f1 = -1.0
    best_state = None
    best_epoch = 0

    patience_counter = 0

    start_time = time.time()

    for epoch in range(
        1,
        HEAD_EPOCHS + 1
    ):

        head.train()

        total_loss = 0.0
        total_samples = 0

        for X, y in train_loader:

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = head(X)

            loss = criterion(
                logits,
                y
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * X.size(0)
            )

            total_samples += (
                X.size(0)
            )

        train_loss = (
            total_loss
            / max(
                total_samples,
                1
            )
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        head.eval()

        with torch.no_grad():

            val_logits = head(
                val_X.to(DEVICE)
            )

            val_probabilities = (
                torch.sigmoid(
                    val_logits
                )
                .cpu()
                .numpy()
                .reshape(-1)
            )

        val_metrics = calculate_metrics(
            val_labels,
            val_probabilities,
            threshold=0.5
        )

        print(
            f"Epoch "
            f"{epoch:02d}/{HEAD_EPOCHS} | "
            f"Train Loss "
            f"{train_loss:.4f} | "
            f"Val Acc "
            f"{val_metrics['accuracy']:.4f} | "
            f"Val F1 "
            f"{val_metrics['f1']:.4f} | "
            f"Val AUC "
            f"{val_metrics['roc_auc']:.4f}"
        )

        if (
            val_metrics["f1"]
            > best_val_f1
        ):

            best_val_f1 = (
                val_metrics["f1"]
            )

            best_epoch = epoch

            best_state = {
                key:
                    value.detach()
                    .cpu()
                    .clone()

                for key, value
                in head.state_dict().items()
            }

            patience_counter = 0

            print(
                "  ✓ New best validation model"
            )

        else:

            patience_counter += 1

            print(
                f"  No improvement "
                f"({patience_counter}/"
                f"{EARLY_STOPPING_PATIENCE})"
            )

            if (
                patience_counter
                >= EARLY_STOPPING_PATIENCE
            ):

                print(
                    "  Early stopping."
                )

                break

    elapsed = (
        time.time()
        - start_time
    )

    head.load_state_dict(
        best_state
    )

    head = head.to(DEVICE)

    print()
    print(
        f"Best head epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best validation F1: "
        f"{best_val_f1:.4f}"
    )

    print(
        f"Head training time: "
        f"{elapsed / 60:.2f} minutes"
    )

    return (
        head,
        best_epoch,
        best_val_f1
    )


# ============================================================
# PREDICTION
# ============================================================

@torch.no_grad()
def predict_head(
    head,
    features
):

    head.eval()

    X = torch.tensor(
        features,
        dtype=torch.float32
    )

    loader = DataLoader(
        X,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    probabilities = []

    for batch in loader:

        batch = batch.to(
            DEVICE
        )

        logits = head(batch)

        probs = torch.sigmoid(
            logits
        )

        probabilities.append(
            probs.cpu()
            .numpy()
            .reshape(-1)
        )

    return np.concatenate(
        probabilities
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "ECO-SENTINEL — PRETRAINED RESNET-50"
    )
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        "Strategy: pretrained ResNet-50 "
        "+ frozen backbone "
        "+ trainable classification head"
    )

    print()
    print(
        "Loading datasets..."
    )

    train_dataset = NPYDataset(
        X_TRAIN,
        Y_TRAIN
    )

    val_dataset = NPYDataset(
        X_VAL,
        Y_VAL
    )

    test_dataset = NPYDataset(
        X_TEST,
        Y_TEST
    )

    print(
        f"Train: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation: "
        f"{len(val_dataset):,}"
    )

    print(
        f"Test: "
        f"{len(test_dataset):,}"
    )

    # ========================================================
    # BUILD BACKBONE
    # ========================================================

    model = build_resnet50()

    print()
    print(
        "ResNet-50 backbone ready."
    )

    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    if (
        FEATURE_TRAIN.exists()
        and FEATURE_VAL.exists()
        and FEATURE_TEST.exists()
    ):

        print()
        print(
            "Cached ResNet-50 features found."
        )

        print(
            "Loading cached features..."
        )

        train_features = np.load(
            FEATURE_TRAIN
        )

        val_features = np.load(
            FEATURE_VAL
        )

        test_features = np.load(
            FEATURE_TEST
        )

        train_labels = np.load(
            Y_TRAIN
        )

        val_labels = np.load(
            Y_VAL
        )

        test_labels = np.load(
            Y_TEST
        )

    else:

        print()
        print(
            "No cached ResNet-50 features found."
        )

        print(
            "Starting feature extraction..."
        )

        train_features, train_labels = (
            extract_features(
                model,
                train_dataset,
                FEATURE_TRAIN,
                "train"
            )
        )

        val_features, val_labels = (
            extract_features(
                model,
                val_dataset,
                FEATURE_VAL,
                "validation"
            )
        )

        test_features, test_labels = (
            extract_features(
                model,
                test_dataset,
                FEATURE_TEST,
                "test"
            )
        )

    print()
    print(
        f"Feature dimension: "
        f"{train_features.shape[1]}"
    )

    # ========================================================
    # TRAIN CLASSIFICATION HEAD
    # ========================================================

    (
        head,
        best_epoch,
        best_val_f1
    ) = train_head(
        train_features,
        train_labels,
        val_features,
        val_labels
    )

    # ========================================================
    # VALIDATION THRESHOLD
    # ========================================================

    print()
    print(
        "Optimizing threshold "
        "using validation data only..."
    )

    val_probabilities = predict_head(
        head,
        val_features
    )

    best_threshold = (
        find_best_threshold(
            val_labels,
            val_probabilities
        )
    )

    val_metrics = calculate_metrics(
        val_labels,
        val_probabilities,
        threshold=best_threshold
    )

    print()
    print(
        f"Best validation threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Validation accuracy: "
        f"{val_metrics['accuracy']:.4f}"
    )

    print(
        f"Validation precision: "
        f"{val_metrics['precision']:.4f}"
    )

    print(
        f"Validation recall: "
        f"{val_metrics['recall']:.4f}"
    )

    print(
        f"Validation F1: "
        f"{val_metrics['f1']:.4f}"
    )

    print(
        f"Validation ROC-AUC: "
        f"{val_metrics['roc_auc']:.4f}"
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    print()
    print("=" * 70)
    print(
        "FINAL TEST EVALUATION"
    )
    print("=" * 70)

    print(
        "The test set is evaluated only now."
    )

    test_probabilities = predict_head(
        head,
        test_features
    )

    test_metrics = calculate_metrics(
        test_labels,
        test_probabilities,
        threshold=best_threshold
    )

    print()
    print(
        f"Threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Accuracy:  "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"F1 Score:  "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC:   "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print()
    print(
        "Confusion Matrix:"
    )

    print(
        np.array(
            test_metrics[
                "confusion_matrix"
            ]
        )
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    checkpoint = {

        "architecture":
            "ResNet-50 pretrained frozen backbone",

        "input_channels":
            INPUT_CHANNELS,

        "feature_dimension":
            int(
                train_features.shape[1]
            ),

        "best_epoch":
            best_epoch,

        "best_validation_f1":
            float(
                best_val_f1
            ),

        "threshold":
            float(
                best_threshold
            ),

        "head_state_dict":
            head.state_dict()
    }

    torch.save(
        checkpoint,
        MODEL_OUT
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics_output = {

        "model":
            "ResNet-50 pretrained frozen backbone",

        "input_channels":
            INPUT_CHANNELS,

        "feature_dimension":
            int(
                train_features.shape[1]
            ),

        "device":
            str(DEVICE),

        "train_samples":
            len(train_dataset),

        "validation_samples":
            len(val_dataset),

        "test_samples":
            len(test_dataset),

        "best_epoch":
            best_epoch,

        "best_validation_f1":
            float(
                best_val_f1
            ),

        "threshold":
            float(
                best_threshold
            ),

        "validation_metrics":
            val_metrics,

        "test_metrics":
            test_metrics
    }

    with open(
        METRICS_OUT,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics_output,
            f,
            indent=2
        )

    print()
    print("=" * 70)
    print(
        "RESNET-50 EXPERIMENT COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        f"Model saved:"
    )

    print(
        MODEL_OUT
    )

    print()
    print(
        f"Metrics saved:"
    )

    print(
        METRICS_OUT
    )

    print()
    print(
        "Current RF benchmark: 88.73%"
    )


if __name__ == "__main__":
    main()