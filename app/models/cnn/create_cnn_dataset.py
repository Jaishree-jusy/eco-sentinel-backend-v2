from pathlib import Path

import numpy as np


# ============================================================
# ECO-SENTINEL
# CNN DATASET CREATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CNN_DIR = PROJECT_ROOT / "data" / "cnn"

IMAGE_2023 = CNN_DIR / "sentinel_2023_spectral.npy"
IMAGE_2025 = CNN_DIR / "sentinel_2025_spectral.npy"
LABELS_FILE = CNN_DIR / "hansen_labels.npy"


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

PATCH_SIZE = 15
HALF = PATCH_SIZE // 2

PATCHES_PER_CLASS = 15000

BLOCK_SIZE = 150

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


def main():

    print("=" * 70)
    print("ECO-SENTINEL CNN DATASET CREATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check input files
    # --------------------------------------------------------

    for path in [IMAGE_2023, IMAGE_2025, LABELS_FILE]:

        if not path.exists():
            raise FileNotFoundError(
                f"Required CNN source file not found:\n{path}"
            )

    print("\nLoading CNN source arrays...")

    image23 = np.load(
        IMAGE_2023,
        mmap_mode="r"
    )

    image25 = np.load(
        IMAGE_2025,
        mmap_mode="r"
    )

    labels = np.load(
        LABELS_FILE,
        mmap_mode="r"
    )

    print("2023 shape:", image23.shape)
    print("2025 shape:", image25.shape)
    print("Labels shape:", labels.shape)

    # --------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------

    if image23.ndim != 3:
        raise ValueError(
            "2023 image must have shape (channels, height, width)."
        )

    if image25.ndim != 3:
        raise ValueError(
            "2025 image must have shape (channels, height, width)."
        )

    if image23.shape != image25.shape:
        raise ValueError(
            "2023 and 2025 arrays do not have identical shapes."
        )

    height = image23.shape[1]
    width = image23.shape[2]

    if labels.shape != (height, width):
        raise ValueError(
            "Labels do not match Sentinel spatial dimensions."
        )

    if image23.shape[0] != 12 or image25.shape[0] != 12:
        raise ValueError(
            "Expected exactly 12 spectral bands for each year."
        )

    print("\nSpatial dimensions:")
    print("  Height:", height)
    print("  Width :", width)

    print("\nCNN channels:")
    print("  2023:", image23.shape[0])
    print("  2025:", image25.shape[0])
    print("  Total:", image23.shape[0] + image25.shape[0])

    # --------------------------------------------------------
    # Find candidate pixels
    # --------------------------------------------------------

    print("\nFinding candidate patch centres...")

    positive_positions = np.argwhere(labels == 1)
    negative_positions = np.argwhere(labels == 0)

    print(
        "Positive candidates:",
        len(positive_positions)
    )

    print(
        "Negative candidates:",
        len(negative_positions)
    )

    # --------------------------------------------------------
    # Remove edge pixels
    # --------------------------------------------------------

    valid_positive = (
        (positive_positions[:, 0] >= HALF)
        & (positive_positions[:, 0] < height - HALF)
        & (positive_positions[:, 1] >= HALF)
        & (positive_positions[:, 1] < width - HALF)
    )

    valid_negative = (
        (negative_positions[:, 0] >= HALF)
        & (negative_positions[:, 0] < height - HALF)
        & (negative_positions[:, 1] >= HALF)
        & (negative_positions[:, 1] < width - HALF)
    )

    positive_positions = positive_positions[valid_positive]
    negative_positions = negative_positions[valid_negative]

    print("\nValid patch centres:")
    print("  Positive:", len(positive_positions))
    print("  Negative:", len(negative_positions))

    # --------------------------------------------------------
    # Create spatial blocks
    # --------------------------------------------------------

    num_block_rows = (
        height + BLOCK_SIZE - 1
    ) // BLOCK_SIZE

    num_block_cols = (
        width + BLOCK_SIZE - 1
    ) // BLOCK_SIZE

    total_blocks = (
        num_block_rows * num_block_cols
    )

    print("\nSpatial blocks:")
    print("  Block size :", f"{BLOCK_SIZE} x {BLOCK_SIZE}")
    print("  Block rows :", num_block_rows)
    print("  Block cols :", num_block_cols)
    print("  Total      :", total_blocks)

    # --------------------------------------------------------
    # Shuffle spatial blocks
    # --------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    block_ids = np.arange(
        total_blocks
    )

    rng.shuffle(block_ids)

    n_train_blocks = int(
        total_blocks * TRAIN_RATIO
    )

    n_val_blocks = int(
        total_blocks * VAL_RATIO
    )

    train_blocks = set(
        block_ids[:n_train_blocks].tolist()
    )

    val_blocks = set(
        block_ids[
            n_train_blocks:
            n_train_blocks + n_val_blocks
        ].tolist()
    )

    test_blocks = set(
        block_ids[
            n_train_blocks + n_val_blocks:
        ].tolist()
    )

    print("\nBlock split:")
    print("  Train blocks:", len(train_blocks))
    print("  Val blocks  :", len(val_blocks))
    print("  Test blocks :", len(test_blocks))

    # --------------------------------------------------------
    # Assign positions to splits
    # --------------------------------------------------------

    def split_positions(positions):

        train = []
        val = []
        test = []

        for row, col in positions:

            block_id = (
                (row // BLOCK_SIZE)
                * num_block_cols
                + (col // BLOCK_SIZE)
            )

            if block_id in train_blocks:

                train.append(
                    (row, col)
                )

            elif block_id in val_blocks:

                val.append(
                    (row, col)
                )

            elif block_id in test_blocks:

                test.append(
                    (row, col)
                )

        return (
            np.asarray(
                train,
                dtype=np.int32
            ),
            np.asarray(
                val,
                dtype=np.int32
            ),
            np.asarray(
                test,
                dtype=np.int32
            ),
        )

    pos_train, pos_val, pos_test = split_positions(
        positive_positions
    )

    neg_train, neg_val, neg_test = split_positions(
        negative_positions
    )

    print("\nAvailable patches by split:")

    print(
        f"  Train: positive={len(pos_train)}, "
        f"negative={len(neg_train)}"
    )

    print(
        f"  Val  : positive={len(pos_val)}, "
        f"negative={len(neg_val)}"
    )

    print(
        f"  Test : positive={len(pos_test)}, "
        f"negative={len(neg_test)}"
    )

    # --------------------------------------------------------
    # Balanced sampling
    # --------------------------------------------------------

    def balanced_sample(
        pos,
        neg,
        amount,
        rng
    ):

        if amount <= 0:
            raise ValueError(
                "Balanced sample size must be greater than zero."
            )

        if len(pos) < amount:
            raise ValueError(
                f"Not enough positive samples: "
                f"{len(pos)} available, {amount} required."
            )

        if len(neg) < amount:
            raise ValueError(
                f"Not enough negative samples: "
                f"{len(neg)} available, {amount} required."
            )

        pos_idx = rng.choice(
            len(pos),
            size=amount,
            replace=False
        )

        neg_idx = rng.choice(
            len(neg),
            size=amount,
            replace=False
        )

        pos_selected = pos[pos_idx]
        neg_selected = neg[neg_idx]

        positions = np.concatenate(
            [
                pos_selected,
                neg_selected
            ],
            axis=0
        )

        y = np.concatenate(
            [
                np.ones(
                    amount,
                    dtype=np.uint8
                ),
                np.zeros(
                    amount,
                    dtype=np.uint8
                ),
            ]
        )

        order = rng.permutation(
            len(positions)
        )

        return (
            positions[order],
            y[order]
        )

    # --------------------------------------------------------
    # Automatically determine available balanced sizes
    # --------------------------------------------------------

    train_amount = min(
        PATCHES_PER_CLASS,
        len(pos_train),
        len(neg_train)
    )

    val_amount = min(
        PATCHES_PER_CLASS,
        len(pos_val),
        len(neg_val)
    )

    test_amount = min(
        PATCHES_PER_CLASS,
        len(pos_test),
        len(neg_test)
    )

    print("\nBalanced sample sizes:")
    print(
        "  Train per class:",
        train_amount
    )

    print(
        "  Val per class  :",
        val_amount
    )

    print(
        "  Test per class :",
        test_amount
    )

    train_positions, train_y = balanced_sample(
        pos_train,
        neg_train,
        train_amount,
        rng
    )

    val_positions, val_y = balanced_sample(
        pos_val,
        neg_val,
        val_amount,
        rng
    )

    test_positions, test_y = balanced_sample(
        pos_test,
        neg_test,
        test_amount,
        rng
    )

    print("\nBalanced dataset:")

    print(
        "  Train:",
        len(train_y)
    )

    print(
        "  Val  :",
        len(val_y)
    )

    print(
        "  Test :",
        len(test_y)
    )

    print("\nClass distribution:")

    print(
        "  Train:",
        "class 0 =",
        int((train_y == 0).sum()),
        "class 1 =",
        int((train_y == 1).sum())
    )

    print(
        "  Val  :",
        "class 0 =",
        int((val_y == 0).sum()),
        "class 1 =",
        int((val_y == 1).sum())
    )

    print(
        "  Test :",
        "class 0 =",
        int((test_y == 0).sum()),
        "class 1 =",
        int((test_y == 1).sum())
    )

    # --------------------------------------------------------
    # Extract patches
    # --------------------------------------------------------

    def extract_patches(positions):

        n = len(positions)

        patches = np.empty(
            (
                n,
                24,
                PATCH_SIZE,
                PATCH_SIZE,
            ),
            dtype=np.float32
        )

        for i, (row, col) in enumerate(positions):

            r0 = row - HALF
            r1 = row + HALF + 1

            c0 = col - HALF
            c1 = col + HALF + 1

            patch23 = image23[
                :,
                r0:r1,
                c0:c1
            ]

            patch25 = image25[
                :,
                r0:r1,
                c0:c1
            ]

            if patch23.shape != (
                12,
                PATCH_SIZE,
                PATCH_SIZE
            ):
                raise ValueError(
                    f"Invalid 2023 patch shape: "
                    f"{patch23.shape}"
                )

            if patch25.shape != (
                12,
                PATCH_SIZE,
                PATCH_SIZE
            ):
                raise ValueError(
                    f"Invalid 2025 patch shape: "
                    f"{patch25.shape}"
                )

            patches[i] = np.concatenate(
                [
                    patch23,
                    patch25
                ],
                axis=0
            )

        return patches

    print("\nExtracting training patches...")

    X_train = extract_patches(
        train_positions
    )

    print("Extracting validation patches...")

    X_val = extract_patches(
        val_positions
    )

    print("Extracting test patches...")

    X_test = extract_patches(
        test_positions
    )

    print("\nPatch shapes:")

    print(
        "  X_train:",
        X_train.shape
    )

    print(
        "  X_val  :",
        X_val.shape
    )

    print(
        "  X_test :",
        X_test.shape
    )

    # --------------------------------------------------------
    # Check NaN / Inf BEFORE normalization
    # --------------------------------------------------------

    print("\nChecking input data before normalization...")

    for name, array in [
        ("X_train", X_train),
        ("X_val", X_val),
        ("X_test", X_test),
    ]:

        nan_count = int(
            np.isnan(array).sum()
        )

        inf_count = int(
            np.isinf(array).sum()
        )

        print(
            f"  {name}: NaN={nan_count}, "
            f"Inf={inf_count}"
        )

    # --------------------------------------------------------
    # Calculate normalization statistics
    #
    # IMPORTANT:
    # Only TRAINING data is used.
    #
    # nanpercentile ignores invalid NaN values.
    # --------------------------------------------------------

    print(
        "\nCalculating normalization statistics "
        "from TRAINING data only..."
    )

    channel_q01 = np.nanpercentile(
        X_train,
        1,
        axis=(0, 2, 3)
    ).astype(np.float32)

    channel_q99 = np.nanpercentile(
        X_train,
        99,
        axis=(0, 2, 3)
    ).astype(np.float32)

    # --------------------------------------------------------
    # Handle channels where all training values are NaN
    # --------------------------------------------------------

    invalid_stats = (
        ~np.isfinite(channel_q01)
        | ~np.isfinite(channel_q99)
    )

    if np.any(invalid_stats):

        bad_channels = np.where(
            invalid_stats
        )[0].tolist()

        raise ValueError(
            "Normalization statistics could not be "
            f"calculated for channels: {bad_channels}"
        )

    channel_q99 = np.maximum(
        channel_q99,
        channel_q01 + 1e-6
    )

    # --------------------------------------------------------
    # Normalize and replace invalid values
    # --------------------------------------------------------

    def normalize(X):

        X = np.asarray(
            X,
            dtype=np.float32
        )

        # Replace NaN and Inf values BEFORE clipping.
        #
        # We use the midpoint of the training 1st/99th
        # percentile range for each channel.
        #
        # This avoids introducing artificial extreme values.
        midpoint = (
            channel_q01
            + channel_q99
        ) / 2.0

        invalid = ~np.isfinite(X)

        if np.any(invalid):

            for channel in range(
                X.shape[1]
            ):

                channel_invalid = invalid[
                    :,
                    channel,
                    :,
                    :
                ]

                if np.any(channel_invalid):

                    X[
                        :,
                        channel,
                        :,
                        :
                    ][channel_invalid] = midpoint[
                        channel
                    ]

        # Clip to training-derived range.

        X = np.clip(
            X,
            channel_q01[
                None,
                :,
                None,
                None
            ],
            channel_q99[
                None,
                :,
                None,
                None
            ]
        )

        # Scale to approximately [0, 1].

        X = (
            X
            - channel_q01[
                None,
                :,
                None,
                None
            ]
        ) / (
            channel_q99[
                None,
                :,
                None,
                None
            ]
            - channel_q01[
                None,
                :,
                None,
                None
            ]
        )

        return X.astype(
            np.float32
        )

    print("Normalizing training data...")

    X_train = normalize(
        X_train
    )

    print("Normalizing validation data...")

    X_val = normalize(
        X_val
    )

    print("Normalizing test data...")

    X_test = normalize(
        X_test
    )

    # --------------------------------------------------------
    # Final NaN / Inf check
    # --------------------------------------------------------

    print("\nFinal data-quality check:")

    all_valid = True

    for name, array in [
        ("X_train", X_train),
        ("X_val", X_val),
        ("X_test", X_test),
    ]:

        nan_count = int(
            np.isnan(array).sum()
        )

        inf_count = int(
            np.isinf(array).sum()
        )

        print(
            f"  {name}: NaN={nan_count}, "
            f"Inf={inf_count}"
        )

        if nan_count > 0 or inf_count > 0:
            all_valid = False

    if not all_valid:

        raise ValueError(
            "NaN or Inf values remain after normalization."
        )

    print("\nNormalized data ranges:")

    print(
        "  Train:",
        float(X_train.min()),
        "to",
        float(X_train.max())
    )

    print(
        "  Val  :",
        float(X_val.min()),
        "to",
        float(X_val.max())
    )

    print(
        "  Test :",
        float(X_test.min()),
        "to",
        float(X_test.max())
    )

    # --------------------------------------------------------
    # Save datasets
    # --------------------------------------------------------

    print("\nSaving CNN datasets...")

    np.save(
        CNN_DIR / "X_train.npy",
        X_train
    )

    np.save(
        CNN_DIR / "y_train.npy",
        train_y
    )

    np.save(
        CNN_DIR / "X_val.npy",
        X_val
    )

    np.save(
        CNN_DIR / "y_val.npy",
        val_y
    )

    np.save(
        CNN_DIR / "X_test.npy",
        X_test
    )

    np.save(
        CNN_DIR / "y_test.npy",
        test_y
    )

    np.save(
        CNN_DIR / "normalization_q01.npy",
        channel_q01
    )

    np.save(
        CNN_DIR / "normalization_q99.npy",
        channel_q99
    )

    np.save(
        CNN_DIR / "train_positions.npy",
        train_positions
    )

    np.save(
        CNN_DIR / "val_positions.npy",
        val_positions
    )

    np.save(
        CNN_DIR / "test_positions.npy",
        test_positions
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\nSaved files:")

    for filename in [
        "X_train.npy",
        "y_train.npy",
        "X_val.npy",
        "y_val.npy",
        "X_test.npy",
        "y_test.npy",
        "normalization_q01.npy",
        "normalization_q99.npy",
        "train_positions.npy",
        "val_positions.npy",
        "test_positions.npy",
    ]:

        print(
            " ",
            CNN_DIR / filename
        )

    print("\n" + "=" * 70)
    print("CNN DATASET CREATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()