from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


# ============================================================
# ECO-SENTINEL
# CNN PATCH DATASET PREPARATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SENTINEL_2023 = (
    PROJECT_ROOT
    / "data"
    / "raw_images"
    / "forest_samples"
    / "Forest_Sample_2023.tif"
)

SENTINEL_2025 = (
    PROJECT_ROOT
    / "data"
    / "raw_images"
    / "forest_samples"
    / "Forest_Sample_2025.tif"
)

HANSEN_REFERENCE = (
    PROJECT_ROOT
    / "data"
    / "supervised"
    / "hansen_reference.tif"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "cnn"

PATCH_SIZE = 15
HALF = PATCH_SIZE // 2

# Sentinel-2 spectral bands only.
# B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12
SPECTRAL_BANDS = [
    1,   # B1
    2,   # B2
    3,   # B3
    4,   # B4
    5,   # B5
    6,   # B6
    7,   # B7
    8,   # B8
    9,   # B8A
    10,  # B9
    11,  # B11
    12,  # B12
]


def main():

    print("=" * 60)
    print("ECO-SENTINEL CNN PATCH PREPARATION")
    print("=" * 60)

    print(f"Project root  : {PROJECT_ROOT}")
    print(f"2023 raster   : {SENTINEL_2023}")
    print(f"2025 raster   : {SENTINEL_2025}")
    print(f"Hansen        : {HANSEN_REFERENCE}")
    print(f"Patch size    : {PATCH_SIZE} x {PATCH_SIZE}")
    print(f"Spectral bands: {len(SPECTRAL_BANDS)} per year")
    print(f"Total channels: {len(SPECTRAL_BANDS) * 2}")

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    for path in [SENTINEL_2023, SENTINEL_2025, HANSEN_REFERENCE]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Open rasters
    # --------------------------------------------------------

    with rasterio.open(SENTINEL_2023) as src23, \
         rasterio.open(SENTINEL_2025) as src25, \
         rasterio.open(HANSEN_REFERENCE) as hansen:

        print("\n--- Raster information ---")

        print("Sentinel 2023:")
        print("  shape :", (src23.height, src23.width))
        print("  bands :", src23.count)
        print("  CRS   :", src23.crs)

        print("Sentinel 2025:")
        print("  shape :", (src25.height, src25.width))
        print("  bands :", src25.count)
        print("  CRS   :", src25.crs)

        print("Hansen:")
        print("  shape :", (hansen.height, hansen.width))
        print("  bands :", hansen.count)
        print("  CRS   :", hansen.crs)

        # ----------------------------------------------------
        # Validate Sentinel grids
        # ----------------------------------------------------

        if src23.width != src25.width or src23.height != src25.height:
            raise ValueError(
                "2023 and 2025 Sentinel rasters do not have the same shape."
            )

        if src23.transform != src25.transform:
            raise ValueError(
                "2023 and 2025 Sentinel rasters do not have the same grid."
            )

        if src23.crs != src25.crs:
            raise ValueError(
                "2023 and 2025 Sentinel rasters do not have the same CRS."
            )

        height = src23.height
        width = src23.width

        print("\nSentinel grids are aligned.")

        # ----------------------------------------------------
        # Read Sentinel spectral bands
        # ----------------------------------------------------

        print("\nReading Sentinel spectral bands...")

        image23 = src23.read(SPECTRAL_BANDS).astype(np.float32)
        image25 = src25.read(SPECTRAL_BANDS).astype(np.float32)

        print("2023 spectral shape:", image23.shape)
        print("2025 spectral shape:", image25.shape)

        # ----------------------------------------------------
        # Read Hansen reference
        # ----------------------------------------------------

        print("\nReading Hansen reference...")

        tree_cover = hansen.read(1)
        loss_year = hansen.read(2)

        print("Original Hansen shape:", tree_cover.shape)

        # ----------------------------------------------------
        # Reproject Hansen onto Sentinel grid
        # ----------------------------------------------------
        #
        # Hansen has a different resolution/grid from Sentinel.
        #
        # Nearest-neighbour is used because tree-cover and
        # loss-year are reference/categorical values.
        # ----------------------------------------------------

        print("\nReprojecting Hansen reference to Sentinel grid...")

        tree_cover_aligned = np.zeros(
            (height, width),
            dtype=np.float32
        )

        loss_year_aligned = np.zeros(
            (height, width),
            dtype=np.float32
        )

        reproject(
            source=tree_cover,
            destination=tree_cover_aligned,
            src_transform=hansen.transform,
            src_crs=hansen.crs,
            dst_transform=src23.transform,
            dst_crs=src23.crs,
            dst_width=width,
            dst_height=height,
            resampling=Resampling.nearest,
        )

        reproject(
            source=loss_year,
            destination=loss_year_aligned,
            src_transform=hansen.transform,
            src_crs=hansen.crs,
            dst_transform=src23.transform,
            dst_crs=src23.crs,
            dst_width=width,
            dst_height=height,
            resampling=Resampling.nearest,
        )

        tree_cover = tree_cover_aligned
        loss_year = loss_year_aligned

        print("Aligned Hansen shape:", tree_cover.shape)

        # ----------------------------------------------------
        # Create reference labels
        #
        # Forest:
        #   tree cover >= 30%
        #
        # Positive:
        #   forest + loss year 2024/2025
        #
        # Negative:
        #   forest + no recorded loss
        #
        # Non-forest:
        #   ignored (255)
        # ----------------------------------------------------

        forest = tree_cover >= 30

        positive = forest & np.isin(loss_year, [24, 25])
        negative = forest & (loss_year == 0)

        print("\n--- Reference labels ---")
        print("Forest pixels :", int(forest.sum()))
        print("Positive pixels:", int(positive.sum()))
        print("Negative pixels:", int(negative.sum()))

        # ----------------------------------------------------
        # Valid CNN patch centres
        #
        # A 15x15 patch needs 7 pixels on each side.
        # ----------------------------------------------------

        valid_centres = np.zeros(
            (height, width),
            dtype=bool
        )

        valid_centres[
            HALF : height - HALF,
            HALF : width - HALF
        ] = True

        positive &= valid_centres
        negative &= valid_centres

        positive_positions = np.argwhere(positive)
        negative_positions = np.argwhere(negative)

        print("\nValid patch centres:")
        print("  Positive:", len(positive_positions))
        print("  Negative:", len(negative_positions))

        # ----------------------------------------------------
        # Save source arrays
        #
        # We are NOT creating train/test patches yet.
        #
        # Spatial splitting will be performed separately so
        # neighbouring patches do not leak between datasets.
        # ----------------------------------------------------

        output_image23 = OUTPUT_DIR / "sentinel_2023_spectral.npy"
        output_image25 = OUTPUT_DIR / "sentinel_2025_spectral.npy"
        output_labels = OUTPUT_DIR / "hansen_labels.npy"

        print("\nSaving CNN source arrays...")

        np.save(output_image23, image23)
        np.save(output_image25, image25)

        # 0 = forest with no recorded loss
        # 1 = forest loss
        # 255 = ignored/non-forest
        labels = np.full(
            (height, width),
            255,
            dtype=np.uint8
        )

        labels[negative] = 0
        labels[positive] = 1

        np.save(output_labels, labels)

        print("\nSaved:")
        print(f"  {output_image23}")
        print(f"  {output_image25}")
        print(f"  {output_labels}")

    print("\n" + "=" * 60)
    print("CNN SOURCE DATA PREPARATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()