"""
Bridge the supervised deforestation output into Module 6.

This does NOT modify Module 6.

It creates a Module-5-compatible polygon analysis manifest
that points Module 6 to the supervised boundary-input GeoJSON.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone

import geopandas as gpd

from app.config import POLYGON_ANALYSIS_DIR


INPUT = r".\data\supervised\deforestation_boundary_input_2023_2025.geojson"

BASE_YEAR = 2023
COMPARISON_YEAR = 2025


def main() -> None:

    print("=" * 60)
    print("SUPERVISED → MODULE 6 BRIDGE")
    print("=" * 60)

    if not os.path.exists(INPUT):
        raise FileNotFoundError(
            f"Supervised boundary input not found: {INPUT}"
        )

    print("Loading supervised boundary input...")

    gdf = gpd.read_file(INPUT)

    if gdf.empty:
        raise ValueError(
            "Supervised boundary input contains zero polygons."
        )

    print(f"Input polygons: {len(gdf):,}")

    required_columns = [
        "polygon_id",
        "prediction",
        "deforestation_probability",
        "area_ha",
        "centroid_lon",
        "centroid_lat",
        "geometry",
    ]

    missing = [
        column
        for column in required_columns
        if column not in gdf.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    os.makedirs(
        POLYGON_ANALYSIS_DIR,
        exist_ok=True,
    )

    run_id = str(uuid.uuid4())

    manifest = {
        "run_id": run_id,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "source_loss_detection_run_id": None,
        "base_year": BASE_YEAR,
        "comparison_year": COMPARISON_YEAR,
        "region": None,
        "geojson_path": os.path.abspath(INPUT),
        "stats": {
            "raw_polygon_count": int(len(gdf)),
            "filtered_polygon_count": int(len(gdf)),
            "min_area_ha_filter": None,
            "total_loss_area_ha": round(
                float(gdf["area_ha"].sum()),
                4,
            ),
            "severity_counts": {},
            "ai_scoring_applied": True,
            "detection_class_counts": {
                "deforestation": int(
                    (gdf["prediction"] == "deforestation").sum()
                )
            },
        },
        "source": "supervised_random_forest",
        "source_geojson": os.path.abspath(INPUT),
    }

    manifest_path = os.path.join(
        POLYGON_ANALYSIS_DIR,
        f"polygon_analysis_supervised_{BASE_YEAR}_{COMPARISON_YEAR}.json",
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print()
    print("BRIDGE COMPLETE")
    print("=" * 60)
    print(f"Polygons : {len(gdf):,}")
    print(
        f"Total area: "
        f"{gdf['area_ha'].sum():,.4f} ha"
    )
    print(f"Manifest : {manifest_path}")
    print()
    print(
        "Module 6 can now consume the supervised "
        "polygons without modifying Module 6."
    )


if __name__ == "__main__":
    main()