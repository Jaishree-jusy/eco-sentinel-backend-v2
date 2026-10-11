from pathlib import Path

import geopandas as gpd


INPUT = Path(
    r".\data\supervised\deforestation_polygons_merged_10m.geojson"
)

OUTPUT = Path(
    r".\data\supervised\deforestation_boundary_input_2023_2025.geojson"
)


def main():
    print("Loading merged supervised polygons...")

    gdf = gpd.read_file(INPUT)

    if gdf.empty:
        raise ValueError("Supervised polygon file is empty.")

    print(f"Input polygons: {len(gdf):,}")

    # Work in the original geographic CRS for the output.
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    gdf = gdf.to_crs("EPSG:4326")

    # Stable IDs based on the final geometry rather than merge_group.
    gdf["polygon_id"] = [
        f"SUP-2023-2025-{i:06d}"
        for i in range(1, len(gdf) + 1)
    ]

    # Centroids are calculated in a metric CRS to avoid geographic-CRS
    # centroid warnings, then converted back to longitude/latitude.
    metric = gdf.to_crs("EPSG:32643")
    centroids = metric.geometry.centroid.to_crs("EPSG:4326")

    gdf["centroid_lon"] = centroids.x
    gdf["centroid_lat"] = centroids.y

    # Keep the fields needed by Module 6 plus useful supervised metadata.
    gdf = gdf[
        [
            "polygon_id",
            "prediction",
            "deforestation_probability",
            "area_ha",
            "centroid_lon",
            "centroid_lat",
            "geometry",
        ]
    ]

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    gdf.to_file(
        OUTPUT,
        driver="GeoJSON",
    )

    print()
    print("BOUNDARY INPUT PREPARATION COMPLETE")
    print("=" * 50)
    print(f"Polygons : {len(gdf):,}")
    print(f"Output   : {OUTPUT}")
    print()
    print("Columns:")
    print(gdf.columns.tolist())


if __name__ == "__main__":
    main()