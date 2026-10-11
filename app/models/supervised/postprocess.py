from pathlib import Path

import geopandas as gpd
from shapely.ops import unary_union


INPUT = Path(
    r".\data\supervised\deforestation_polygons_2023_2025.geojson"
)

OUTPUT = Path(
    r".\data\supervised\deforestation_polygons_merged_10m.geojson"
)

MERGE_DISTANCE_M = 10.0


def main():
    print("Loading raw supervised polygons...")

    gdf = gpd.read_file(INPUT)

    print(f"Raw polygons: {len(gdf):,}")

    # Convert from latitude/longitude to metres.
    metric = gdf.to_crs("EPSG:32643")

    print(f"Merging polygons within {MERGE_DISTANCE_M:.0f} m...")

    buffered = metric.geometry.buffer(MERGE_DISTANCE_M)

    merged_buffers = unary_union(buffered)

    if merged_buffers.geom_type == "Polygon":
        groups = [merged_buffers]
    else:
        groups = list(merged_buffers.geoms)

    print(f"Connected groups: {len(groups):,}")

    output_rows = []

    for group_id, group in enumerate(groups, start=1):

        matches = metric.geometry.intersects(group)

        selected = metric.loc[matches]

        if selected.empty:
            continue

        merged = unary_union(selected.geometry)

        if merged.is_empty:
            continue

        total_area_ha = merged.area / 10_000.0

        mean_probability = selected[
            "deforestation_probability"
        ].mean()

        output_rows.append(
            {
                "merge_group": group_id,
                "prediction": "deforestation",
                "deforestation_probability": round(
                    float(mean_probability),
                    4,
                ),
                "area_ha": round(
                    float(total_area_ha),
                    4,
                ),
                "geometry": merged,
            }
        )

    result = gpd.GeoDataFrame(
        output_rows,
        geometry="geometry",
        crs=metric.crs,
    )

    result = result.to_crs("EPSG:4326")

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_file(
        OUTPUT,
        driver="GeoJSON",
    )

    print()
    print("POST-PROCESSING COMPLETE")
    print("=" * 50)
    print(f"Input polygons : {len(gdf):,}")
    print(f"Merged polygons: {len(result):,}")
    print(f"Output         : {OUTPUT}")
    print(
        f"Total area     : "
        f"{result['area_ha'].sum():,.4f} ha"
    )


if __name__ == "__main__":
    main()