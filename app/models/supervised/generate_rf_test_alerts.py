
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone

import geopandas as gpd

from app.alerts.alert_builder import build_inside_forest_alert

ROOT = Path("data/supervised/live_runs")
BOUNDARY_DIR = Path("data/boundary_check")
TAG = "rf-test-ed15535c"

BOUNDARY_MANIFEST = (
    BOUNDARY_DIR
    / "boundary_check_89248402-27d9-4083-be16-219968ae4c93.json"
)
LEGALITY_FILE = (
    BOUNDARY_DIR
    / f"legality_checked_2023_2025_{TAG}.geojson"
)


def main():
    if not BOUNDARY_MANIFEST.exists():
        raise FileNotFoundError(BOUNDARY_MANIFEST)
    if not LEGALITY_FILE.exists():
        raise FileNotFoundError(LEGALITY_FILE)

    polygons = gpd.read_file(LEGALITY_FILE)

    if polygons.empty:
        raise RuntimeError("No inside-forest polygons found for the test.")

    generated_at = datetime.now(timezone.utc).isoformat()
    alerts = []

    for _, row in polygons.head(10).iterrows():
        # The existing builder expects mean_ml_probability.
        row = row.copy()
        row["mean_ml_probability"] = row.get(
            "deforestation_probability"
        )
        row["detection_class"] = "random_forest_deforestation"

        alert = build_inside_forest_alert(
            row,
            generated_at=generated_at,
        )

        # Unique test IDs ensure no existing evidence or block is reused.
        alert["alert_id"] = f"RFTEST-{uuid.uuid4().hex[:12]}"
        alerts.append(alert)

    output = ROOT / f"alerts_inside_{TAG}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(alerts, indent=2), encoding="utf-8")

    print(f"Test alerts created: {len(alerts)}")
    print(f"Output: {output}")
    print("Sample alert:")
    print(json.dumps(alerts[0], indent=2))


if __name__ == "__main__":
    main()
