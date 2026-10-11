"""
Tests for Module 10 - Government Dashboard.

Uses monkeypatching to inject synthetic data into the dashboard routes,
so we can verify every page actually renders (catches Jinja2 template
errors, broken url_for() references, missing context variables) without
needing your real multi-GB pipeline output or a real SQLite database.
"""

import pytest
from app import create_app
import app.dashboards.government.routes as routes_module


SAMPLE_ALERT = {
    "alert_id": "ALERT-IN-00001",
    "date_time": "2026-08-04T00:00:00",
    "location": {"latitude": 12.85, "longitude": 74.90},
    "area_lost_ha": 2.5,
    "severity_level": "moderate",
    "region_division": "Dakshina Kannada",
    "polygon_id": "LOSS-abc123def456",
    "geometry": {"type": "Polygon", "coordinates": [[[74.9, 12.85], [74.91, 12.85], [74.91, 12.86], [74.9, 12.85]]]},
    "evidence_available": False,
    "evidence": None,
    "blockchain_available": False,
    "blockchain": None,
}

SAMPLE_META = {
    "base_year": 2023, "comparison_year": 2025, "total_alerts": 1,
    "total_area_ha": 2.5, "evidence_generated_count": 0,
}


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_dashboard_no_data(client, monkeypatch):
    monkeypatch.setattr(routes_module, "load_dashboard_data", lambda: (None, None))
    response = client.get("/government/")
    assert response.status_code == 200
    assert b"No data available" in response.data


def test_dashboard_with_data(client, monkeypatch):
    monkeypatch.setattr(routes_module, "load_dashboard_data", lambda: ([SAMPLE_ALERT], SAMPLE_META))
    response = client.get("/government/")
    assert response.status_code == 200
    assert b"2023" in response.data


def test_alerts_geojson_endpoint(client, monkeypatch):
    monkeypatch.setattr(routes_module, "load_dashboard_data", lambda: ([SAMPLE_ALERT], SAMPLE_META))
    response = client.get("/government/api/alerts.geojson")
    assert response.status_code == 200
    data = response.get_json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 1
    assert data["features"][0]["properties"]["alert_id"] == "ALERT-IN-00001"


def test_alerts_list_page(client, monkeypatch):
    # alerts_list() now queries get_summary + get_alerts_page_data directly
    # (SQL-level pagination), not load_dashboard_data - mock those instead.
    monkeypatch.setattr(routes_module, "get_summary", lambda branch: SAMPLE_META)
    monkeypatch.setattr(routes_module, "get_alerts_page_data", lambda page, per_page: ([SAMPLE_ALERT], 1))
    response = client.get("/government/alerts")
    assert response.status_code == 200
    assert b"ALERT-IN-00001" in response.data


def test_alerts_list_no_data(client, monkeypatch):
    monkeypatch.setattr(routes_module, "get_summary", lambda branch: None)
    response = client.get("/government/alerts")
    assert response.status_code == 200
    assert b"No data available" in response.data


def test_alert_detail_page_found(client, monkeypatch):
    monkeypatch.setattr(routes_module, "get_alert_by_id", lambda aid: SAMPLE_ALERT if aid == "ALERT-IN-00001" else None)
    response = client.get("/government/alerts/ALERT-IN-00001")
    assert response.status_code == 200
    assert b"Dakshina Kannada" in response.data
    assert b"Evidence not yet generated" in response.data  # honest state, not faked


def test_alert_detail_page_not_found(client, monkeypatch):
    monkeypatch.setattr(routes_module, "get_alert_by_id", lambda aid: None)
    response = client.get("/government/alerts/NONEXISTENT")
    assert response.status_code == 404


def test_analytics_page(client, monkeypatch):
    # analytics() now queries get_summary + get_analytics_data directly
    # (SQL GROUP BY), not load_dashboard_data - mock those instead.
    monkeypatch.setattr(routes_module, "get_summary", lambda branch: SAMPLE_META)
    monkeypatch.setattr(
        routes_module, "get_analytics_data",
        lambda: ({"mild": 0, "moderate": 1, "severe": 0}, [("Dakshina Kannada", 1)]),
    )
    response = client.get("/government/analytics")
    assert response.status_code == 200
    assert b"Dakshina Kannada" in response.data


def test_analytics_no_data(client, monkeypatch):
    monkeypatch.setattr(routes_module, "get_summary", lambda branch: None)
    response = client.get("/government/analytics")
    assert response.status_code == 200
    assert b"No data available" in response.data


def test_blockchain_page_no_ledger(client, monkeypatch, tmp_path):
    fake_ledger = str(tmp_path / "nonexistent_ledger.jsonl")
    monkeypatch.setattr(routes_module, "LEDGER_PATH", fake_ledger)
    response = client.get("/government/blockchain")
    assert response.status_code == 200
    assert b"No blockchain ledger found" in response.data


def test_evidence_file_path_traversal_blocked(client):
    response = client.get("/government/evidence/ALERT-IN-00001/../../../etc/passwd")
    assert response.status_code in (403, 404)


def test_live_monitoring_status_scheduler_stopped(client, monkeypatch, tmp_path):
    monkeypatch.setattr(routes_module, "ACQUISITION_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(routes_module, "is_scheduler_running", lambda: False)
    response = client.get("/government/api/live-monitoring/status")
    assert response.status_code == 200
    data = response.get_json()
    assert data["monitoring"]["enabled"] is False
    assert data["monitoring"]["scheduler_running"] is False
    assert data["latest_acquisition"]["run_id"] is None


def test_live_monitoring_status_scheduler_running(client, monkeypatch, tmp_path):
    monkeypatch.setattr(routes_module, "ACQUISITION_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(routes_module, "is_scheduler_running", lambda: True)
    response = client.get("/government/api/live-monitoring/status")
    assert response.status_code == 200
    data = response.get_json()
    assert data["monitoring"]["enabled"] is True
    assert data["monitoring"]["scheduler_running"] is True


def test_live_monitoring_status_with_successful_run(client, monkeypatch, tmp_path):
    import json
    run_file = tmp_path / "run_test1.json"
    manifest = {
        "run_id": "test-run-123",
        "triggered_at": "2026-10-09T10:00:00+00:00",
        "status": "success",
        "comparison_year": 2025,
        "datasets": {
            "forest_sample_2025": {"path": "/path/to/raster.tif"}
        },
        "gee_metadata": {
            "gee_image_id": "COPERNICUS/S2_SR_HARMONIZED/test_img",
            "acquisition_date": 1790487340240,
            "cloud_cover_pct": 2.5,
            "was_new_download": True,
            "was_realigned": True,
            "local_path": "/path/to/raster.tif",
        }
    }
    run_file.write_text(json.dumps(manifest))
    monkeypatch.setattr(routes_module, "ACQUISITION_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(routes_module, "is_scheduler_running", lambda: False)

    response = client.get("/government/api/live-monitoring/status")
    assert response.status_code == 200
    data = response.get_json()
    acq = data["latest_acquisition"]
    assert acq["status"] == "success"
    assert acq["run_id"] == "test-run-123"
    assert acq["gee_image_id"] == "COPERNICUS/S2_SR_HARMONIZED/test_img"
    assert acq["cloud_cover_pct"] == 2.5
    assert acq["local_path"] == "/path/to/raster.tif"
    assert acq["error_message"] is None


def test_live_monitoring_status_with_failed_run(client, monkeypatch, tmp_path):
    import json
    run_file = tmp_path / "run_test_failed.json"
    manifest = {
        "run_id": "test-failed-456",
        "triggered_at": "2026-10-10T12:00:00+00:00",
        "status": "failed",
        "error_message": "Earth Engine query timeout",
        "gee_metadata": None,
    }
    run_file.write_text(json.dumps(manifest))
    monkeypatch.setattr(routes_module, "ACQUISITION_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(routes_module, "is_scheduler_running", lambda: False)

    response = client.get("/government/api/live-monitoring/status")
    assert response.status_code == 200
    data = response.get_json()
    acq = data["latest_acquisition"]
    assert acq["status"] == "failed"
    assert acq["run_id"] == "test-failed-456"
    assert acq["error_message"] == "Earth Engine query timeout"
    assert acq["gee_image_id"] is None


def test_fetch_live_data_success(client, monkeypatch):
    from app.dashboards.government import routes as routes_mod

    mock_result = {
        "pipeline_run_id": "pipeline-run-101",
        "status": "success",
        "stage_run_ids": {
            "acquisition": "fetch-run-789",
            "gee_processing": "gee-run-1",
            "database_sync": "sync-run-1",
        },
        "sync_stats": {"alerts_synced": 5},
        "gee_metadata": {
            "gee_image_id": "COPERNICUS/S2_SR_HARMONIZED/fetched_img",
            "acquisition_date": 1790487340240,
            "cloud_cover_pct": 1.5,
            "was_new_download": True,
            "was_realigned": True,
            "local_path": "/path/to/downloaded.tif",
        },
        "acquisition_manifest": {
            "datasets": {"forest_sample_2025": {"path": "/path/to/downloaded.tif"}}
        },
    }
    monkeypatch.setattr(routes_mod, "run_full_pipeline", lambda **kw: mock_result)

    response = client.post("/government/api/live-monitoring/fetch")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["status"] == "success"
    assert data["pipeline_run_id"] == "pipeline-run-101"
    assert data["run_id"] == "fetch-run-789"
    assert data["sync_stats"]["alerts_synced"] == 5
    assert "successfully" in data["message"]
    assert data["gee"]["image_id"] == "COPERNICUS/S2_SR_HARMONIZED/fetched_img"
    assert data["gee"]["local_path"] == "/path/to/downloaded.tif"


def test_fetch_live_data_no_new_data(client, monkeypatch):
    from app.dashboards.government import routes as routes_mod

    mock_result = {
        "pipeline_run_id": "pipeline-run-102",
        "status": "no_new_data",
        "message": "No new satellite imagery available - downstream pipeline not rerun, previous results preserved.",
        "stage_run_ids": {"acquisition": "acq-no-new"},
        "gee_metadata": {
            "gee_image_id": "COPERNICUS/S2_SR_HARMONIZED/cached_img",
            "acquisition_date": 1790487340240,
            "cloud_cover_pct": 3.0,
            "was_new_download": False,
            "was_realigned": False,
            "local_path": "/path/to/cached.tif",
        },
        "acquisition_manifest": {
            "datasets": {"forest_sample_2025": {"path": "/path/to/cached.tif"}}
        },
    }
    monkeypatch.setattr(routes_mod, "run_full_pipeline", lambda **kw: mock_result)

    response = client.post("/government/api/live-monitoring/fetch")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["status"] == "no_new_data"
    assert data["pipeline_run_id"] == "pipeline-run-102"
    assert "No new satellite imagery available" in data["message"]
    assert data["gee"]["was_new_download"] is False


def test_fetch_live_data_failure(client, monkeypatch):
    from app.dashboards.government import routes as routes_mod
    from app.orchestration.pipeline_runner import PipelineRunError

    def fake_run(**kw):
        raise PipelineRunError("acquisition", RuntimeError("GEE quota exceeded"))

    monkeypatch.setattr(routes_mod, "run_full_pipeline", fake_run)
    response = client.post("/government/api/live-monitoring/fetch")
    assert response.status_code == 500
    data = response.get_json()
    assert data["success"] is False
    assert data["status"] == "failed"
    assert data["failed_stage"] == "acquisition"
    assert "GEE quota exceeded" in data["message"]


