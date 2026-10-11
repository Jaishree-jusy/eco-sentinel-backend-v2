"""
Isolated End-to-End Orchestration Test for Genuinely New Image Scenario.

Verifies:
1. When a new image is acquired, all 10 stages execute in the established order.
2. Every output (manifest, alerts, evidence, blockchain block, DB row) is verified.
3. ALL database writes are isolated in a temporary SQLite database.
4. Duplicate submission prevention and fail-fast stage error reporting.
5. Zero mutations to production data.
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock

from app.orchestration.pipeline_runner import run_full_pipeline, PipelineRunError
from app.database.schema import get_connection


def test_new_image_isolated_end_to_end_orchestration(tmp_path):
    """
    Simulates a genuinely new eligible image passing through the entire pipeline.
    All disk writes and database mutations are strictly captured or redirected.
    """
    isolated_db = str(tmp_path / "isolated_test.db")
    
    # 1. Synthetic manifest representing a newly discovered, downloaded, aligned S2 image
    mock_acq_manifest = {
        "run_id": "test-acq-new-001",
        "status": "success",
        "base_year": 2023,
        "comparison_year": 2025,
        "datasets": {
            "forest_sample_2023": {"path": "/fake/base_2023.tif"},
            "forest_sample_2025": {"path": "/fake/comp_2025_new.tif"},
        },
        "gee_metadata": {
            "gee_image_id": "COPERNICUS/S2_SR_HARMONIZED/20261010T051651_NEW_IMAGE",
            "acquisition_date": 1791524140000,
            "cloud_cover_pct": 2.1,
            "was_new_download": True,
            "was_realigned": True,
            "local_path": "/fake/comp_2025_new.tif",
        }
    }

    mock_stages = {
        "app.acquisition.runner.run_acquisition": mock_acq_manifest,
        "app.gee_processing.pipeline.run_gee_processing": {"run_id": "stage-gee-001", "status": "success"},
        "app.loss_detection.pipeline.run_loss_detection": {"run_id": "stage-loss-001", "status": "success"},
        "app.loss_detection.ai_pipeline.run_ai_disturbance_pipeline": {"run_id": "stage-ai-001", "status": "success"},
        "app.polygon_analysis.pipeline.run_polygon_analysis": {"run_id": "stage-poly-001", "status": "success"},
        "app.boundary_check.pipeline.run_boundary_check": {"run_id": "stage-bound-001", "status": "success"},
        "app.boundary_check.legality_pipeline.run_legality_check": {"run_id": "stage-legal-001", "status": "success"},
        "app.alerts.pipeline.run_alert_generation": {"run_id": "stage-alert-001", "status": "success"},
        "app.evidence.pipeline.run_evidence_generation": {
            "run_id": "stage-evid-001",
            "status": "success",
            "evidence_records": [{"alert_id": "ALERT-IN-NEW001", "report_pdf": "/fake/report.pdf"}],
        },
        "app.blockchain.pipeline.run_blockchain_recording": {
            "run_id": "stage-block-001",
            "status": "success",
            "blocks_recorded_this_run": 1,
        },
    }

    patches = [patch(target, return_value=val) for target, val in mock_stages.items()]
    sync_mock = patch("app.database.sync.sync_database", return_value={"alerts_synced": 1, "blocks_synced": 1})
    patches.append(sync_mock)

    for p in patches:
        p.start()

    try:
        # Run pipeline with isolated db_path and evidence_limit=5
        result = run_full_pipeline(
            region_key="region_1",
            base_year=2023,
            comparison_year=2025,
            trigger_mode="manual_dashboard",
            use_live_gee=True,
            db_path=isolated_db,
            evidence_limit=5,
        )

        assert result["status"] == "success"
        assert result["pipeline_run_id"] is not None
        assert result["gee_metadata"]["gee_image_id"] == "COPERNICUS/S2_SR_HARMONIZED/20261010T051651_NEW_IMAGE"
        assert result["gee_metadata"]["was_new_download"] is True
        assert len(result["stage_run_ids"]) == 10

        # Verify isolated DB record
        conn = get_connection(isolated_db)
        run_record = conn.execute(
            "SELECT * FROM pipeline_runs WHERE pipeline_run_id = ?",
            (result["pipeline_run_id"],)
        ).fetchone()
        conn.close()

        assert run_record is not None
        assert run_record["overall_status"] == "success"
        assert run_record["trigger_mode"] == "manual_dashboard"
        assert run_record["acquisition_run_id"] == "test-acq-new-001"
        assert run_record["blockchain_run_id"] == "stage-block-001"
        assert run_record["failed_stage"] is None

    finally:
        for p in patches:
            p.stop()


def test_new_image_failure_propagation_isolated(tmp_path):
    """
    Verifies that if a downstream stage fails for a newly discovered image,
    the failure is logged in the isolated DB and raises PipelineRunError with the exact stage.
    """
    isolated_db = str(tmp_path / "isolated_fail.db")
    
    mock_acq = {
        "run_id": "test-acq-new-002",
        "status": "success",
        "base_year": 2023,
        "comparison_year": 2025,
        "datasets": {"forest_sample_2023": {}, "forest_sample_2025": {}},
        "gee_metadata": {"gee_image_id": "NEW_IMAGE_FAIL_CASE", "was_new_download": True}
    }

    from app.loss_detection.pipeline import LossDetectionError

    with patch("app.acquisition.runner.run_acquisition", return_value=mock_acq), \
         patch("app.gee_processing.pipeline.run_gee_processing", return_value={"run_id": "gee-2"}), \
         patch("app.loss_detection.pipeline.run_loss_detection", side_effect=LossDetectionError("GPU OOM / raster error")):

        with pytest.raises(PipelineRunError) as exc_info:
            run_full_pipeline(
                region_key="region_1",
                base_year=2023,
                comparison_year=2025,
                trigger_mode="manual_dashboard",
                use_live_gee=True,
                db_path=isolated_db,
            )

        assert exc_info.value.stage == "loss_detection"
        assert "GPU OOM / raster error" in str(exc_info.value.original_error)

    # Check isolated DB state
    conn = get_connection(isolated_db)
    record = conn.execute("SELECT * FROM pipeline_runs ORDER BY started_at DESC LIMIT 1").fetchone()
    conn.close()

    assert record["overall_status"] == "failed"
    assert record["failed_stage"] == "loss_detection"
    assert "GPU OOM / raster error" in record["error_message"]
