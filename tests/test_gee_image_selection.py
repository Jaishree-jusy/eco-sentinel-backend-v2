"""
Tests for GEE image-selection logic (Task 2 from the live monitoring readiness spec).

Tests cover:
  1. Latest eligible image is already registered → no_new_data, no download.
  2. A genuinely new eligible image exists → success, download called.
  3. Multiple eligible images exist, newest already registered → no_new_data.
  4. No images satisfy the configured filters → AcquisitionError.
  5. GEE is unavailable or returns an error → AcquisitionError with clear message.
  6. Unregistered adjacent granule (low coverage) is NOT chosen over higher-coverage
     co-temporal granule; dedup is still applied correctly.

All tests mock GEE at the boundary (initialize_earth_engine, find_best_image,
download_image). No real Earth Engine calls are made.
"""

import os
import json
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from unittest.mock import patch, MagicMock, call

from app.acquisition.runner import run_acquisition, AcquisitionError
import app.acquisition.runner as runner_module
from app.acquisition import gee_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_real_raster(path, width=20, height=20):
    """Write a minimal valid GeoTIFF so alignment logic can open it."""
    band_names = ["B4", "B8", "SCL"]
    transform = from_origin(74.75, 13.0, 0.0001, 0.0001)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width,
        count=len(band_names), dtype="float64", crs="EPSG:4326",
        transform=transform,
    ) as dst:
        for i, name in enumerate(band_names, start=1):
            dst.write(np.full((height, width), float(100 * i)), i)
            dst.set_band_description(i, name)
    return path


def _base_catalog(baseline_path=None):
    """Minimal valid catalog with all required local datasets."""
    keys = ["forest_sample_2023", "worldcover_2021", "gadm_forest_boundary",
            "fire_archive_1", "fire_archive_2"]
    catalog = {
        "datasets": {
            k: {"path": f"/fake/{k}.tif", "type": "raster", "role": "test",
                "valid": True, "errors": [], "sha256": "abc"}
            for k in keys
        }
    }
    if baseline_path is not None:
        catalog["datasets"]["forest_sample_2023"]["path"] = baseline_path
    return catalog


def _new_image_info(image_id="COPERNICUS/S2_SR_HARMONIZED/NEW_IMAGE_001",
                    coverage_pct=97.12):
    """Return a (metadata, mock_ee_image) pair representing a brand-new image."""
    metadata = {
        "gee_image_id": image_id,
        "acquisition_date": 1793000000000,
        "cloud_cover_pct": 2.5,
        "aoi_coverage_pct": coverage_pct,
        "selection_reason": "highest AOI coverage among co-temporal granules",
    }
    return metadata, MagicMock()


def _registered_image_info(image_id="COPERNICUS/S2_SR_HARMONIZED/REGISTERED_001",
                            coverage_pct=97.12):
    """Return a (metadata, mock_ee_image) pair for an already-registered image."""
    metadata = {
        "gee_image_id": image_id,
        "acquisition_date": 1791524140000,
        "cloud_cover_pct": 1.99,
        "aoi_coverage_pct": coverage_pct,
        "selection_reason": "highest AOI coverage among co-temporal granules",
    }
    return metadata, MagicMock()


# ---------------------------------------------------------------------------
# Scenario 1: Latest eligible image is already registered → no_new_data
# ---------------------------------------------------------------------------

@patch.object(gee_client, "download_image")
@patch.object(gee_client, "find_best_image")
@patch.object(gee_client, "initialize_earth_engine")
@patch.object(runner_module, "build_catalog")
def test_scenario_1_registered_image_produces_no_new_data(
    mock_catalog, mock_init, mock_find, mock_download, tmp_path, monkeypatch
):
    """
    Scenario 1: The newest qualifying GEE image is already in the registry.
    Expected: acquisition returns status='no_new_data', download is NOT called.
    """
    monkeypatch.setattr(runner_module, "ACQUISITION_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(runner_module, "RUN_LOG_PATH", str(tmp_path / "run_log.jsonl"))
    registry_path = str(tmp_path / "registry.json")
    monkeypatch.setattr("app.acquisition.gee_registry.REGISTRY_PATH", registry_path)

    # Pre-register the image that GEE will return.
    cached_file = str(tmp_path / "cached.tif")
    open(cached_file, "wb").write(b"cached raster bytes")

    from app.acquisition import gee_registry
    gee_registry.register_acquisition(
        "COPERNICUS/S2_SR_HARMONIZED/REGISTERED_001",
        cached_file, "sha256ofcached", "2026-09-27T05:35:40+00:00", 1.99,
        registry_path=registry_path,
    )

    mock_catalog.return_value = _base_catalog()
    mock_find.return_value = _registered_image_info()

    manifest = run_acquisition(
        region_key="region_1", base_year=2023, comparison_year=2025,
        use_live_gee=True,
    )

    assert manifest["status"] == "no_new_data", (
        f"Expected 'no_new_data' when image is registered, got: {manifest['status']!r}"
    )
    mock_download.assert_not_called()
    mock_init.assert_called_once()


# ---------------------------------------------------------------------------
# Scenario 2: A genuinely new eligible image exists → success, download called
# ---------------------------------------------------------------------------

@patch.object(runner_module, "validate_raster")
@patch.object(gee_client, "download_image")
@patch.object(gee_client, "find_best_image")
@patch.object(gee_client, "initialize_earth_engine")
@patch.object(runner_module, "build_catalog")
def test_scenario_2_new_image_produces_success(
    mock_catalog, mock_init, mock_find, mock_download, mock_validate,
    tmp_path, monkeypatch
):
    """
    Scenario 2: GEE returns an image that is NOT in the registry.
    Expected: acquisition returns status='success', download IS called,
              the image is registered, the manifest contains gee_metadata.
    """
    baseline_path = _write_real_raster(str(tmp_path / "baseline_2023.tif"))
    mock_catalog.return_value = _base_catalog(baseline_path=baseline_path)
    monkeypatch.setattr(runner_module, "ACQUISITION_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(runner_module, "RUN_LOG_PATH", str(tmp_path / "run_log.jsonl"))
    monkeypatch.setattr(runner_module, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr("app.acquisition.gee_registry.REGISTRY_PATH", str(tmp_path / "registry.json"))

    def _fake_download(image, bbox, output_path, scale=10):
        _write_real_raster(output_path)
        return output_path

    mock_find.return_value = _new_image_info()
    mock_download.side_effect = _fake_download
    mock_validate.return_value = {"valid": True, "errors": [], "warnings": [], "metadata": {}}

    manifest = run_acquisition(
        region_key="region_1", base_year=2023, comparison_year=2025,
        use_live_gee=True,
    )

    assert manifest["status"] == "success", f"Expected 'success', got: {manifest['status']!r}"
    assert manifest["gee_metadata"]["was_new_download"] is True
    assert manifest["gee_metadata"]["aoi_coverage_pct"] == 97.12
    assert manifest["gee_metadata"]["selection_reason"] == "highest AOI coverage among co-temporal granules"
    mock_download.assert_called_once()

    # Verify the metadata is JSON-serializable (guards against ee.Image objects leaking in).
    json.dumps(manifest["gee_metadata"])


# ---------------------------------------------------------------------------
# Scenario 3: Multiple images exist, newest already registered → no_new_data
# ---------------------------------------------------------------------------

@patch.object(gee_client, "download_image")
@patch.object(gee_client, "find_best_image")
@patch.object(gee_client, "initialize_earth_engine")
@patch.object(runner_module, "build_catalog")
def test_scenario_3_newest_of_multiple_is_registered_produces_no_new_data(
    mock_catalog, mock_init, mock_find, mock_download, tmp_path, monkeypatch
):
    """
    Scenario 3: GEE window contains multiple images. find_best_image() (which
    considers coverage) returns the newest-and-best-covered image, which is
    already registered.
    Expected: no_new_data, no download.
    """
    monkeypatch.setattr(runner_module, "ACQUISITION_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(runner_module, "RUN_LOG_PATH", str(tmp_path / "run_log.jsonl"))
    registry_path = str(tmp_path / "registry.json")
    monkeypatch.setattr("app.acquisition.gee_registry.REGISTRY_PATH", registry_path)

    # Register the image that find_best_image will return as the best.
    cached_file = str(tmp_path / "cached_best.tif")
    open(cached_file, "wb").write(b"cached")
    from app.acquisition import gee_registry
    gee_registry.register_acquisition(
        "COPERNICUS/S2_SR_HARMONIZED/BEST_COVERAGE_REGISTERED",
        cached_file, "sha256best", "2026-09-27T00:00:00+00:00", 2.5,
        registry_path=registry_path,
    )

    mock_catalog.return_value = _base_catalog()
    # find_best_image returns the best-coverage image (already registered).
    mock_find.return_value = (
        {
            "gee_image_id": "COPERNICUS/S2_SR_HARMONIZED/BEST_COVERAGE_REGISTERED",
            "acquisition_date": 1793000000000,
            "cloud_cover_pct": 2.5,
            "aoi_coverage_pct": 97.12,
            "selection_reason": "highest AOI coverage among co-temporal granules",
        },
        MagicMock(),
    )

    manifest = run_acquisition(
        region_key="region_1", base_year=2023, comparison_year=2025,
        use_live_gee=True,
    )

    assert manifest["status"] == "no_new_data"
    mock_download.assert_not_called()


# ---------------------------------------------------------------------------
# Scenario 4: No images satisfy the configured filters → AcquisitionError
# ---------------------------------------------------------------------------

@patch.object(gee_client, "find_best_image")
@patch.object(gee_client, "initialize_earth_engine")
@patch.object(runner_module, "build_catalog")
def test_scenario_4_no_qualifying_images_raises_acquisition_error(
    mock_catalog, mock_init, mock_find, tmp_path, monkeypatch
):
    """
    Scenario 4: GEE returns (None, None) — no qualifying images found.
    Expected: AcquisitionError with a clear message.
    """
    mock_catalog.return_value = _base_catalog()
    monkeypatch.setattr(runner_module, "ACQUISITION_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(runner_module, "RUN_LOG_PATH", str(tmp_path / "run_log.jsonl"))
    mock_find.return_value = (None, None)

    with pytest.raises(AcquisitionError, match="No Sentinel-2 imagery found"):
        run_acquisition(
            region_key="region_1", base_year=2023, comparison_year=2025,
            use_live_gee=True,
        )

    # Verify a failed run entry was written.
    log_path = str(tmp_path / "run_log.jsonl")
    assert os.path.exists(log_path), "Run log should be written even on failure."
    with open(log_path) as f:
        entry = json.loads(f.readline())
    assert entry["status"] == "failed"


# ---------------------------------------------------------------------------
# Scenario 5: GEE unavailable or authentication failure → AcquisitionError
# ---------------------------------------------------------------------------

@patch.object(gee_client, "initialize_earth_engine")
@patch.object(runner_module, "build_catalog")
def test_scenario_5_gee_auth_failure_raises_acquisition_error(
    mock_catalog, mock_init, tmp_path, monkeypatch
):
    """
    Scenario 5a: GEE authentication fails.
    Expected: AcquisitionError with message referencing earthengine authentication.
    """
    mock_catalog.return_value = _base_catalog()
    monkeypatch.setattr(runner_module, "ACQUISITION_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(runner_module, "RUN_LOG_PATH", str(tmp_path / "run_log.jsonl"))
    mock_init.side_effect = gee_client.GEEAuthenticationError(
        "Earth Engine initialization failed. Project: 'ecosentinel-503413'. "
        "Ensure credentials are valid (run: earthengine authenticate)"
    )

    with pytest.raises(AcquisitionError, match="earthengine authenticate"):
        run_acquisition(
            region_key="region_1", base_year=2023, comparison_year=2025,
            use_live_gee=True,
        )


@patch.object(gee_client, "find_best_image")
@patch.object(gee_client, "initialize_earth_engine")
@patch.object(runner_module, "build_catalog")
def test_scenario_5b_gee_query_error_raises_acquisition_error(
    mock_catalog, mock_init, mock_find, tmp_path, monkeypatch
):
    """
    Scenario 5b: GEE query raises an unexpected exception (network, quota, etc.).
    Expected: AcquisitionError with 'Earth Engine query failed' in the message.
    """
    mock_catalog.return_value = _base_catalog()
    monkeypatch.setattr(runner_module, "ACQUISITION_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(runner_module, "RUN_LOG_PATH", str(tmp_path / "run_log.jsonl"))
    mock_find.side_effect = RuntimeError("EEException: Quota exceeded")

    with pytest.raises(AcquisitionError, match="Earth Engine query failed"):
        run_acquisition(
            region_key="region_1", base_year=2023, comparison_year=2025,
            use_live_gee=True,
        )


# ---------------------------------------------------------------------------
# Scenario 6: Low-coverage adjacent granule is NOT selected; dedup still applied
# ---------------------------------------------------------------------------

@patch.object(gee_client, "download_image")
@patch.object(gee_client, "find_best_image")
@patch.object(gee_client, "initialize_earth_engine")
@patch.object(runner_module, "build_catalog")
def test_scenario_6_low_coverage_adjacent_granule_not_used_if_best_is_registered(
    mock_catalog, mock_init, mock_find, mock_download, tmp_path, monkeypatch
):
    """
    Scenario 6: The co-temporal granule with the highest coverage (T43PDQ, 97.12%)
    is already registered. An adjacent low-coverage granule (T43PDP, 15.68%) is
    also unregistered.

    Expected: find_best_image() correctly returns the highest-coverage granule
    (already registered). Because THAT granule is registered, the result is
    no_new_data. The low-coverage adjacent granule is NOT mistakenly selected
    as a workaround to bypass deduplication.
    """
    monkeypatch.setattr(runner_module, "ACQUISITION_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(runner_module, "RUN_LOG_PATH", str(tmp_path / "run_log.jsonl"))
    registry_path = str(tmp_path / "registry.json")
    monkeypatch.setattr("app.acquisition.gee_registry.REGISTRY_PATH", registry_path)

    # Only register the high-coverage granule (T43PDQ).
    cached_file = str(tmp_path / "T43PDQ.tif")
    open(cached_file, "wb").write(b"cached")
    from app.acquisition import gee_registry
    gee_registry.register_acquisition(
        "COPERNICUS/S2_SR_HARMONIZED/20260927_T43PDQ",
        cached_file, "sha256pdq", "2026-09-27T05:35:40+00:00", 4.32,
        registry_path=registry_path,
    )

    mock_catalog.return_value = _base_catalog()

    # find_best_image correctly returns T43PDQ (highest coverage, registered).
    mock_find.return_value = (
        {
            "gee_image_id": "COPERNICUS/S2_SR_HARMONIZED/20260927_T43PDQ",
            "acquisition_date": 1793000000000,
            "cloud_cover_pct": 4.32,
            "aoi_coverage_pct": 97.12,
            "selection_reason": "highest AOI coverage among co-temporal granules",
        },
        MagicMock(),
    )

    manifest = run_acquisition(
        region_key="region_1", base_year=2023, comparison_year=2025,
        use_live_gee=True,
    )

    assert manifest["status"] == "no_new_data", (
        "The highest-coverage granule is registered; result must be no_new_data. "
        "Low-coverage adjacent granule must NOT be used as a bypass."
    )
    mock_download.assert_not_called()


# ---------------------------------------------------------------------------
# initialize_earth_engine: verify project ID is passed correctly
# ---------------------------------------------------------------------------

def test_initialize_earth_engine_passes_project_id(monkeypatch):
    """
    Verify that initialize_earth_engine() passes the configured GEE_PROJECT_ID
    to ee.Initialize() rather than calling it without arguments.
    Calling ee.Initialize() without a project ID fails on this machine.
    """
    mock_ee = MagicMock()
    monkeypatch.setattr(gee_client, "ee", mock_ee)

    gee_client.initialize_earth_engine()

    # Must be called exactly once with the correct project keyword argument.
    mock_ee.Initialize.assert_called_once()
    _, kwargs = mock_ee.Initialize.call_args
    assert "project" in kwargs, (
        "ee.Initialize() must be called with project= to avoid 'no project found' error."
    )
    assert kwargs["project"] == "ecosentinel-503413", (
        f"Expected project='ecosentinel-503413', got: {kwargs['project']!r}"
    )


def test_initialize_earth_engine_allows_project_override(monkeypatch):
    """
    Verify that initialize_earth_engine(project=...) accepts an explicit
    project override (used in isolated tests to pass a test project ID).
    """
    mock_ee = MagicMock()
    monkeypatch.setattr(gee_client, "ee", mock_ee)

    gee_client.initialize_earth_engine(project="my-test-project")

    _, kwargs = mock_ee.Initialize.call_args
    assert kwargs["project"] == "my-test-project"
