"""Tests for src/models/base.py and src/models/watershed_model.py.

Cellpose and SAM need heavy, environment-specific dependencies (a working
torch+torchvision pair, and for SAM a downloaded checkpoint); their tests
skip gracefully when those aren't available rather than asserting success,
consistent with this repo's "skip if data/deps absent" pattern. Watershed
and the base contract have no such dependency and are always exercised.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import io_utils  # noqa: E402
from models.base import BoundaryModel, labels_to_boundary_probability, validate_boundary_map  # noqa: E402
from models.pidinet_model import PidiNetBoundaryModel  # noqa: E402
from models.watershed_model import WatershedBoundaryModel  # noqa: E402


def test_boundary_model_is_abstract():
    with pytest.raises(TypeError):
        BoundaryModel()


def test_validate_boundary_map_accepts_valid_output():
    tile = np.zeros((64, 64, 3), dtype=np.float32)
    boundary_map = np.zeros((64, 64), dtype=np.float32)
    validate_boundary_map(tile, boundary_map)  # must not raise


def test_validate_boundary_map_rejects_shape_mismatch():
    tile = np.zeros((64, 64, 3), dtype=np.float32)
    boundary_map = np.zeros((32, 32), dtype=np.float32)
    with pytest.raises(ValueError):
        validate_boundary_map(tile, boundary_map)


def test_validate_boundary_map_rejects_wrong_dtype():
    tile = np.zeros((64, 64), dtype=np.float32)
    boundary_map = np.zeros((64, 64), dtype=np.float64)
    with pytest.raises(ValueError):
        validate_boundary_map(tile, boundary_map)


def test_validate_boundary_map_rejects_out_of_range():
    tile = np.zeros((64, 64), dtype=np.float32)
    boundary_map = np.full((64, 64), 1.5, dtype=np.float32)
    with pytest.raises(ValueError):
        validate_boundary_map(tile, boundary_map)


def test_labels_to_boundary_probability_shape_and_range():
    labels = np.zeros((100, 100), dtype=np.int32)
    labels[:, 50:] = 1  # two regions split by a vertical line
    boundary_map = labels_to_boundary_probability(labels, sigma_px=1.0)
    assert boundary_map.shape == labels.shape
    assert boundary_map.dtype == np.float32
    assert boundary_map.min() >= 0.0 and boundary_map.max() <= 1.0


def test_labels_to_boundary_probability_peaks_at_the_boundary():
    labels = np.zeros((100, 100), dtype=np.int32)
    labels[:, 50:] = 1
    boundary_map = labels_to_boundary_probability(labels, sigma_px=1.0)
    at_boundary = boundary_map[:, 48:52].mean()
    far_from_boundary = boundary_map[:, :20].mean()
    assert at_boundary > far_from_boundary


def test_watershed_model_scales_parameters_with_magnification():
    model_100x = WatershedBoundaryModel(magnification=100)
    model_200x = WatershedBoundaryModel(magnification=200)

    assert model_200x.gradient_smoothing_sigma_px == pytest.approx(2 * model_100x.gradient_smoothing_sigma_px)
    assert model_200x.h_minima_depth == pytest.approx(2 * model_100x.h_minima_depth)
    # min_region_area derives from grain diameter squared -> ~4x at 2x magnification.
    assert model_200x.min_region_area_px2 == pytest.approx(4 * model_100x.min_region_area_px2, rel=0.01)


def test_watershed_model_predicts_valid_boundary_map_on_synthetic_tile():
    rng = np.random.default_rng(0)
    tile = rng.random((256, 256, 3)).astype(np.float32)
    model = WatershedBoundaryModel(magnification=100)
    boundary_map = model.predict(tile)
    validate_boundary_map(tile, boundary_map)


def test_watershed_model_predicts_on_real_tile():
    tile_log_path = io_utils.REPO_ROOT / "data" / "interim" / "tile_log.csv"
    if not tile_log_path.exists():
        pytest.skip("data/interim/tile_log.csv not present; run Step 7 first")

    import pandas as pd

    tile_log = pd.read_csv(tile_log_path)
    row = tile_log[tile_log["magnification"] == 100].iloc[0]
    tile = np.load(row["tile_path"])

    model = WatershedBoundaryModel(magnification=100)
    boundary_map = model.predict(tile)
    validate_boundary_map(tile, boundary_map)


def test_pidinet_model_falls_back_or_runs_and_produces_valid_output():
    rng = np.random.default_rng(0)
    tile = rng.random((256, 256, 3)).astype(np.float32)

    model = PidiNetBoundaryModel(magnification=100)
    boundary_map = model.predict(tile)
    validate_boundary_map(tile, boundary_map)
    # Whichever path ran (real PiDiNet or the Canny fallback), the model
    # must be honest about which one it used.
    assert isinstance(model.substituted, bool)
    if model.substituted:
        assert model.substitution_reason != ""


def test_pidinet_diameter_free_but_sigma_scales_when_using_fallback():
    model_100x = PidiNetBoundaryModel(magnification=100)
    model_200x = PidiNetBoundaryModel(magnification=200)
    if not model_100x.substituted:
        pytest.skip("real PiDiNet available in this environment; fallback-only scaling check doesn't apply")

    sigma_100x = io_utils.scale_for_magnification(model_100x.canny_sigma_100x, 100)
    sigma_200x = io_utils.scale_for_magnification(model_200x.canny_sigma_100x, 200)
    assert sigma_200x == pytest.approx(2 * sigma_100x)


def test_cellpose_model_construction_or_documented_skip():
    tile_log_path = io_utils.REPO_ROOT / "data" / "interim" / "tile_log.csv"
    if not tile_log_path.exists():
        pytest.skip("data/interim/tile_log.csv not present; run Step 7 first")

    from models.cellpose_model import CellposeBoundaryModel

    try:
        model = CellposeBoundaryModel(magnification=100, gpu=False)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"cellpose unavailable in this environment: {exc}")

    import pandas as pd

    tile_log = pd.read_csv(tile_log_path)
    row = tile_log[tile_log["magnification"] == 100].iloc[0]
    tile = np.load(row["tile_path"])
    boundary_map = model.predict(tile)
    validate_boundary_map(tile, boundary_map)


def test_sam_model_requires_checkpoint_path():
    with pytest.raises(FileNotFoundError):
        from models.sam_model import SAMBoundaryModel

        SAMBoundaryModel(magnification=100)
