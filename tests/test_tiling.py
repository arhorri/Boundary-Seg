"""Tests for src/tiling.py — origin/edge logic, coordinate inverse, and the round-trip check."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import enhance  # noqa: E402
import io_utils  # noqa: E402
import tiling  # noqa: E402


def test_tile_origins_evenly_divisible_needs_no_flush_tile():
    # 2304 - 512 = 1792 = 7 * 256 exactly.
    origins = tiling._tile_origins(2304, 512, 256)
    assert origins[0] == 0
    assert origins[-1] == 2304 - 512
    assert len(origins) == 8
    assert all(b - a == 256 for a, b in zip(origins, origins[1:]))


def test_tile_origins_adds_flush_tile_when_not_evenly_divisible():
    # 3456 - 512 = 2944; 2944 / 256 = 11.5, so a flush tile must be appended.
    origins = tiling._tile_origins(3456, 512, 256)
    assert origins[0] == 0
    assert origins[-1] == 3456 - 512
    # The last gap is smaller than stride (flush tile overlaps its neighbour more than 50%).
    assert 0 < origins[-1] - origins[-2] <= 256


def test_tile_origins_raises_for_image_smaller_than_tile():
    with pytest.raises(ValueError):
        tiling._tile_origins(300, 512, 256)


def test_extract_tiles_covers_every_pixel_exactly_once_at_minimum():
    rng = np.random.default_rng(0)
    image = rng.random((1000, 1300, 3)).astype(np.float32)
    tiles = tiling.extract_tiles(image, tile_size_px=512, stride_px=256)

    coverage = np.zeros((1000, 1300), dtype=bool)
    for t in tiles:
        assert t["tile"].shape == (512, 512, 3)
        row0, col0 = t["row0"], t["col0"]
        coverage[row0 : row0 + 512, col0 : col0 + 512] = True
    assert coverage.all()


def test_extract_tiles_grid_indices_are_dense():
    rng = np.random.default_rng(0)
    image = rng.random((1000, 1300)).astype(np.float32)
    tiles = tiling.extract_tiles(image, tile_size_px=512, stride_px=256)
    n_rows = max(t["tile_row_idx"] for t in tiles) + 1
    n_cols = max(t["tile_col_idx"] for t in tiles) + 1
    assert len(tiles) == n_rows * n_cols


def test_tile_box_and_local_to_global():
    assert tiling.tile_box(row0=256, col0=512, tile_size_px=512) == (256, 768, 512, 1024)
    assert tiling.local_to_global(row0=256, col0=512, local_row=10, local_col=20) == (266, 532)


def test_expected_grain_count_per_tile_decreases_with_magnification():
    grains_100x = tiling.expected_grain_count_per_tile(tile_size_px=512, grain_diameter_px=40)
    grains_200x = tiling.expected_grain_count_per_tile(tile_size_px=512, grain_diameter_px=80)
    assert grains_200x < grains_100x
    # Area scales with the square of the linear diameter -> ~4x fewer grains at 2x magnification.
    assert grains_100x / grains_200x == pytest.approx(4.0, rel=0.01)


def test_round_trip_reconstruction_matches_source_exactly_synthetic(tmp_path):
    rng = np.random.default_rng(0)
    image = rng.random((1000, 1300, 3)).astype(np.float32)
    tiles = tiling.extract_tiles(image, tile_size_px=512, stride_px=256)

    log_rows = []
    for t in tiles:
        tile_path = tmp_path / f"tile_r{t['row0']:05d}_c{t['col0']:05d}.npy"
        np.save(tile_path, t["tile"])
        log_rows.append(
            {
                "tile_filename": tile_path.name,
                "source_filename": "synthetic.jpg",
                "magnification": 100,
                "tile_row_idx": t["tile_row_idx"],
                "tile_col_idx": t["tile_col_idx"],
                "row0": t["row0"],
                "col0": t["col0"],
                "tile_size_px": 512,
                "source_height_px": image.shape[0],
                "source_width_px": image.shape[1],
                "tile_path": str(tile_path),
            }
        )
    tile_log = pd.DataFrame(log_rows)

    reconstructed = tiling.reconstruct_image_from_tiles(tile_log, "synthetic.jpg")
    assert reconstructed.shape == image.shape
    assert np.array_equal(reconstructed, image)


def test_extract_dev_tiles_end_to_end_and_round_trip(tmp_path):
    if not enhance.ENHANCE_LOG_PATH.exists():
        pytest.skip("data/interim/enhanced/enhance_log.csv not present; run Step 5 first")

    log_df = tiling.extract_dev_tiles(
        output_dir=tmp_path / "tiles",
        log_path=tmp_path / "tile_log.csv",
    )

    expected_columns = {
        "tile_filename", "source_filename", "magnification", "tile_row_idx", "tile_col_idx",
        "row0", "col0", "tile_size_px", "source_height_px", "source_width_px", "tile_path",
    }
    assert expected_columns <= set(log_df.columns)
    assert set(log_df["magnification"]) >= {100, 200}
    assert (tmp_path / "tile_log.csv").exists()

    for _, row in log_df.iterrows():
        assert Path(row["tile_path"]).exists()

    # Round-trip every distinct source image against the real enhanced .npy on disk.
    for source_filename in log_df["source_filename"].unique():
        reconstructed = tiling.reconstruct_image_from_tiles(log_df, source_filename)
        enhance_log = pd.read_csv(enhance.ENHANCE_LOG_PATH)
        clahe_path = enhance_log[enhance_log["filename"] == source_filename]["clahe_output_path"].iloc[0]
        original = np.load(io_utils.REPO_ROOT / clahe_path)
        assert reconstructed.shape == original.shape
        assert np.array_equal(reconstructed, original), f"round-trip mismatch for {source_filename}"

    # 200x tiles must show fewer expected grains than 100x tiles for the same
    # source sample -- fewer grains per tile at higher magnification is
    # correct and must not have been "fixed" by tile-size scaling.
    tile_size_px = int(log_df["tile_size_px"].iloc[0])
    config = io_utils.load_config()
    grain_diameter_100x = config["enhance"]["grain_diameter_at_100x_px"]
    grains_100x = tiling.expected_grain_count_per_tile(
        tile_size_px, io_utils.scale_for_magnification(grain_diameter_100x, 100)
    )
    grains_200x = tiling.expected_grain_count_per_tile(
        tile_size_px, io_utils.scale_for_magnification(grain_diameter_100x, 200)
    )
    assert grains_200x < grains_100x
