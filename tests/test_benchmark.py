"""Tests for src/benchmark.py — tile selection, graceful per-model failure, and the full run."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import benchmark  # noqa: E402
import io_utils  # noqa: E402
import tiling  # noqa: E402


def _fake_tile_log():
    rows = []
    for mag in (100, 200):
        for i in range(5):
            rows.append(
                {
                    "tile_filename": f"tile_{mag}_{i}.npy",
                    "source_filename": f"source_{mag}.jpg",
                    "magnification": mag,
                    "tile_path": f"/fake/tile_{mag}_{i}.npy",
                }
            )
    return pd.DataFrame(rows)


def test_select_benchmark_tiles_respects_count_per_magnification():
    tile_log = _fake_tile_log()
    selected = benchmark.select_benchmark_tiles(tile_log, tiles_per_magnification=2, seed=42)
    counts = selected["magnification"].value_counts()
    assert counts[100] == 2
    assert counts[200] == 2


def test_select_benchmark_tiles_is_deterministic():
    tile_log = _fake_tile_log()
    a = benchmark.select_benchmark_tiles(tile_log, tiles_per_magnification=2, seed=42)
    b = benchmark.select_benchmark_tiles(tile_log, tiles_per_magnification=2, seed=42)
    assert list(a["tile_filename"]) == list(b["tile_filename"])


def test_select_benchmark_tiles_clamps_to_available_tiles():
    tile_log = _fake_tile_log()
    selected = benchmark.select_benchmark_tiles(tile_log, tiles_per_magnification=100, seed=42)
    assert len(selected) == len(tile_log)  # can't select more than exist


def test_build_models_for_magnification_captures_failures_not_raises():
    models = benchmark.build_models_for_magnification(100, io_utils.CONFIG_PATH)
    assert set(models.keys()) == {"cellpose", "sam", "pidinet", "watershed"}
    # watershed and pidinet (its fallback needs no external weights) must always load.
    assert not isinstance(models["watershed"], Exception)
    assert not isinstance(models["pidinet"], Exception)
    # sam always fails to load without a configured checkpoint -- that must be
    # captured, not raised.
    assert isinstance(models["sam"], Exception)


def test_run_benchmark_end_to_end(tmp_path):
    if not tiling.TILE_LOG_PATH.exists():
        pytest.skip("data/interim/tile_log.csv not present; run Step 7 first")

    timing_df = benchmark.run_benchmark(
        timing_csv_path=tmp_path / "benchmark_timing.csv",
        figure_path=tmp_path / "benchmark_comparison.png",
    )

    assert (tmp_path / "benchmark_timing.csv").exists()
    assert (tmp_path / "benchmark_comparison.png").exists()

    expected_columns = {
        "tile_filename", "source_filename", "magnification", "model",
        "elapsed_seconds", "status", "detail",
    }
    assert expected_columns <= set(timing_df.columns)
    assert set(timing_df["magnification"]) >= {100, 200}
    assert set(timing_df["model"]) == {"cellpose", "sam", "pidinet", "watershed"}

    # Every row is one of the three documented outcomes.
    assert set(timing_df["status"]) <= {"ok", "load_failed", "predict_failed"}

    # watershed and pidinet must succeed on every selected tile in this
    # environment (no external weights needed); a full run must never
    # silently drop a magnification's timing entirely.
    for model_name in ("watershed", "pidinet"):
        model_rows = timing_df[timing_df["model"] == model_name]
        assert (model_rows["status"] == "ok").all(), f"{model_name} failed on some tiles: {model_rows}"
        assert set(model_rows["magnification"]) >= {100, 200}
