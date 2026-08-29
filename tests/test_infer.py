"""Tests for src/infer.py — path/log helpers, resumability, and per-tile failure handling."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import infer  # noqa: E402
import io_utils  # noqa: E402
import tiling  # noqa: E402


def test_prediction_path_layout(tmp_path):
    path = infer.prediction_path("watershed", "62990661-C-100X.JPG", "62990661-C-100X_r00000_c00000.npy", tmp_path)
    assert path == tmp_path / "watershed" / "62990661-C-100X" / "62990661-C-100X_r00000_c00000.npy"


def test_append_log_row_writes_header_once(tmp_path):
    log_path = tmp_path / "log.csv"
    row = {k: "x" for k in infer.LOG_FIELDNAMES}
    infer._append_log_row(log_path, row)
    infer._append_log_row(log_path, row)

    lines = log_path.read_text().splitlines()
    assert lines[0] == ",".join(infer.LOG_FIELDNAMES)
    assert len(lines) == 3  # header + two data rows
    assert lines.count(lines[0]) == 1  # header not duplicated


def _write_fake_tile(tmp_path, name, size=32):
    path = tmp_path / name
    np.save(path, np.random.default_rng(0).random((size, size, 3)).astype(np.float32))
    return path


def _fake_tile_log(tmp_path, n_per_mag=3):
    rows = []
    for mag in (100, 200):
        for i in range(n_per_mag):
            tile_path = _write_fake_tile(tmp_path, f"tile_{mag}_{i}.npy")
            rows.append(
                {
                    "tile_filename": f"tile_{mag}_{i}.npy",
                    "source_filename": f"source_{mag}.jpg",
                    "magnification": mag,
                    "tile_path": str(tile_path),
                }
            )
    log_path = tmp_path / "tile_log.csv"
    pd.DataFrame(rows).to_csv(log_path, index=False)
    return log_path


def test_run_model_inference_watershed_end_to_end_and_resumable(tmp_path):
    tile_log_path = _fake_tile_log(tmp_path)
    predictions_dir = tmp_path / "predictions"
    log_path = tmp_path / "inference_log.csv"

    summary = infer.run_model_inference(
        "watershed",
        tile_log_path=tile_log_path,
        predictions_dir=predictions_dir,
        log_path=log_path,
    )
    assert summary["total"] == 6
    assert summary["already_done"] == 0
    assert summary["ran"] == 6
    assert summary["ok"] == 6
    assert summary["failed"] == 0

    tile_log = pd.read_csv(tile_log_path)
    for _, row in tile_log.iterrows():
        out = infer.prediction_path("watershed", row["source_filename"], row["tile_filename"], predictions_dir)
        assert out.exists()
        saved = np.load(out)
        assert saved.dtype == np.float32
        assert saved.min() >= 0.0 and saved.max() <= 1.0

    log_df = pd.read_csv(log_path)
    assert len(log_df) == 6
    assert (log_df["status"] == "ok").all()

    # Second run: everything already present -> nothing recomputed, log untouched.
    summary_2 = infer.run_model_inference(
        "watershed",
        tile_log_path=tile_log_path,
        predictions_dir=predictions_dir,
        log_path=log_path,
    )
    assert summary_2["already_done"] == 6
    assert summary_2["ran"] == 0
    log_df_2 = pd.read_csv(log_path)
    assert len(log_df_2) == 6  # unchanged


def test_run_model_inference_resumes_partial_progress(tmp_path):
    tile_log_path = _fake_tile_log(tmp_path)
    predictions_dir = tmp_path / "predictions"
    log_path = tmp_path / "inference_log.csv"

    tile_log = pd.read_csv(tile_log_path)
    first_row = tile_log.iloc[0]
    pre_existing = infer.prediction_path(
        "watershed", first_row["source_filename"], first_row["tile_filename"], predictions_dir
    )
    pre_existing.parent.mkdir(parents=True, exist_ok=True)
    np.save(pre_existing, np.zeros((10, 10), dtype=np.float32))

    summary = infer.run_model_inference(
        "watershed",
        tile_log_path=tile_log_path,
        predictions_dir=predictions_dir,
        log_path=log_path,
    )
    assert summary["already_done"] == 1
    assert summary["ran"] == 5
    # The pre-existing file must not have been overwritten by a real prediction.
    assert np.array_equal(np.load(pre_existing), np.zeros((10, 10), dtype=np.float32))


def test_run_model_inference_logs_and_skips_failures_without_crashing(tmp_path, monkeypatch):
    tile_log_path = _fake_tile_log(tmp_path, n_per_mag=2)
    predictions_dir = tmp_path / "predictions"
    log_path = tmp_path / "inference_log.csv"

    class FlakyModel:
        def __init__(self, magnification, config_path=None):
            self.magnification = magnification
            self.calls = 0

        def predict(self, tile):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("synthetic failure")
            return np.zeros(tile.shape[:2], dtype=np.float32)

    monkeypatch.setitem(infer.MODEL_CLASSES, "watershed", FlakyModel)

    summary = infer.run_model_inference(
        "watershed",
        tile_log_path=tile_log_path,
        predictions_dir=predictions_dir,
        log_path=log_path,
    )
    # One FlakyModel instance is constructed per magnification (100X, 200X),
    # and each fresh instance fails on its own first call -- so the "fails
    # once" behaviour fires once per magnification group, i.e. twice here.
    assert summary["ran"] == 4
    assert summary["failed"] == 2
    assert summary["ok"] == 2
    assert len(summary["failures"]) == 2
    assert all("synthetic failure" in f["detail"] for f in summary["failures"])

    log_df = pd.read_csv(log_path)
    assert (log_df["status"] == "failed").sum() == 2
    assert (log_df["status"] == "ok").sum() == 2


def test_run_model_inference_unknown_model_raises():
    with pytest.raises(ValueError):
        infer.run_model_inference("not_a_real_model")


def test_run_all_inference_orders_sam_last():
    assert infer.DEFAULT_MODEL_ORDER[-1] == "sam"
    assert infer.DEFAULT_MODEL_ORDER[0] in ("watershed", "pidinet")


def test_run_all_inference_runs_only_requested_models(tmp_path):
    tile_log_path = _fake_tile_log(tmp_path, n_per_mag=1)
    predictions_dir = tmp_path / "predictions"
    log_path = tmp_path / "inference_log.csv"

    summaries = infer.run_all_inference(
        models=["watershed"],
        tile_log_path=tile_log_path,
        predictions_dir=predictions_dir,
        log_path=log_path,
    )
    assert set(summaries) == {"watershed"}
    assert summaries["watershed"]["ok"] == 2


def test_full_dev_tile_set_end_to_end_watershed_and_pidinet(tmp_path):
    if not tiling.TILE_LOG_PATH.exists():
        pytest.skip("data/interim/tile_log.csv not present; run Step 7 first")

    tile_log = pd.read_csv(tiling.TILE_LOG_PATH)
    total = len(tile_log)
    predictions_dir = tmp_path / "predictions"
    log_path = tmp_path / "inference_log.csv"

    for model_name in ("watershed", "pidinet"):
        summary = infer.run_model_inference(
            model_name, predictions_dir=predictions_dir, log_path=log_path
        )
        present = summary["already_done"] + summary["ok"]
        assert present == total, f"{model_name}: only {present}/{total} predictions present"
        assert summary["failed"] == 0, f"{model_name} had failures: {summary['failures']}"
