"""Tests for scripts/bootstrap_session.py — existence checks and the full idempotent run."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import enhance  # noqa: E402
import io_utils  # noqa: E402
import normalise  # noqa: E402
import tiling  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import bootstrap_session  # noqa: E402


def test_check_manifest_false_when_missing(tmp_path):
    assert bootstrap_session.check_manifest(tmp_path / "no_such_manifest.csv") is False


def test_check_manifest_true_when_present(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("filename\na.jpg\n")
    assert bootstrap_session.check_manifest(manifest_path) is True


def test_all_paths_exist_false_for_missing_log(tmp_path):
    assert bootstrap_session._all_paths_exist(tmp_path / "missing.csv", ["output_path"]) is False


def test_all_paths_exist_false_for_empty_log(tmp_path):
    log_path = tmp_path / "log.csv"
    pd.DataFrame(columns=["output_path"]).to_csv(log_path, index=False)
    assert bootstrap_session._all_paths_exist(log_path, ["output_path"]) is False


def test_all_paths_exist_false_when_referenced_file_missing(tmp_path):
    log_path = tmp_path / "log.csv"
    pd.DataFrame({"output_path": [str(tmp_path / "does_not_exist.npy")]}).to_csv(log_path, index=False)
    assert bootstrap_session._all_paths_exist(log_path, ["output_path"]) is False


def test_all_paths_exist_true_when_referenced_file_present(tmp_path):
    npy_path = tmp_path / "present.npy"
    np.save(npy_path, np.zeros((4, 4), dtype=np.float32))
    log_path = tmp_path / "log.csv"
    pd.DataFrame({"output_path": [str(npy_path)]}).to_csv(log_path, index=False)
    assert bootstrap_session._all_paths_exist(log_path, ["output_path"]) is True


def test_all_paths_exist_checks_every_listed_column(tmp_path):
    present = tmp_path / "present.npy"
    np.save(present, np.zeros((4, 4), dtype=np.float32))
    log_path = tmp_path / "log.csv"
    pd.DataFrame(
        {"a_path": [str(present)], "b_path": [str(tmp_path / "missing.npy")]}
    ).to_csv(log_path, index=False)
    assert bootstrap_session._all_paths_exist(log_path, ["a_path", "b_path"]) is False


def test_bootstrap_session_reports_found_when_everything_present():
    if not (
        io_utils.MANIFEST_PATH.exists()
        and normalise.CROP_LOG_PATH.exists()
        and enhance.ENHANCE_LOG_PATH.exists()
        and tiling.TILE_LOG_PATH.exists()
    ):
        pytest.skip("full pipeline (Steps 1/3/5/7) not already materialised in this environment")

    results = bootstrap_session.bootstrap_session()

    assert set(results) == {"manifest", "normalised images", "enhanced images", "tiles"}
    assert all(status == "found" for status in results.values()), results
