"""Tests for src/gate.py — relative-change math and the Step 6 QC gate."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import enhance  # noqa: E402
import gate  # noqa: E402
import io_utils  # noqa: E402
import quality  # noqa: E402


def test_illumination_relative_decrease_basic():
    assert gate.illumination_relative_decrease(0.2, 0.1) == pytest.approx(0.5)
    assert gate.illumination_relative_decrease(0.2, 0.2) == pytest.approx(0.0)
    assert gate.illumination_relative_decrease(0.2, 0.3) == pytest.approx(-0.5)


def test_illumination_relative_decrease_zero_before():
    assert gate.illumination_relative_decrease(0.0, 0.0) == 0.0
    assert gate.illumination_relative_decrease(0.0, 0.1) == -float("inf")


def test_noise_relative_increase_basic():
    assert gate.noise_relative_increase(0.02, 0.01) == pytest.approx(-0.5)
    assert gate.noise_relative_increase(0.02, 0.03) == pytest.approx(0.5)


def test_noise_relative_increase_zero_before():
    assert gate.noise_relative_increase(0.0, 0.0) == 0.0
    assert gate.noise_relative_increase(0.0, 0.01) == float("inf")


def test_focus_relative_decrease_basic():
    assert gate.focus_relative_decrease(10.0, 5.0) == pytest.approx(0.5)
    assert gate.focus_relative_decrease(10.0, 15.0) == pytest.approx(-0.5)


def _row(**kwargs):
    return pd.Series(kwargs)


def test_gate_image_all_pass():
    before = _row(filename="a.jpg", magnification=100, illumination_range_frac=0.2, noise_mad_sigma=0.02, focus_variance_of_laplacian=10.0)
    after = _row(filename="a.jpg", magnification=100, illumination_range_frac=0.05, noise_mad_sigma=0.015, focus_variance_of_laplacian=12.0)
    cfg = {"illumination_relative_improvement_min": 0.5, "noise_relative_increase_max": 0.0, "focus_relative_decrease_max": 0.0}
    result = gate.gate_image(before, after, cfg)
    assert result["illumination_pass"] and result["noise_pass"] and result["focus_pass"]
    assert result["gate_pass"] is True


def test_gate_image_fails_on_insufficient_illumination_improvement():
    before = _row(filename="a.jpg", magnification=100, illumination_range_frac=0.2, noise_mad_sigma=0.02, focus_variance_of_laplacian=10.0)
    after = _row(filename="a.jpg", magnification=100, illumination_range_frac=0.15, noise_mad_sigma=0.015, focus_variance_of_laplacian=12.0)
    cfg = {"illumination_relative_improvement_min": 0.5, "noise_relative_increase_max": 0.0, "focus_relative_decrease_max": 0.0}
    result = gate.gate_image(before, after, cfg)
    assert result["illumination_pass"] is False
    assert result["gate_pass"] is False


def test_gate_image_fails_on_noise_increase():
    before = _row(filename="a.jpg", magnification=100, illumination_range_frac=0.2, noise_mad_sigma=0.02, focus_variance_of_laplacian=10.0)
    after = _row(filename="a.jpg", magnification=100, illumination_range_frac=0.05, noise_mad_sigma=0.03, focus_variance_of_laplacian=12.0)
    cfg = {"illumination_relative_improvement_min": 0.5, "noise_relative_increase_max": 0.0, "focus_relative_decrease_max": 0.0}
    result = gate.gate_image(before, after, cfg)
    assert result["noise_pass"] is False
    assert result["gate_pass"] is False


def test_gate_image_fails_on_focus_decrease():
    before = _row(filename="a.jpg", magnification=100, illumination_range_frac=0.2, noise_mad_sigma=0.02, focus_variance_of_laplacian=10.0)
    after = _row(filename="a.jpg", magnification=100, illumination_range_frac=0.05, noise_mad_sigma=0.015, focus_variance_of_laplacian=8.0)
    cfg = {"illumination_relative_improvement_min": 0.5, "noise_relative_increase_max": 0.0, "focus_relative_decrease_max": 0.0}
    result = gate.gate_image(before, after, cfg)
    assert result["focus_pass"] is False
    assert result["gate_pass"] is False


def test_run_qc_gate_end_to_end(tmp_path):
    if not quality.QUALITY_CSV_PATH.exists():
        pytest.skip("data/outputs/quality_before.csv not present; run Step 4 first")
    if not enhance.ENHANCE_LOG_PATH.exists():
        pytest.skip("data/interim/enhanced/enhance_log.csv not present; run Step 5 first")

    comparison_df = gate.run_qc_gate(
        quality_after_csv_path=tmp_path / "quality_after.csv",
        gate_comparison_csv_path=tmp_path / "quality_gate.csv",
        heatmap_dir=tmp_path / "heatmaps_after",
    )

    expected_columns = {
        "filename", "magnification",
        "illumination_range_frac_before", "illumination_range_frac_after",
        "illumination_relative_decrease", "illumination_pass",
        "noise_mad_sigma_before", "noise_mad_sigma_after", "noise_relative_increase", "noise_pass",
        "focus_variance_of_laplacian_before", "focus_variance_of_laplacian_after",
        "focus_relative_decrease", "focus_pass", "gate_pass",
    }
    assert expected_columns <= set(comparison_df.columns)
    assert (tmp_path / "quality_after.csv").exists()
    assert (tmp_path / "quality_gate.csv").exists()
    assert set(comparison_df["magnification"]) >= {100, 200}

    # Re-derive gate_pass independently from the logged before/after numbers
    # to confirm the printed verdicts match the underlying arithmetic.
    for _, row in comparison_df.iterrows():
        expected = (
            row["illumination_relative_decrease"] >= 0.5
            and row["noise_relative_increase"] <= 0.0
            and row["focus_relative_decrease"] <= 0.0
        )
        assert row["gate_pass"] == expected


def test_run_qc_gate_requires_gate_config_section(tmp_path):
    if not quality.QUALITY_CSV_PATH.exists():
        pytest.skip("data/outputs/quality_before.csv not present; run Step 4 first")
    if not enhance.ENHANCE_LOG_PATH.exists():
        pytest.skip("data/interim/enhanced/enhance_log.csv not present; run Step 5 first")

    bare_config = tmp_path / "default.yaml"
    bare_config.write_text(io_utils.CONFIG_PATH.read_text().split("\ngate:")[0])

    with pytest.raises(KeyError):
        gate.run_qc_gate(
            config_path=bare_config,
            quality_after_csv_path=tmp_path / "quality_after.csv",
            gate_comparison_csv_path=tmp_path / "quality_gate.csv",
            heatmap_dir=tmp_path / "heatmaps_after",
        )
