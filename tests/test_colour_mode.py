"""Tests for src/colour_mode.py — effect-size math and the end-to-end diagnostic."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import colour_mode  # noqa: E402
import io_utils  # noqa: E402


def test_cohens_d_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 10_000)
    b = rng.normal(0, 1, 10_000)
    assert abs(colour_mode.cohens_d(a, b)) < 0.05


def test_cohens_d_matches_known_separation():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 100_000)
    b = rng.normal(1, 1, 100_000)  # 1 std-dev shift -> d ~= 1.0
    d = colour_mode.cohens_d(a, b)
    assert 0.9 < d < 1.1


def test_circular_effect_size_near_zero_across_the_wrap():
    rng = np.random.default_rng(0)
    # Two wide, "identical" hue clusters straddling the 0/360 boundary
    # should read as adjacent (small angular diff) and as a small effect
    # relative to their spread, not as maximally separated.
    a = rng.normal(359, 30, 10_000) % 360
    b = rng.normal(1, 30, 10_000) % 360
    effect, diff_deg = colour_mode.circular_effect_size(a, b)
    assert abs(diff_deg) < 5
    assert effect < 0.2


def test_circular_effect_size_large_for_opposite_hues():
    rng = np.random.default_rng(0)
    a = rng.normal(10, 2, 10_000) % 360
    b = rng.normal(190, 2, 10_000) % 360
    effect, diff_deg = colour_mode.circular_effect_size(a, b)
    assert effect > 5
    assert abs(abs(diff_deg) - 180) < 5


def test_bgr_uint8_to_hsv_float_pure_colours():
    # Pure red (BGR order): B=0, G=0, R=255
    red = np.array([[[0, 0, 255]]], dtype=np.uint8)
    hsv = colour_mode.bgr_uint8_to_hsv_float(red)
    assert abs(hsv[0, 0, 0] - 0.0) < 1e-3  # hue ~ 0 degrees
    assert abs(hsv[0, 0, 1] - 1.0) < 1e-3  # fully saturated
    assert abs(hsv[0, 0, 2] - 1.0) < 1e-3  # full value

    gray = np.array([[[128, 128, 128]]], dtype=np.uint8)
    hsv_gray = colour_mode.bgr_uint8_to_hsv_float(gray)
    assert hsv_gray[0, 0, 1] < 1e-3  # zero saturation


def test_decide_colour_mode_end_to_end(tmp_path):
    if not io_utils.RAW_DIR.exists():
        pytest.skip("data/raw not present in this environment")

    config_src = io_utils.CONFIG_PATH.read_text()
    config_copy = tmp_path / "default.yaml"
    config_copy.write_text(config_src)
    figure_path = tmp_path / "colour_mode_diagnostic.png"

    decision = colour_mode.decide_colour_mode(
        config_path=config_copy,
        figure_path=figure_path,
    )

    assert decision in {"colour", "grayscale", "disagreement"}
    assert figure_path.exists() and figure_path.stat().st_size > 0

    if decision != "disagreement":
        assert f"decision: {decision}" in config_copy.read_text()
