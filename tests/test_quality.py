"""Tests for src/quality.py — the four metrics, tiling, and the illumination acceptance check."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import io_utils  # noqa: E402
import normalise  # noqa: E402
import quality  # noqa: E402


def test_to_luminance_passes_through_grayscale():
    gray = np.random.default_rng(0).random((20, 20)).astype(np.float32)
    lum = quality.to_luminance(gray)
    assert lum.shape == gray.shape
    assert np.allclose(lum, gray)


def test_to_luminance_collapses_rgb():
    rgb = np.random.default_rng(0).random((20, 20, 3)).astype(np.float32)
    lum = quality.to_luminance(rgb)
    assert lum.shape == (20, 20)


def test_focus_higher_for_sharp_than_blurred():
    rng = np.random.default_rng(0)
    checkerboard = (np.indices((256, 256)).sum(axis=0) % 16 < 8).astype(np.float32)
    from skimage.filters import gaussian

    blurred = gaussian(checkerboard, sigma=5, preserve_range=True).astype(np.float32)
    sharp_focus = quality.focus_variance_of_laplacian(checkerboard, ksize=3)
    blurred_focus = quality.focus_variance_of_laplacian(blurred, ksize=3)
    assert sharp_focus > blurred_focus


def test_illumination_range_fraction_near_zero_for_uniform_image():
    flat = np.full((300, 300), 0.5, dtype=np.float32)
    fit = quality.fit_illumination(flat, sigma_px=50)
    frac = quality.illumination_range_fraction(fit, image_range=1.0)
    assert frac < 1e-6


def test_illumination_range_fraction_large_for_gradient_image():
    ramp = np.tile(np.linspace(0, 1, 300, dtype=np.float32), (300, 1))
    fit = quality.fit_illumination(ramp, sigma_px=20)
    image_range = float(ramp.max() - ramp.min())
    frac = quality.illumination_range_fraction(fit, image_range)
    assert frac > 0.5


def test_noise_mad_sigma_recovers_known_noise_level():
    rng = np.random.default_rng(0)
    flat = np.full((400, 400), 0.5, dtype=np.float32)
    true_sigma = 0.02
    noisy = flat + rng.normal(0, true_sigma, flat.shape).astype(np.float32)
    estimated = quality.noise_mad_sigma(noisy, sigma_px=1.0)
    assert abs(estimated - true_sigma) < 0.005


def test_contrast_metrics_zero_spread_for_constant_image():
    flat = np.full((50, 50), 0.3, dtype=np.float32)
    spread, entropy = quality.contrast_metrics(flat, 5, 95)
    assert spread == 0.0
    assert entropy == 0.0


def test_assess_image_quality_tile_grid_covers_whole_image():
    rng = np.random.default_rng(0)
    image = rng.random((1000, 1300)).astype(np.float32)
    rows = quality.assess_image_quality(
        filename="synthetic.jpg",
        magnification=100,
        image=image,
        laplacian_ksize=3,
        illumination_sigma_px=50,
        illumination_flag_threshold=0.10,
        noise_sigma_px=1.0,
        contrast_low_pct=5,
        contrast_high_pct=95,
        tile_size=512,
    )
    whole = [r for r in rows if r["scope"] == "image"]
    tiles = [r for r in rows if r["scope"] == "tile"]
    assert len(whole) == 1
    assert len(tiles) == 2 * 3  # ceil(1000/512) x ceil(1300/512)
    total_tile_area = sum(r["tile_height_px"] * r["tile_width_px"] for r in tiles)
    assert total_tile_area == image.shape[0] * image.shape[1]


def test_assess_quality_flags_illumination_on_100x_dark_band(tmp_path):
    if not normalise.CROP_LOG_PATH.exists():
        pytest.skip("data/interim/normalised/crop_log.csv not present; run Step 3 first")

    df = quality.assess_quality(
        output_csv_path=tmp_path / "quality_before.csv",
        heatmap_dir=tmp_path / "heatmaps",
    )

    whole_100x = df[(df["scope"] == "image") & (df["magnification"] == 100)]
    assert not whole_100x.empty
    assert whole_100x["illumination_flag"].any(), (
        "No 100x dev image was flagged for illumination non-uniformity; "
        "the known dark band should trip illumination_flag."
    )


def test_assess_quality_writes_expected_columns(tmp_path):
    if not normalise.CROP_LOG_PATH.exists():
        pytest.skip("data/interim/normalised/crop_log.csv not present; run Step 3 first")

    df = quality.assess_quality(
        output_csv_path=tmp_path / "quality_before.csv",
        heatmap_dir=tmp_path / "heatmaps",
    )
    expected = {
        "filename",
        "magnification",
        "scope",
        "tile_row",
        "tile_col",
        "focus_variance_of_laplacian",
        "illumination_level",
        "illumination_range_frac",
        "illumination_flag",
        "noise_mad_sigma",
        "contrast_spread",
        "entropy",
    }
    assert expected <= set(df.columns)
    assert (tmp_path / "quality_before.csv").exists()
    heatmaps = list((tmp_path / "heatmaps").glob("*.png"))
    assert len(heatmaps) == df["filename"].nunique()
