"""Tests for src/enhance.py — flat-field, boundary-preserving denoise, CLAHE, and the pipeline."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import enhance  # noqa: E402
import io_utils  # noqa: E402
import normalise  # noqa: E402


def _gradient_image(height=300, width=300):
    ramp = np.tile(np.linspace(0.2, 0.9, width, dtype=np.float32), (height, 1))
    return ramp


def _textured_image_with_thin_line(height=300, width=300, line_width_px=2):
    rng = np.random.default_rng(0)
    texture = rng.uniform(0.5, 0.8, size=(height, width)).astype(np.float32)
    # A vertical multiplicative background gradient, like an illumination artefact.
    gradient = np.tile(np.linspace(0.6, 1.0, height, dtype=np.float32).reshape(-1, 1), (1, width))
    image = texture * gradient
    mid = width // 2
    half = line_width_px // 2
    image[:, mid - half : mid - half + line_width_px] = 0.05  # thin dark "boundary"
    return np.clip(image, 0.0, 1.0).astype(np.float32), mid, half, line_width_px


def test_estimate_background_is_fast_and_smooth(tmp_path):
    image = _gradient_image()
    background = enhance.estimate_background(image, radius_px=50, working_radius_px=20)
    assert background.shape == image.shape
    # A pure-gradient input should be recovered smoothly, without the
    # sharpness the raw ramp doesn't have anyway -- just sanity-check range.
    assert background.min() >= 0.0
    assert background.max() <= 1.5


def test_flat_field_correct_removes_large_scale_gradient():
    image, *_ = _textured_image_with_thin_line()
    corrected, background = enhance.flat_field_correct(image, radius_px=50, working_radius_px=20)
    assert corrected.shape == image.shape
    assert background.shape == image.shape
    assert corrected.min() >= 0.0 and corrected.max() <= 1.0 + 1e-6

    # Row-mean trend should be much flatter after correction than before.
    raw_row_means = image.mean(axis=1)
    corrected_row_means = corrected.mean(axis=1)
    raw_spread = raw_row_means.max() - raw_row_means.min()
    corrected_spread = corrected_row_means.max() - corrected_row_means.min()
    assert corrected_spread < raw_spread


def test_assert_denoise_preserves_boundaries_passes_for_small_patch():
    limit = enhance.assert_denoise_preserves_boundaries(
        resolved_patch_size_px=3.0, magnification=100, boundary_width_at_100x_px=2.0
    )
    assert limit == pytest.approx(2.0)


def test_assert_denoise_preserves_boundaries_raises_for_large_patch():
    with pytest.raises(ValueError):
        enhance.assert_denoise_preserves_boundaries(
            resolved_patch_size_px=20.0, magnification=100, boundary_width_at_100x_px=2.0
        )


def test_denoise_image_rejects_oversized_patch_config():
    image, *_ = _textured_image_with_thin_line()
    with pytest.raises(ValueError):
        enhance.denoise_image(
            image,
            patch_size_px=20.0,  # radius 10px, way past the 2px 100x boundary limit
            patch_distance_px=6.0,
            h_multiplier=0.8,
            magnification=100,
            boundary_width_at_100x_px=2.0,
            noise_sigma_px=1.0,
        )


def test_denoise_image_preserves_shape_grayscale_and_colour():
    gray, *_ = _textured_image_with_thin_line()
    denoised_gray, info = enhance.denoise_image(
        gray, patch_size_px=3.0, patch_distance_px=6.0, h_multiplier=0.8,
        magnification=100, boundary_width_at_100x_px=2.0, noise_sigma_px=1.0,
    )
    assert denoised_gray.shape == gray.shape
    assert info["patch_size_px"] == 3

    colour = np.stack([gray, gray, gray], axis=-1)
    denoised_colour, _ = enhance.denoise_image(
        colour, patch_size_px=3.0, patch_distance_px=6.0, h_multiplier=0.8,
        magnification=100, boundary_width_at_100x_px=2.0, noise_sigma_px=1.0,
    )
    assert denoised_colour.shape == colour.shape


def test_apply_clahe_output_in_unit_range():
    image, *_ = _textured_image_with_thin_line()
    out = enhance.apply_clahe(image, kernel_size_px=100, clip_limit=0.01)
    assert out.shape == image.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_zoom_crop_box_scales_with_grain_diameter():
    shape = (2000, 2000)
    box_100x = enhance.zoom_crop_box(shape, grain_diameter_px=40, crop_size_in_grain_diameters=8)
    box_200x = enhance.zoom_crop_box(shape, grain_diameter_px=80, crop_size_in_grain_diameters=8)
    size_100x = box_100x[1] - box_100x[0]
    size_200x = box_200x[1] - box_200x[0]
    assert size_200x == pytest.approx(2 * size_100x, rel=0.05)


def test_zoom_crop_box_clips_to_image_bounds():
    box = enhance.zoom_crop_box((50, 50), grain_diameter_px=40, crop_size_in_grain_diameters=8)
    assert box[1] - box[0] <= 50
    assert box[3] - box[2] <= 50


def test_enhance_image_thin_boundary_survives_full_pipeline():
    image, mid, half, line_width = _textured_image_with_thin_line()
    stages, resolved = enhance.enhance_image(
        filename="synthetic.jpg",
        magnification=100,
        image=image,
        flat_field_radius_px=50,
        flat_field_working_radius_px=20,
        patch_size_px=3,
        patch_distance_px=6,
        h_multiplier=0.8,
        boundary_width_at_100x_px=2.0,
        noise_sigma_px=1.0,
        clahe_kernel_size_px=100,
        clahe_clip_limit=0.01,
    )
    final = stages["clahe"]
    assert final.shape == image.shape

    line_cols = slice(mid - half, mid - half + line_width)
    line_intensity = final[:, line_cols].mean()
    # Neighbourhood just outside the line on both sides.
    surround = np.concatenate(
        [final[:, mid - half - 10 : mid - half].ravel(), final[:, mid - half + line_width : mid - half + line_width + 10].ravel()]
    )
    surround_intensity = surround.mean()
    assert line_intensity < surround_intensity - 0.1, (
        f"thin boundary washed out: line={line_intensity:.3f} vs surround={surround_intensity:.3f}"
    )


def test_enhance_dev_images_end_to_end(tmp_path):
    if not normalise.CROP_LOG_PATH.exists():
        pytest.skip("data/interim/normalised/crop_log.csv not present; run Step 3 first")

    log_df = enhance.enhance_dev_images(
        output_dir=tmp_path / "enhanced",
        log_path=tmp_path / "enhance_log.csv",
        figure_dir=tmp_path / "figures",
    )

    expected_columns = {
        "filename", "magnification", "flat_field_radius_px", "denoise_patch_size_px",
        "denoise_patch_distance_px", "denoise_boundary_limit_px", "clahe_kernel_size_px",
        "flat_field_output_path", "denoised_output_path", "clahe_output_path", "figure_path",
    }
    assert expected_columns <= set(log_df.columns)

    for _, row in log_df.iterrows():
        for key in ("flat_field_output_path", "denoised_output_path", "clahe_output_path"):
            assert (io_utils.REPO_ROOT / row[key]).exists()
        assert (io_utils.REPO_ROOT / row["figure_path"]).exists()

    # At least one 100x and one 200x image were processed (required by the acceptance figure set).
    assert set(log_df["magnification"]) >= {100, 200}
