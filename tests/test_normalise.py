"""Tests for src/normalise.py — border detection, dtype round-trips, and the pipeline."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import io_utils  # noqa: E402
import normalise  # noqa: E402


def _textured(rng, shape):
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def test_detect_border_crop_finds_no_border_in_fully_textured_image():
    rng = np.random.default_rng(0)
    gray = _textured(rng, (400, 500))
    crop = normalise.detect_border_crop(gray, variance_ratio=0.05, max_crop_fraction=0.15)
    assert crop == {"top": 0, "bottom": 0, "left": 0, "right": 0, "capped": False}


def test_detect_border_crop_finds_uniform_top_strip():
    rng = np.random.default_rng(0)
    gray = _textured(rng, (400, 500))
    gray[:30, :] = 12  # uniform border strip, 30 px tall
    crop = normalise.detect_border_crop(gray, variance_ratio=0.05, max_crop_fraction=0.15)
    assert crop["top"] == 30
    assert crop["bottom"] == 0
    assert crop["left"] == 0
    assert crop["right"] == 0
    assert not crop["capped"]


def test_detect_border_crop_caps_runaway_crop():
    rng = np.random.default_rng(0)
    gray = _textured(rng, (400, 500))
    gray[:200, :] = 5  # half the image is "uniform" -- pathological input
    crop = normalise.detect_border_crop(gray, variance_ratio=0.05, max_crop_fraction=0.15)
    assert crop["top"] == int(400 * 0.15)
    assert crop["capped"] is True


def test_to_uint8_and_uint16_round_trip():
    rng = np.random.default_rng(0)
    arr = rng.random((10, 10)).astype(np.float32)
    u8 = normalise.to_uint8(arr)
    u16 = normalise.to_uint16(arr)
    assert u8.dtype == np.uint8 and u8.max() <= 255
    assert u16.dtype == np.uint16 and u16.max() <= 65535
    assert np.allclose(u8.astype(np.float32) / 255.0, arr, atol=1 / 255)


def test_to_uint8_clips_out_of_range_values():
    arr = np.array([-0.5, 0.0, 0.5, 1.0, 1.5], dtype=np.float32)
    u8 = normalise.to_uint8(arr)
    assert u8.tolist() == [0, 0, 128, 255, 255]


def test_normalise_image_grayscale_decision_produces_single_channel():
    if not io_utils.RAW_DIR.exists():
        pytest.skip("data/raw not present in this environment")
    manifest = io_utils.load_manifest()
    dev_filename = manifest[manifest["split"] == "dev"]["filename"].iloc[0]
    result = normalise.normalise_image(
        filename=dev_filename,
        manifest=manifest,
        raw_dir=io_utils.RAW_DIR,
        colour_decision="grayscale",
        variance_ratio=0.05,
        max_crop_fraction=0.15,
    )
    assert result["float_image"].ndim == 2
    assert result["float_image"].dtype == np.float32
    assert result["float_image"].min() >= 0.0 and result["float_image"].max() <= 1.0


def test_normalise_image_colour_decision_produces_rgb():
    if not io_utils.RAW_DIR.exists():
        pytest.skip("data/raw not present in this environment")
    manifest = io_utils.load_manifest()
    dev_filename = manifest[manifest["split"] == "dev"]["filename"].iloc[0]
    result = normalise.normalise_image(
        filename=dev_filename,
        manifest=manifest,
        raw_dir=io_utils.RAW_DIR,
        colour_decision="colour",
        variance_ratio=0.05,
        max_crop_fraction=0.15,
    )
    assert result["float_image"].ndim == 3
    assert result["float_image"].shape[2] == 3


def test_normalise_image_rejects_unknown_colour_decision():
    if not io_utils.RAW_DIR.exists():
        pytest.skip("data/raw not present in this environment")
    manifest = io_utils.load_manifest()
    dev_filename = manifest[manifest["split"] == "dev"]["filename"].iloc[0]
    with pytest.raises(ValueError):
        normalise.normalise_image(
            filename=dev_filename,
            manifest=manifest,
            raw_dir=io_utils.RAW_DIR,
            colour_decision="sepia",
            variance_ratio=0.05,
            max_crop_fraction=0.15,
        )


def test_normalise_dev_images_does_not_resize(tmp_path):
    if not io_utils.RAW_DIR.exists():
        pytest.skip("data/raw not present in this environment")

    manifest = io_utils.load_manifest()
    dev = manifest[manifest["split"] == "dev"]

    log_df = normalise.normalise_dev_images(
        output_dir=tmp_path / "normalised",
        log_path=tmp_path / "crop_log.csv",
        figure_path=tmp_path / "diagnostic.png",
    )

    assert len(log_df) == len(dev)
    for _, row in log_df.iterrows():
        manifest_row = dev[dev["filename"] == row["filename"]].iloc[0]
        # Output dims must equal original minus crop -- never resampled.
        assert row["output_height"] == manifest_row["height"] - row["crop_top_px"] - row["crop_bottom_px"]
        assert row["output_width"] == manifest_row["width"] - row["crop_left_px"] - row["crop_right_px"]
        saved = np.load(tmp_path / "normalised" / f"{Path(row['filename']).stem}.npy")
        assert saved.shape[0] == row["output_height"]
        assert saved.shape[1] == row["output_width"]
        assert saved.dtype == np.float32


def test_normalise_dev_images_requires_colour_mode_decision(tmp_path):
    if not io_utils.RAW_DIR.exists():
        pytest.skip("data/raw not present in this environment")

    bare_config = tmp_path / "default.yaml"
    bare_config.write_text("reference_magnification: 100\nrandom_seed: 42\n")

    with pytest.raises(KeyError):
        normalise.normalise_dev_images(
            config_path=bare_config,
            output_dir=tmp_path / "normalised",
            log_path=tmp_path / "crop_log.csv",
            figure_path=tmp_path / "diagnostic.png",
        )
