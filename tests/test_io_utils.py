"""Tests for src/io_utils.py — manifest building, split assignment, and I/O guards."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import io_utils  # noqa: E402


def test_scale_for_magnification_scales_linearly():
    assert io_utils.scale_for_magnification(30, 100, reference_magnification=100) == 30
    assert io_utils.scale_for_magnification(30, 200, reference_magnification=100) == 60
    assert io_utils.scale_for_magnification(30, 50, reference_magnification=100) == 15


def test_scale_for_magnification_reads_reference_from_config():
    if not io_utils.CONFIG_PATH.exists():
        pytest.skip("config/default.yaml not present in this environment")
    value = io_utils.scale_for_magnification(100, 200)
    assert value == 200  # reference_magnification in config/default.yaml is 100


def test_parse_filename_valid():
    parsed = io_utils.parse_filename("62990661-C-100X.JPG")
    assert parsed == {"sample_id": "62990661", "magnification": 100}

    parsed = io_utils.parse_filename("62990661-L-200X.JPG")
    assert parsed == {"sample_id": "62990661", "magnification": 200}


def test_parse_filename_invalid_raises():
    with pytest.raises(ValueError):
        io_utils.parse_filename("not_a_valid_name.jpg")
    with pytest.raises(ValueError):
        io_utils.parse_filename("62990661-C-100.JPG")  # missing trailing X


def test_build_manifest_on_real_raw_dir():
    if not io_utils.RAW_DIR.exists():
        pytest.skip("data/raw not present in this environment")

    df = io_utils.build_manifest()
    raw_files = io_utils.scan_raw_directory()

    assert len(df) == len(raw_files)
    assert list(df.columns) == io_utils.MANIFEST_COLUMNS
    assert set(df["magnification"]) <= {100, 200}
    assert df["height"].gt(0).all() and df["width"].gt(0).all()

    # Every magnification group with >1 image appears in both splits.
    for mag, group in df.groupby("magnification"):
        if len(group) > 1:
            assert set(group["split"]) == {"dev", "test"}
        else:
            assert set(group["split"]) == {"dev"}
            assert group["split_note"].iloc[0] != ""


def test_build_manifest_is_deterministic():
    if not io_utils.RAW_DIR.exists():
        pytest.skip("data/raw not present in this environment")

    df1 = io_utils.build_manifest()
    df2 = io_utils.build_manifest()
    pd_equal = df1.reset_index(drop=True).equals(df2.reset_index(drop=True))
    assert pd_equal


def test_load_image_rejects_test_split_from_non_evaluate_caller():
    if not io_utils.RAW_DIR.exists():
        pytest.skip("data/raw not present in this environment")

    df = io_utils.build_manifest()
    test_rows = df[df["split"] == "test"]
    if test_rows.empty:
        pytest.skip("no test-split rows produced for this raw set")

    with pytest.raises(PermissionError):
        io_utils.load_image(test_rows.iloc[0]["filename"], df)


def test_load_image_allows_dev_split():
    if not io_utils.RAW_DIR.exists():
        pytest.skip("data/raw not present in this environment")

    df = io_utils.build_manifest()
    dev_rows = df[df["split"] == "dev"]
    image = io_utils.load_image(dev_rows.iloc[0]["filename"], df)
    assert isinstance(image, np.ndarray)
    assert image.ndim in (2, 3)


def test_save_image_rejects_non_uint8_jpeg(tmp_path):
    arr16 = np.zeros((4, 4), dtype=np.uint16)
    with pytest.raises(ValueError):
        io_utils.save_image(tmp_path / "out.jpg", arr16)
