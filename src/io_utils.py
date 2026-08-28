"""Loading, saving, and manifest handling for the boundary-segmentation pipeline.

All raw-image discovery, magnification parsing, dev/test splitting, and
dtype-preserving image I/O live here so no other module scatters ad-hoc
filename parsing or cv2 calls.
"""

from __future__ import annotations

import hashlib
import inspect
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.csv"
CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# <id>-<letter>-<mag>X.<ext>  e.g. 62990661-C-100X.JPG
FILENAME_RE = re.compile(
    r"^(?P<sample_id>[A-Za-z0-9]+)-(?P<letter>[A-Za-z])-(?P<magnification>\d+)X\.(?P<ext>[A-Za-z0-9]+)$"
)

MANIFEST_COLUMNS = [
    "filename",
    "sample_id",
    "magnification",
    "height",
    "width",
    "split",
    "split_note",
]


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load the pipeline config from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def config_hash(config_path: Path = CONFIG_PATH) -> str:
    """Return a short sha256 hash of the raw config file bytes, for logging."""
    return hashlib.sha256(Path(config_path).read_bytes()).hexdigest()[:12]


def scale_for_magnification(
    value_at_100x: float,
    magnification: float,
    reference_magnification: Optional[float] = None,
    config_path: Path = CONFIG_PATH,
) -> float:
    """Resolve a spatial parameter declared at reference magnification to the working value.

    Every spatial parameter (a length in pixels: kernel radius, patch size,
    expected feature size, tolerance, ...) is declared in config/default.yaml
    at `reference_magnification` (100x) and must be resolved through this
    function for the image actually being processed — features scale
    linearly with magnification. See CLAUDE.md "Magnification handling".

    Args:
        value_at_100x: the parameter's value in pixels at reference magnification.
        magnification: the magnification of the image being processed.
        reference_magnification: overrides the config's reference magnification.
        config_path: where to read the reference magnification from, if not given.

    Returns:
        The parameter's value in pixels, scaled for `magnification`.
    """
    if reference_magnification is None:
        reference_magnification = load_config(config_path)["reference_magnification"]
    return value_at_100x * (magnification / reference_magnification)


def parse_filename(filename: str) -> dict:
    """Parse a raw-image filename into its sample id and magnification.

    Expected pattern: ``<id>-<letter>-<mag>X.<ext>``, e.g. ``62990661-C-100X.JPG``.

    Raises:
        ValueError: if the filename does not match the expected pattern.
            Never falls back to an assumed magnification.
    """
    match = FILENAME_RE.match(filename)
    if match is None:
        raise ValueError(
            f"Filename {filename!r} does not match expected pattern "
            f"'<id>-<letter>-<mag>X.<ext>' (e.g. '62990661-C-100X.JPG'). "
            "Refusing to guess a magnification."
        )
    return {
        "sample_id": match.group("sample_id"),
        "magnification": int(match.group("magnification")),
    }


def scan_raw_directory(raw_dir: Path = RAW_DIR) -> list[Path]:
    """Return the sorted list of image files in ``raw_dir``.

    Sorted by filename so downstream ordering (and therefore the seeded
    split) is deterministic regardless of filesystem iteration order.
    """
    raw_dir = Path(raw_dir)
    files = [
        p
        for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.name)


def _read_image_unchanged(path: Path) -> np.ndarray:
    """Read an image preserving its native dtype and channel count. Internal use only."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise IOError(f"Failed to read image: {path}")
    return image


def _assign_splits(records: list[dict], seed: int) -> None:
    """Assign a ``split`` and ``split_note`` to each record in place.

    Stratified by magnification, roughly 2:1 dev:test per magnification
    group, deterministic given ``seed`` and sorted filename order. A
    magnification with only one image is entirely assigned to dev, with
    the limitation noted on that row.
    """
    by_mag: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_mag[r["magnification"]].append(r)

    for mag in sorted(by_mag):
        group = sorted(by_mag[mag], key=lambda r: r["filename"])
        n = len(group)

        if n == 1:
            group[0]["split"] = "dev"
            group[0]["split_note"] = (
                f"only 1 image at {mag}X; magnification cannot be represented "
                "in both splits, forced to dev"
            )
            continue

        test_n = max(1, round(n / 3))
        test_n = min(test_n, n - 1)  # dev always keeps at least one

        rng = random.Random(seed)
        indices = list(range(n))
        rng.shuffle(indices)
        test_indices = set(indices[:test_n])

        for i, r in enumerate(group):
            r["split"] = "test" if i in test_indices else "dev"
            r["split_note"] = ""


def build_manifest(
    raw_dir: Path = RAW_DIR,
    manifest_path: Path = MANIFEST_PATH,
    config_path: Path = CONFIG_PATH,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Scan ``raw_dir``, parse magnifications, assign splits, and write the manifest.

    Deterministic: re-running with the same raw files and seed produces an
    identical manifest.csv.
    """
    config = load_config(config_path)
    if seed is None:
        seed = config["random_seed"]

    files = scan_raw_directory(raw_dir)
    if not files:
        raise FileNotFoundError(f"No supported image files found in {raw_dir}")

    records = []
    for path in files:
        parsed = parse_filename(path.name)
        image = _read_image_unchanged(path)
        height, width = image.shape[0], image.shape[1]
        records.append(
            {
                "filename": path.name,
                "sample_id": parsed["sample_id"],
                "magnification": parsed["magnification"],
                "height": height,
                "width": width,
            }
        )

    _assign_splits(records, seed=seed)

    df = pd.DataFrame(records, columns=MANIFEST_COLUMNS)
    df = df.sort_values("filename").reset_index(drop=True)

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)

    print(f"config hash: {config_hash(config_path)}  seed: {seed}")
    print(f"manifest written: {manifest_path}  ({len(df)} rows)")
    print("magnification distribution by split:")
    print(
        df.pivot_table(
            index="magnification", columns="split", values="filename", aggfunc="count"
        )
        .fillna(0)
        .astype(int)
    )

    return df


def load_manifest(manifest_path: Path = MANIFEST_PATH) -> pd.DataFrame:
    """Load the manifest CSV written by build_manifest()."""
    return pd.read_csv(manifest_path)


def _assert_test_access_allowed(split: str) -> None:
    """Raise unless the calling module is src/evaluate.py, when split == 'test'."""
    if split != "test":
        return
    caller_frame = inspect.stack()[2]
    caller_file = Path(caller_frame.filename).name
    if caller_file != "evaluate.py":
        raise PermissionError(
            f"Refusing to load a test-split image from {caller_file!r}; "
            "only src/evaluate.py may access test-split images."
        )


def load_image(
    filename: str,
    manifest: pd.DataFrame,
    raw_dir: Path = RAW_DIR,
) -> np.ndarray:
    """Load a raw image by filename, preserving its native dtype.

    Looks up the split from ``manifest`` (not a caller-supplied argument, so
    it cannot be spoofed) and refuses to load a test-split image unless
    called from src/evaluate.py.
    """
    rows = manifest.loc[manifest["filename"] == filename]
    if rows.empty:
        raise KeyError(f"{filename!r} not found in manifest")
    split = rows.iloc[0]["split"]
    _assert_test_access_allowed(split)
    return _read_image_unchanged(Path(raw_dir) / filename)


def save_image(path: Path, image: np.ndarray) -> None:
    """Save an image, refusing to silently convert dtype.

    JPEG cannot hold 16-bit data; saving a non-uint8 array to a .jpg/.jpeg
    path raises rather than silently truncating.
    """
    path = Path(path)
    if image.dtype != np.uint8 and path.suffix.lower() in (".jpg", ".jpeg"):
        raise ValueError(
            f"Refusing to save a {image.dtype} array to {path} (JPEG is 8-bit "
            "only and would silently truncate); use .png or .tif instead."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise IOError(f"Failed to write image: {path}")


if __name__ == "__main__":
    build_manifest()
