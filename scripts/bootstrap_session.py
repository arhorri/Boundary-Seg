"""Idempotent session bootstrap for Colab/Kaggle.

Checks, in dependency order, whether the manifest, normalised images,
enhanced images, and tiles are present and complete, and regenerates only
whatever is missing by calling the existing Step 1/3/5/7 functions
directly -- no logic is reimplemented here.

Why "enhanced images" is checked even though it wasn't asked for by name:
src/tiling.py (Step 7) reads data/interim/enhanced/enhance_log.csv (Step
5's output), not the Step 3 normalised images directly. Skipping that
check would make this script fail on a fresh clone, since tiling can't
run without it -- so it's included as a required link in the chain.

Why this is needed at all: git-tracked logs (manifest.csv, crop_log.csv,
enhance_log.csv, tile_log.csv) survive a fresh clone, but the large
intermediate .npy arrays they reference are gitignored and do not -- a
dropped Colab session leaves the logs present but the actual data gone.
This script detects that gap (existence only, not staleness -- if you
change a config parameter that affects an earlier stage, delete that
stage's output and re-run this, or just re-run the specific step's
script) and regenerates only what's actually missing.

data/raw/ itself is out of scope: it's gitignored and this script has no
way to fetch it, so if it's absent, the manifest stage below fails
loudly with a clear message, as intended.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import enhance  # noqa: E402
import io_utils  # noqa: E402
import normalise  # noqa: E402
import tiling  # noqa: E402


def _all_paths_exist(log_path: Path, path_columns: list[str]) -> bool:
    """True if `log_path` exists, is non-empty, and every path it references exists on disk."""
    log_path = Path(log_path)
    if not log_path.exists():
        return False
    log = pd.read_csv(log_path)
    if log.empty:
        return False
    for column in path_columns:
        for value in log[column]:
            candidate = Path(value)
            if not candidate.exists() and not (io_utils.REPO_ROOT / value).exists():
                return False
    return True


def check_manifest(manifest_path: Path = io_utils.MANIFEST_PATH) -> bool:
    return Path(manifest_path).exists()


def check_normalised(crop_log_path: Path = normalise.CROP_LOG_PATH) -> bool:
    return _all_paths_exist(crop_log_path, ["output_path"])


def check_enhanced(enhance_log_path: Path = enhance.ENHANCE_LOG_PATH) -> bool:
    return _all_paths_exist(
        enhance_log_path, ["flat_field_output_path", "denoised_output_path", "clahe_output_path"]
    )


def check_tiles(tile_log_path: Path = tiling.TILE_LOG_PATH) -> bool:
    return _all_paths_exist(tile_log_path, ["tile_path"])


def bootstrap_session(config_path: Path = io_utils.CONFIG_PATH) -> dict[str, str]:
    """Check manifest -> normalised -> enhanced -> tiles, in order, regenerating only what's missing."""
    stages = [
        ("manifest", check_manifest, lambda: io_utils.build_manifest(config_path=config_path)),
        (
            "normalised images",
            check_normalised,
            lambda: normalise.normalise_dev_images(config_path=config_path),
        ),
        ("enhanced images", check_enhanced, lambda: enhance.enhance_dev_images(config_path=config_path)),
        ("tiles", check_tiles, lambda: tiling.extract_dev_tiles(config_path=config_path)),
    ]

    results: dict[str, str] = {}
    for label, check_fn, regenerate_fn in stages:
        if check_fn():
            print(f"[found]      {label}")
            results[label] = "found"
        else:
            print(f"[regenerate] {label} missing or incomplete -- regenerating...")
            regenerate_fn()
            print(f"[done]       {label} regenerated")
            results[label] = "regenerated"

    print("\nbootstrap summary:")
    for label, status in results.items():
        print(f"  {label}: {status}")

    return results


if __name__ == "__main__":
    bootstrap_session()
