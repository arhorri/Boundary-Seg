"""Step 3 — format normalisation.

For every dev-split image: detect and crop uniform microscope borders or
burned-in annotation bars, convert to the colour mode decided in Step 2,
and produce a float32 [0, 1] array for downstream processing. Native pixel
size is never touched — no resize, downscale, or resample — because it
carries the magnification information the rest of the pipeline depends on.

Only dev-split images are processed here; CLAUDE.md forbids any code path
outside src/evaluate.py from touching test-split images, and
io_utils.load_image() enforces that.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import io_utils  # noqa: E402

OUTPUT_DIR = io_utils.REPO_ROOT / "data" / "interim" / "normalised"
CROP_LOG_PATH = OUTPUT_DIR / "crop_log.csv"
DIAGNOSTIC_FIGURE_PATH = io_utils.REPO_ROOT / "data" / "outputs" / "normalise_crop_diagnostic.png"

# Fallbacks if config/default.yaml has no `normalise:` section yet.
DEFAULT_BORDER_VARIANCE_RATIO = 0.05
DEFAULT_BORDER_MAX_CROP_FRACTION = 0.15


def _uniform_run_length(variances: np.ndarray, threshold: float, max_run: int) -> int:
    """Count how many leading entries of `variances` are <= threshold, capped at max_run."""
    limit = min(max_run, len(variances))
    run = 0
    for value in variances[:limit]:
        if value <= threshold:
            run += 1
        else:
            break
    return run


def detect_border_crop(
    gray: np.ndarray,
    variance_ratio: float = DEFAULT_BORDER_VARIANCE_RATIO,
    max_crop_fraction: float = DEFAULT_BORDER_MAX_CROP_FRACTION,
) -> dict:
    """Detect uniform border/annotation-bar rows and columns near each edge.

    A row (or column) is treated as border if its intensity variance is at
    or below `variance_ratio` times the image's own median row (or column)
    variance — i.e. the threshold is derived from the image, not a fixed
    absolute value. Scanning from each edge stops at the first row/column
    exceeding the threshold, capped at `max_crop_fraction` of that
    dimension as a safety valve against runaway crops on pathological input.

    Args:
        gray: single-channel image, any numeric dtype.
        variance_ratio: unitless threshold ratio (see module docstring).
        max_crop_fraction: max fraction of height/width croppable per edge.

    Returns:
        dict with pixel counts to crop: "top", "bottom", "left", "right",
        and "capped": True if any edge hit the max_crop_fraction limit.
    """
    height, width = gray.shape
    row_var = gray.var(axis=1)
    col_var = gray.var(axis=0)
    row_threshold = float(np.median(row_var)) * variance_ratio
    col_threshold = float(np.median(col_var)) * variance_ratio
    max_rows = int(height * max_crop_fraction)
    max_cols = int(width * max_crop_fraction)

    top = _uniform_run_length(row_var, row_threshold, max_rows)
    bottom = _uniform_run_length(row_var[::-1], row_threshold, max_rows)
    left = _uniform_run_length(col_var, col_threshold, max_cols)
    right = _uniform_run_length(col_var[::-1], col_threshold, max_cols)

    capped = (
        (max_rows > 0 and (top == max_rows or bottom == max_rows))
        or (max_cols > 0 and (left == max_cols or right == max_cols))
    )

    return {"top": top, "bottom": bottom, "left": left, "right": right, "capped": capped}


def to_uint8(image_float01: np.ndarray) -> np.ndarray:
    """Convert a float32 [0, 1] array to uint8 [0, 255] for saving/viewing."""
    clipped = np.clip(image_float01, 0.0, 1.0)
    return np.round(clipped * 255.0).astype(np.uint8)


def to_uint16(image_float01: np.ndarray) -> np.ndarray:
    """Convert a float32 [0, 1] array to uint16 [0, 65535] for a higher-fidelity save path."""
    clipped = np.clip(image_float01, 0.0, 1.0)
    return np.round(clipped * 65535.0).astype(np.uint16)


def normalise_image(
    filename: str,
    manifest: pd.DataFrame,
    raw_dir: Path,
    colour_decision: str,
    variance_ratio: float,
    max_crop_fraction: float,
) -> dict:
    """Load, border-crop, and colour-convert one dev-split image.

    Returns a dict with the float32 [0, 1] result plus everything needed
    to log the crop.
    """
    image_bgr = io_utils.load_image(filename, manifest, raw_dir=raw_dir)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    crop = detect_border_crop(gray, variance_ratio, max_crop_fraction)
    top, bottom, left, right = crop["top"], crop["bottom"], crop["left"], crop["right"]
    cropped_bgr = image_bgr[top : height - bottom, left : width - right]

    if colour_decision == "grayscale":
        converted = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)
    elif colour_decision == "colour":
        converted = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(
            f"Unexpected colour_mode.decision {colour_decision!r} in config; "
            "expected 'colour' or 'grayscale'. Re-run src/colour_mode.py (Step 2)."
        )

    max_val = float(np.iinfo(converted.dtype).max)
    float_image = converted.astype(np.float32) / max_val

    return {
        "filename": filename,
        "original_height": height,
        "original_width": width,
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "capped": crop["capped"],
        "float_image": float_image,
    }


def _plot_crop_diagnostic(results: list[dict], figure_path: Path) -> None:
    """Save a montage of each processed image with the retained region outlined."""
    n = len(results)
    n_cols = min(n, 2)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)

    for i, result in enumerate(results):
        ax = axes[i // n_cols][i % n_cols]
        preview = result["float_image"]
        cmap = "gray" if preview.ndim == 2 else None
        ax.imshow(to_uint8(preview), cmap=cmap)
        h, w = preview.shape[:2]
        rect = patches.Rectangle((0, 0), w - 1, h - 1, linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        ax.set_title(
            f"{result['filename']}\ncrop T{result['top']} B{result['bottom']} "
            f"L{result['left']} R{result['right']} px"
            + (" [CAPPED]" if result["capped"] else "")
        )
        ax.axis("off")

    for j in range(n, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=100)
    plt.close(fig)


def normalise_dev_images(
    manifest_path: Path = io_utils.MANIFEST_PATH,
    raw_dir: Path = io_utils.RAW_DIR,
    config_path: Path = io_utils.CONFIG_PATH,
    output_dir: Path = OUTPUT_DIR,
    log_path: Optional[Path] = None,
    figure_path: Path = DIAGNOSTIC_FIGURE_PATH,
) -> pd.DataFrame:
    """Normalise every dev-split image and write outputs + a crop log to disk.

    Resumable: each image's .npy is written independently, so a dropped
    session can re-run this and only redo what's missing (existing files
    are simply overwritten deterministically, since detection is a pure
    function of the input image and config).
    """
    config = io_utils.load_config(config_path)
    colour_mode_cfg = config.get("colour_mode", {})
    colour_decision = colour_mode_cfg.get("decision")
    if colour_decision is None:
        raise KeyError(
            "config/default.yaml has no colour_mode.decision; run src/colour_mode.py (Step 2) first."
        )

    normalise_cfg = config.get("normalise", {})
    variance_ratio = normalise_cfg.get("border_variance_ratio", DEFAULT_BORDER_VARIANCE_RATIO)
    max_crop_fraction = normalise_cfg.get("border_max_crop_fraction", DEFAULT_BORDER_MAX_CROP_FRACTION)

    manifest = io_utils.load_manifest(manifest_path)
    dev = manifest[manifest["split"] == "dev"].sort_values("filename")
    if dev.empty:
        raise ValueError("No dev-split images in manifest; run Step 1 first.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"colour_mode.decision={colour_decision!r}  "
        f"border_variance_ratio={variance_ratio}  border_max_crop_fraction={max_crop_fraction}  "
        f"config hash={io_utils.config_hash(config_path)}"
    )

    log_rows = []
    results = []
    for _, row in dev.iterrows():
        result = normalise_image(
            filename=row["filename"],
            manifest=manifest,
            raw_dir=raw_dir,
            colour_decision=colour_decision,
            variance_ratio=variance_ratio,
            max_crop_fraction=max_crop_fraction,
        )
        results.append(result)

        stem = Path(row["filename"]).stem
        out_path = output_dir / f"{stem}.npy"
        np.save(out_path, result["float_image"])

        cropped_h, cropped_w = result["float_image"].shape[:2]
        try:
            output_path_str = str(out_path.relative_to(io_utils.REPO_ROOT))
        except ValueError:
            output_path_str = str(out_path)
        log_rows.append(
            {
                "filename": row["filename"],
                "magnification": row["magnification"],
                "colour_mode": colour_decision,
                "original_height": result["original_height"],
                "original_width": result["original_width"],
                "crop_top_px": result["top"],
                "crop_bottom_px": result["bottom"],
                "crop_left_px": result["left"],
                "crop_right_px": result["right"],
                "crop_capped": result["capped"],
                "output_height": cropped_h,
                "output_width": cropped_w,
                "output_path": output_path_str,
            }
        )
        print(
            f"{row['filename']} ({row['magnification']}X): "
            f"crop top={result['top']} bottom={result['bottom']} left={result['left']} right={result['right']} px"
            f"{' [CAPPED]' if result['capped'] else ''} -> {cropped_h}x{cropped_w} -> {out_path}"
        )

    log_df = pd.DataFrame(log_rows)
    if log_path is None:
        log_path = CROP_LOG_PATH
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(log_path, index=False)
    print(f"crop log written: {log_path}")

    _plot_crop_diagnostic(results, figure_path)
    print(f"diagnostic figure written: {figure_path}")

    return log_df


if __name__ == "__main__":
    normalise_dev_images()
