"""Step 4 — quality assessment.

Computes four quality metrics from the Step 3 normalised arrays
(data/interim/normalised/*.npy), each for the whole dev image and for
every 512 px tile, so partial/localised defects are caught. Focus is not
comparable across magnifications and must only ever be compared within a
magnification group (see quality_before.csv's `magnification` column).

Illumination is handled specially: the large-sigma Gaussian fit is
computed once over the whole image (a genuinely global model), and each
tile reports the local level of that same global fit — this is what makes
a broad illumination defect (e.g. a dark band spanning many tiles) visible
in the per-image heatmap, rather than each tile re-fitting its own,
mostly-meaningless local illumination model.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.filters import gaussian
from skimage.measure import shannon_entropy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import io_utils  # noqa: E402
import normalise  # noqa: E402

TILE_SIZE_PX = 512  # fixed, bounded by GPU memory -- not scaled (CLAUDE.md)

QUALITY_CSV_PATH = io_utils.REPO_ROOT / "data" / "outputs" / "quality_before.csv"
HEATMAP_DIR = io_utils.REPO_ROOT / "data" / "outputs" / "quality_heatmaps"

DEFAULT_LAPLACIAN_KSIZE = 3
DEFAULT_ILLUMINATION_SIGMA_AT_100X_PX = 200.0
DEFAULT_NOISE_SIGMA_AT_100X_PX = 1.0
DEFAULT_ILLUMINATION_FLAG_THRESHOLD = 0.10
DEFAULT_CONTRAST_LOW_PERCENTILE = 5
DEFAULT_CONTRAST_HIGH_PERCENTILE = 95


def to_luminance(image: np.ndarray) -> np.ndarray:
    """Collapse an (H, W) or (H, W, 3) float array to an (H, W) luminance array."""
    if image.ndim == 2:
        return image.astype(np.float32)
    return cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY)


def focus_variance_of_laplacian(luminance: np.ndarray, ksize: int) -> float:
    """Sharpness proxy: variance of the Laplacian. Pixel-scale, not comparable across magnifications."""
    # cv2.Laplacian only accepts matching src/dst depths on some builds
    # (float32 src -> CV_64F dst errors), so cast explicitly.
    laplacian = cv2.Laplacian(luminance.astype(np.float64), cv2.CV_64F, ksize=ksize)
    return float(laplacian.var())


def fit_illumination(luminance: np.ndarray, sigma_px: float) -> np.ndarray:
    """Large-sigma Gaussian low-pass fit standing in for the illumination field."""
    return gaussian(luminance, sigma=sigma_px, preserve_range=True)


def illumination_range_fraction(fit: np.ndarray, image_range: float) -> float:
    """Dynamic range of the illumination fit as a fraction of the image's own range."""
    if image_range <= 0:
        return 0.0
    return float((fit.max() - fit.min()) / image_range)


def noise_mad_sigma(luminance: np.ndarray, sigma_px: float) -> float:
    """Robust noise-sigma estimate: MAD of the small-sigma-Gaussian high-pass residual."""
    smooth = gaussian(luminance, sigma=sigma_px, preserve_range=True)
    residual = luminance - smooth
    mad = np.median(np.abs(residual - np.median(residual)))
    return float(mad * 1.4826)  # MAD -> equivalent Gaussian sigma


def contrast_metrics(luminance: np.ndarray, low_percentile: float, high_percentile: float) -> tuple[float, float]:
    """(percentile spread, Shannon entropy) of the luminance distribution."""
    lo, hi = np.percentile(luminance, [low_percentile, high_percentile])
    spread = float(hi - lo)
    entropy = float(shannon_entropy(luminance))
    return spread, entropy


def _local_metrics(
    luminance: np.ndarray,
    laplacian_ksize: int,
    noise_sigma_px: float,
    contrast_low_pct: float,
    contrast_high_pct: float,
) -> dict:
    """The three metrics that are meaningful computed purely within a region (no global context)."""
    spread, entropy = contrast_metrics(luminance, contrast_low_pct, contrast_high_pct)
    return {
        "focus_variance_of_laplacian": focus_variance_of_laplacian(luminance, laplacian_ksize),
        "noise_mad_sigma": noise_mad_sigma(luminance, noise_sigma_px),
        "contrast_spread": spread,
        "entropy": entropy,
    }


def _iter_tiles(height: int, width: int, tile_size: int = TILE_SIZE_PX):
    for tile_row, row0 in enumerate(range(0, height, tile_size)):
        row1 = min(row0 + tile_size, height)
        for tile_col, col0 in enumerate(range(0, width, tile_size)):
            col1 = min(col0 + tile_size, width)
            yield tile_row, tile_col, row0, row1, col0, col1


def _plot_heatmap(
    filename: str, tile_df: pd.DataFrame, heatmap_dir: Path, suffix: str = "_quality_heatmap.png"
) -> Path:
    """Save a 2x2 grid of per-tile metric heatmaps for one image."""
    n_rows = int(tile_df["tile_row"].max()) + 1
    n_cols = int(tile_df["tile_col"].max()) + 1
    metrics = [
        ("focus_variance_of_laplacian", "Focus (Var. of Laplacian)"),
        ("illumination_level", "Illumination level (local fit mean)"),
        ("noise_mad_sigma", "Noise (MAD sigma)"),
        ("contrast_spread", "Contrast (5-95pct spread)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (column, title) in zip(axes.ravel(), metrics):
        grid = np.full((n_rows, n_cols), np.nan)
        for _, r in tile_df.iterrows():
            grid[int(r["tile_row"]), int(r["tile_col"])] = r[column]
        im = ax.imshow(grid, cmap="viridis")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("tile col")
        ax.set_ylabel("tile row")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(filename)
    fig.tight_layout()
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    out_path = heatmap_dir / f"{Path(filename).stem}{suffix}"
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def assess_image_quality(
    filename: str,
    magnification: int,
    image: np.ndarray,
    laplacian_ksize: int,
    illumination_sigma_px: float,
    illumination_flag_threshold: float,
    noise_sigma_px: float,
    contrast_low_pct: float,
    contrast_high_pct: float,
    tile_size: int = TILE_SIZE_PX,
) -> list[dict]:
    """Compute whole-image and per-tile metric rows for one normalised image."""
    luminance = to_luminance(image)
    height, width = luminance.shape
    image_range = float(luminance.max() - luminance.min())

    fit = fit_illumination(luminance, illumination_sigma_px)
    illum_frac = illumination_range_fraction(fit, image_range)

    rows = []
    whole = _local_metrics(luminance, laplacian_ksize, noise_sigma_px, contrast_low_pct, contrast_high_pct)
    whole.update(
        {
            "filename": filename,
            "magnification": magnification,
            "scope": "image",
            "tile_row": -1,
            "tile_col": -1,
            "tile_top_px": 0,
            "tile_left_px": 0,
            "tile_height_px": height,
            "tile_width_px": width,
            "illumination_level": float(fit.mean()),
            "illumination_range_frac": illum_frac,
            "illumination_flag": bool(illum_frac > illumination_flag_threshold),
        }
    )
    rows.append(whole)

    for tile_row, tile_col, row0, row1, col0, col1 in _iter_tiles(height, width, tile_size):
        tile_lum = luminance[row0:row1, col0:col1]
        tile_fit = fit[row0:row1, col0:col1]
        tile_metrics = _local_metrics(tile_lum, laplacian_ksize, noise_sigma_px, contrast_low_pct, contrast_high_pct)
        tile_illum_frac = illumination_range_fraction(tile_fit, image_range)
        tile_metrics.update(
            {
                "filename": filename,
                "magnification": magnification,
                "scope": "tile",
                "tile_row": tile_row,
                "tile_col": tile_col,
                "tile_top_px": row0,
                "tile_left_px": col0,
                "tile_height_px": row1 - row0,
                "tile_width_px": col1 - col0,
                "illumination_level": float(tile_fit.mean()),
                "illumination_range_frac": tile_illum_frac,
                "illumination_flag": bool(tile_illum_frac > illumination_flag_threshold),
            }
        )
        rows.append(tile_metrics)

    return rows


def resolve_quality_params(config: dict) -> dict:
    """Pull the `quality:` config section, filled in with module defaults."""
    quality_cfg = config.get("quality", {})
    return {
        "laplacian_ksize": quality_cfg.get("laplacian_ksize", DEFAULT_LAPLACIAN_KSIZE),
        "illumination_sigma_100x": quality_cfg.get(
            "illumination_sigma_at_100x_px", DEFAULT_ILLUMINATION_SIGMA_AT_100X_PX
        ),
        "illumination_flag_threshold": quality_cfg.get(
            "illumination_flag_threshold", DEFAULT_ILLUMINATION_FLAG_THRESHOLD
        ),
        "noise_sigma_100x": quality_cfg.get("noise_sigma_at_100x_px", DEFAULT_NOISE_SIGMA_AT_100X_PX),
        "contrast_low_pct": quality_cfg.get("contrast_low_percentile", DEFAULT_CONTRAST_LOW_PERCENTILE),
        "contrast_high_pct": quality_cfg.get("contrast_high_percentile", DEFAULT_CONTRAST_HIGH_PERCENTILE),
    }


def assess_quality_from_entries(
    entries: pd.DataFrame,
    image_path_column: str,
    params: dict,
    output_csv_path: Path,
    heatmap_dir: Path,
    tile_size: int = TILE_SIZE_PX,
    heatmap_suffix: str = "_quality_heatmap.png",
) -> pd.DataFrame:
    """Shared core: compute whole-image + per-tile metrics for a set of (filename, magnification, path) rows.

    Used by both assess_quality() (Step 4, normalised dev images) and
    src/gate.py's re-check (Step 6, enhanced dev images) so the exact same
    metric code runs on both, which is what makes the before/after
    comparison meaningful.
    """
    all_rows = []
    for _, entry in entries.sort_values("filename").iterrows():
        filename = entry["filename"]
        magnification = int(entry["magnification"])
        image = np.load(io_utils.REPO_ROOT / entry[image_path_column])

        illumination_sigma_px = io_utils.scale_for_magnification(params["illumination_sigma_100x"], magnification)
        noise_sigma_px = io_utils.scale_for_magnification(params["noise_sigma_100x"], magnification)

        rows = assess_image_quality(
            filename=filename,
            magnification=magnification,
            image=image,
            laplacian_ksize=params["laplacian_ksize"],
            illumination_sigma_px=illumination_sigma_px,
            illumination_flag_threshold=params["illumination_flag_threshold"],
            noise_sigma_px=noise_sigma_px,
            contrast_low_pct=params["contrast_low_pct"],
            contrast_high_pct=params["contrast_high_pct"],
            tile_size=tile_size,
        )
        all_rows.extend(rows)

        whole = rows[0]
        tile_df = pd.DataFrame([r for r in rows if r["scope"] == "tile"])
        heatmap_path = _plot_heatmap(filename, tile_df, heatmap_dir, suffix=heatmap_suffix)

        print(
            f"{filename} ({magnification}X, illum_sigma={illumination_sigma_px:.1f}px, "
            f"noise_sigma={noise_sigma_px:.2f}px): "
            f"focus={whole['focus_variance_of_laplacian']:.2f} "
            f"illum_frac={whole['illumination_range_frac']:.3f}"
            f"{' [ILLUMINATION FLAGGED]' if whole['illumination_flag'] else ''} "
            f"noise_sigma_est={whole['noise_mad_sigma']:.4f} "
            f"contrast_spread={whole['contrast_spread']:.3f} entropy={whole['entropy']:.3f} "
            f"-> {heatmap_path}"
        )

    df = pd.DataFrame(all_rows)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"quality metrics written: {output_csv_path} ({len(df)} rows)")
    print("NOTE: focus_variance_of_laplacian is not comparable across magnifications; group by `magnification`.")

    return df


def assess_quality(
    crop_log_path: Path = normalise.CROP_LOG_PATH,
    config_path: Path = io_utils.CONFIG_PATH,
    output_csv_path: Path = QUALITY_CSV_PATH,
    heatmap_dir: Path = HEATMAP_DIR,
    tile_size: int = TILE_SIZE_PX,
) -> pd.DataFrame:
    """Run quality assessment over every normalised dev image and write quality_before.csv."""
    config = io_utils.load_config(config_path)
    params = resolve_quality_params(config)

    crop_log = pd.read_csv(crop_log_path)
    if crop_log.empty:
        raise ValueError(f"{crop_log_path} is empty; run src/normalise.py (Step 3) first.")

    print(
        f"laplacian_ksize={params['laplacian_ksize']} (fixed, not scaled)  "
        f"illumination_sigma_at_100x_px={params['illumination_sigma_100x']}  "
        f"noise_sigma_at_100x_px={params['noise_sigma_100x']}  "
        f"illumination_flag_threshold={params['illumination_flag_threshold']}  "
        f"config hash={io_utils.config_hash(config_path)}"
    )

    return assess_quality_from_entries(
        crop_log, "output_path", params, output_csv_path, heatmap_dir, tile_size
    )


if __name__ == "__main__":
    assess_quality()
