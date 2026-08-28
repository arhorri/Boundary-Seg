"""Step 5 — restoration and enhancement.

Applies, strictly in this order, to every Step 3 normalised dev image:

1. Flat-field correction (background estimated via rolling-ball, divided
   out, rescaled). Done first because it changes the local statistics
   every later step depends on.
2. Non-local-means denoising, patch geometry deliberately kept under the
   resolved boundary-width limit so thin grain boundaries survive
   (assert_denoise_preserves_boundaries enforces this).
3. CLAHE, with a kernel coarse relative to grain size so uniform grain
   interiors aren't turned into false texture.

Every spatial parameter is declared at the 100x reference in
config/default.yaml and resolved via io_utils.scale_for_magnification.
Each stage's output is saved independently so the pipeline is resumable
and every stage can be inspected on its own.
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
from skimage.exposure import equalize_adapthist
from skimage.restoration import denoise_nl_means, rolling_ball

sys.path.insert(0, str(Path(__file__).resolve().parent))
import io_utils  # noqa: E402
import normalise  # noqa: E402
import quality  # noqa: E402

OUTPUT_DIR = io_utils.REPO_ROOT / "data" / "interim" / "enhanced"
ENHANCE_LOG_PATH = OUTPUT_DIR / "enhance_log.csv"
FIGURE_DIR = io_utils.REPO_ROOT / "data" / "outputs" / "enhance_stages"

STAGE_NAMES = ["raw", "flat_field", "denoised", "clahe"]

DEFAULT_FLAT_FIELD_RADIUS_AT_100X_PX = 200.0
DEFAULT_FLAT_FIELD_WORKING_RADIUS_PX = 20.0
DEFAULT_DENOISE_PATCH_SIZE_AT_100X_PX = 3.0
DEFAULT_DENOISE_PATCH_DISTANCE_AT_100X_PX = 6.0
DEFAULT_DENOISE_H_MULTIPLIER = 0.8
DEFAULT_CLAHE_KERNEL_SIZE_AT_100X_PX = 200.0
DEFAULT_CLAHE_CLIP_LIMIT = 0.01
DEFAULT_BOUNDARY_WIDTH_AT_100X_PX = 2.0
DEFAULT_GRAIN_DIAMETER_AT_100X_PX = 40.0
DEFAULT_CROP_SIZE_IN_GRAIN_DIAMETERS = 8.0


def _round_odd(value: float, minimum: int = 1) -> int:
    n = int(round(value))
    if n < minimum:
        n = minimum
    if n % 2 == 0:
        n += 1
    return n


def _round_positive_int(value: float, minimum: int = 1) -> int:
    return max(minimum, int(round(value)))


def estimate_background(
    channel: np.ndarray,
    radius_px: float,
    working_radius_px: float = DEFAULT_FLAT_FIELD_WORKING_RADIUS_PX,
) -> np.ndarray:
    """Estimate a slowly-varying background via rolling-ball, at tractable cost.

    A full-resolution rolling-ball at radius_px ~ 200px on an ~8 megapixel
    image is minutes-to-hours (benchmarked). Since the background is by
    definition a low-frequency field, it is estimated on an area-averaged
    downsampled copy (so the working radius stays near `working_radius_px`)
    and resized back up with cubic interpolation. Only this internal model
    is downsampled -- the corrected image returned to the caller stays at
    native resolution.
    """
    height, width = channel.shape
    downsample_factor = max(1, int(round(radius_px / working_radius_px)))

    if downsample_factor > 1:
        small_w = max(1, round(width / downsample_factor))
        small_h = max(1, round(height / downsample_factor))
        small = cv2.resize(channel.astype(np.float32), (small_w, small_h), interpolation=cv2.INTER_AREA)
        small_radius = max(radius_px / downsample_factor, 1.0)
    else:
        small = channel.astype(np.float32)
        small_radius = radius_px

    small_background = rolling_ball(small.astype(np.float64), radius=small_radius, workers=-1).astype(np.float32)

    if downsample_factor > 1:
        background = cv2.resize(small_background, (width, height), interpolation=cv2.INTER_CUBIC)
    else:
        background = small_background

    return background


def flat_field_correct(
    image: np.ndarray,
    radius_px: float,
    working_radius_px: float = DEFAULT_FLAT_FIELD_WORKING_RADIUS_PX,
) -> tuple[np.ndarray, np.ndarray]:
    """Divide out an estimated background, per channel, then rescale to [0, 1].

    Returns (corrected_image, background). Background is stacked to match
    `image`'s shape for easy inspection/saving.
    """
    channels = [image] if image.ndim == 2 else [image[..., c] for c in range(image.shape[2])]

    corrected_channels, background_channels = [], []
    epsilon = 1e-6
    for channel in channels:
        channel = channel.astype(np.float32)
        background = estimate_background(channel, radius_px, working_radius_px)
        corrected = channel / (background + epsilon) * float(background.mean())
        c_min, c_max = float(corrected.min()), float(corrected.max())
        corrected = (corrected - c_min) / (c_max - c_min) if c_max > c_min else np.zeros_like(corrected)
        corrected_channels.append(corrected.astype(np.float32))
        background_channels.append(background)

    if image.ndim == 2:
        return corrected_channels[0], background_channels[0]
    return np.stack(corrected_channels, axis=-1), np.stack(background_channels, axis=-1)


def assert_denoise_preserves_boundaries(
    resolved_patch_size_px: float, magnification: float, boundary_width_at_100x_px: float
) -> float:
    """Fail loudly if the configured NLM patch would blur out the thinnest boundaries.

    Returns the resolved boundary-width limit (pixels) for this magnification.
    """
    boundary_limit_px = io_utils.scale_for_magnification(boundary_width_at_100x_px, magnification)
    patch_radius_px = resolved_patch_size_px / 2.0
    if patch_radius_px > boundary_limit_px:
        raise ValueError(
            f"Denoise patch radius {patch_radius_px:.2f}px exceeds the resolved boundary-width "
            f"limit {boundary_limit_px:.2f}px at {magnification}X; this configuration would blur "
            "out the thinnest grain boundaries. Reduce enhance.denoise_patch_size_at_100x_px."
        )
    return boundary_limit_px


def denoise_image(
    image: np.ndarray,
    patch_size_px: float,
    patch_distance_px: float,
    h_multiplier: float,
    magnification: float,
    boundary_width_at_100x_px: float,
    noise_sigma_px: float,
) -> tuple[np.ndarray, dict]:
    """Non-local-means denoise, with the boundary-preservation guard applied first."""
    boundary_limit_px = assert_denoise_preserves_boundaries(patch_size_px, magnification, boundary_width_at_100x_px)

    patch_size = _round_odd(patch_size_px)
    patch_distance = _round_positive_int(patch_distance_px)

    luminance = quality.to_luminance(image)
    sigma_est = quality.noise_mad_sigma(luminance, sigma_px=noise_sigma_px)
    h = h_multiplier * sigma_est

    channel_axis = -1 if image.ndim == 3 else None
    denoised = denoise_nl_means(
        image.astype(np.float32),
        patch_size=patch_size,
        patch_distance=patch_distance,
        h=h,
        fast_mode=True,
        channel_axis=channel_axis,
    )
    denoised = np.clip(denoised, 0.0, 1.0).astype(np.float32)

    info = {
        "patch_size_px": patch_size,
        "patch_distance_px": patch_distance,
        "boundary_limit_px": boundary_limit_px,
        "sigma_est": sigma_est,
        "h": h,
    }
    return denoised, info


def apply_clahe(image: np.ndarray, kernel_size_px: float, clip_limit: float) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation with a magnification-scaled kernel."""
    kernel_size = _round_positive_int(kernel_size_px)
    equalized = equalize_adapthist(np.clip(image, 0.0, 1.0), kernel_size=kernel_size, clip_limit=clip_limit)
    return np.clip(equalized, 0.0, 1.0).astype(np.float32)


def zoom_crop_box(
    image_shape: tuple[int, int], grain_diameter_px: float, crop_size_in_grain_diameters: float
) -> tuple[int, int, int, int]:
    """A centred crop box sized in grain diameters, so it's comparable across magnifications."""
    height, width = image_shape[0], image_shape[1]
    crop_size = max(16, int(round(grain_diameter_px * crop_size_in_grain_diameters)))
    crop_size = min(crop_size, height, width)
    row0 = max(0, (height - crop_size) // 2)
    col0 = max(0, (width - crop_size) // 2)
    return row0, row0 + crop_size, col0, col0 + crop_size


def crop_image(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    row0, row1, col0, col1 = box
    return image[row0:row1, col0:col1]


def _plot_enhance_stages(
    filename: str,
    magnification: int,
    stages: dict[str, np.ndarray],
    crop_box: tuple[int, int, int, int],
    resolved: dict,
    figure_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for col, name in enumerate(STAGE_NAMES):
        stage_image = stages[name]
        cmap = "gray" if stage_image.ndim == 2 else None

        axes[0][col].imshow(normalise.to_uint8(stage_image), cmap=cmap)
        axes[0][col].set_title(name)
        axes[0][col].axis("off")

        crop = crop_image(stage_image, crop_box)
        axes[1][col].imshow(normalise.to_uint8(crop), cmap=cmap)
        axes[1][col].axis("off")

    crop_px = crop_box[1] - crop_box[0]
    axes[1][0].set_title(f"zoom ({crop_px}px)", fontsize=9)

    param_text = (
        f"{filename}  {magnification}X  |  "
        f"flat_field_radius={resolved['flat_field_radius_px']:.1f}px "
        f"(working_radius={resolved['flat_field_working_radius_px']:.1f}px)  |  "
        f"denoise patch_size={resolved['patch_size_px']}px patch_distance={resolved['patch_distance_px']}px "
        f"h={resolved['h']:.4f} (boundary_limit={resolved['boundary_limit_px']:.2f}px)  |  "
        f"clahe_kernel_size={resolved['clahe_kernel_size_px']}px clip_limit={resolved['clahe_clip_limit']}"
    )
    fig.suptitle(param_text, fontsize=9)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=110)
    plt.close(fig)


def enhance_image(
    filename: str,
    magnification: int,
    image: np.ndarray,
    flat_field_radius_px: float,
    flat_field_working_radius_px: float,
    patch_size_px: float,
    patch_distance_px: float,
    h_multiplier: float,
    boundary_width_at_100x_px: float,
    noise_sigma_px: float,
    clahe_kernel_size_px: float,
    clahe_clip_limit: float,
) -> tuple[dict[str, np.ndarray], dict]:
    """Run the raw -> flat-field -> denoise -> CLAHE pipeline on one image."""
    flat_fielded, background = flat_field_correct(image, flat_field_radius_px, flat_field_working_radius_px)
    denoised, denoise_info = denoise_image(
        flat_fielded,
        patch_size_px=patch_size_px,
        patch_distance_px=patch_distance_px,
        h_multiplier=h_multiplier,
        magnification=magnification,
        boundary_width_at_100x_px=boundary_width_at_100x_px,
        noise_sigma_px=noise_sigma_px,
    )
    clahe_kernel_size = _round_positive_int(clahe_kernel_size_px)
    clahe_result = apply_clahe(denoised, clahe_kernel_size_px, clahe_clip_limit)

    stages = {"raw": image, "flat_field": flat_fielded, "denoised": denoised, "clahe": clahe_result}
    resolved = {
        "flat_field_radius_px": flat_field_radius_px,
        "flat_field_working_radius_px": flat_field_working_radius_px,
        "patch_size_px": denoise_info["patch_size_px"],
        "patch_distance_px": denoise_info["patch_distance_px"],
        "boundary_limit_px": denoise_info["boundary_limit_px"],
        "sigma_est": denoise_info["sigma_est"],
        "h": denoise_info["h"],
        "clahe_kernel_size_px": clahe_kernel_size,
        "clahe_clip_limit": clahe_clip_limit,
    }
    return stages, resolved


def enhance_dev_images(
    crop_log_path: Path = normalise.CROP_LOG_PATH,
    config_path: Path = io_utils.CONFIG_PATH,
    output_dir: Path = OUTPUT_DIR,
    log_path: Optional[Path] = None,
    figure_dir: Path = FIGURE_DIR,
) -> pd.DataFrame:
    """Enhance every normalised dev image and write stage outputs + a log + acceptance figures."""
    config = io_utils.load_config(config_path)
    enhance_cfg = config.get("enhance", {})
    quality_cfg = config.get("quality", {})

    flat_field_radius_100x = enhance_cfg.get("flat_field_radius_at_100x_px", DEFAULT_FLAT_FIELD_RADIUS_AT_100X_PX)
    flat_field_working_radius = enhance_cfg.get(
        "flat_field_working_radius_px", DEFAULT_FLAT_FIELD_WORKING_RADIUS_PX
    )
    patch_size_100x = enhance_cfg.get("denoise_patch_size_at_100x_px", DEFAULT_DENOISE_PATCH_SIZE_AT_100X_PX)
    patch_distance_100x = enhance_cfg.get(
        "denoise_patch_distance_at_100x_px", DEFAULT_DENOISE_PATCH_DISTANCE_AT_100X_PX
    )
    h_multiplier = enhance_cfg.get("denoise_h_multiplier", DEFAULT_DENOISE_H_MULTIPLIER)
    boundary_width_100x = enhance_cfg.get("boundary_width_at_100x_px", DEFAULT_BOUNDARY_WIDTH_AT_100X_PX)
    clahe_kernel_100x = enhance_cfg.get("clahe_kernel_size_at_100x_px", DEFAULT_CLAHE_KERNEL_SIZE_AT_100X_PX)
    clahe_clip_limit = enhance_cfg.get("clahe_clip_limit", DEFAULT_CLAHE_CLIP_LIMIT)
    grain_diameter_100x = enhance_cfg.get("grain_diameter_at_100x_px", DEFAULT_GRAIN_DIAMETER_AT_100X_PX)
    crop_size_in_diameters = enhance_cfg.get(
        "crop_size_in_grain_diameters", DEFAULT_CROP_SIZE_IN_GRAIN_DIAMETERS
    )
    # Reuses Step 4's noise-sigma spatial parameter (same underlying quantity:
    # a small-sigma Gaussian for a high-pass noise residual).
    noise_sigma_100x = quality_cfg.get("noise_sigma_at_100x_px", quality.DEFAULT_NOISE_SIGMA_AT_100X_PX)

    crop_log = pd.read_csv(crop_log_path)
    if crop_log.empty:
        raise ValueError(f"{crop_log_path} is empty; run src/normalise.py (Step 3) first.")

    print(
        f"flat_field_radius_at_100x_px={flat_field_radius_100x}  "
        f"denoise_patch_size_at_100x_px={patch_size_100x}  "
        f"denoise_patch_distance_at_100x_px={patch_distance_100x}  "
        f"boundary_width_at_100x_px={boundary_width_100x}  "
        f"clahe_kernel_size_at_100x_px={clahe_kernel_100x}  "
        f"config hash={io_utils.config_hash(config_path)}"
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = Path(figure_dir)

    log_rows = []
    for _, entry in crop_log.sort_values("filename").iterrows():
        filename = entry["filename"]
        magnification = int(entry["magnification"])
        image = np.load(io_utils.REPO_ROOT / entry["output_path"])

        flat_field_radius_px = io_utils.scale_for_magnification(flat_field_radius_100x, magnification)
        patch_size_px = io_utils.scale_for_magnification(patch_size_100x, magnification)
        patch_distance_px = io_utils.scale_for_magnification(patch_distance_100x, magnification)
        clahe_kernel_size_px = io_utils.scale_for_magnification(clahe_kernel_100x, magnification)
        noise_sigma_px = io_utils.scale_for_magnification(noise_sigma_100x, magnification)
        grain_diameter_px = io_utils.scale_for_magnification(grain_diameter_100x, magnification)

        stages, resolved = enhance_image(
            filename=filename,
            magnification=magnification,
            image=image,
            flat_field_radius_px=flat_field_radius_px,
            flat_field_working_radius_px=flat_field_working_radius,
            patch_size_px=patch_size_px,
            patch_distance_px=patch_distance_px,
            h_multiplier=h_multiplier,
            boundary_width_at_100x_px=boundary_width_100x,
            noise_sigma_px=noise_sigma_px,
            clahe_kernel_size_px=clahe_kernel_size_px,
            clahe_clip_limit=clahe_clip_limit,
        )

        stem = Path(filename).stem
        stage_paths = {}
        for stage_name in ("flat_field", "denoised", "clahe"):
            stage_path = output_dir / f"{stem}_{stage_name}.npy"
            np.save(stage_path, stages[stage_name])
            stage_paths[stage_name] = stage_path

        crop_box = zoom_crop_box(image.shape, grain_diameter_px, crop_size_in_diameters)
        figure_path = figure_dir / f"{stem}_enhance_stages.png"
        _plot_enhance_stages(filename, magnification, stages, crop_box, resolved, figure_path)

        try:
            rel_paths = {k: str(v.relative_to(io_utils.REPO_ROOT)) for k, v in stage_paths.items()}
            figure_path_str = str(figure_path.relative_to(io_utils.REPO_ROOT))
        except ValueError:
            rel_paths = {k: str(v) for k, v in stage_paths.items()}
            figure_path_str = str(figure_path)

        log_rows.append(
            {
                "filename": filename,
                "magnification": magnification,
                "flat_field_radius_px": flat_field_radius_px,
                "flat_field_working_radius_px": flat_field_working_radius,
                "denoise_patch_size_px": resolved["patch_size_px"],
                "denoise_patch_distance_px": resolved["patch_distance_px"],
                "denoise_boundary_limit_px": resolved["boundary_limit_px"],
                "denoise_sigma_est": resolved["sigma_est"],
                "denoise_h": resolved["h"],
                "clahe_kernel_size_px": resolved["clahe_kernel_size_px"],
                "clahe_clip_limit": resolved["clahe_clip_limit"],
                "flat_field_output_path": rel_paths["flat_field"],
                "denoised_output_path": rel_paths["denoised"],
                "clahe_output_path": rel_paths["clahe"],
                "figure_path": figure_path_str,
            }
        )

        print(
            f"{filename} ({magnification}X): flat_field_radius={flat_field_radius_px:.1f}px "
            f"patch_size={resolved['patch_size_px']}px patch_distance={resolved['patch_distance_px']}px "
            f"boundary_limit={resolved['boundary_limit_px']:.2f}px clahe_kernel={resolved['clahe_kernel_size_px']}px "
            f"-> {figure_path}"
        )

    log_df = pd.DataFrame(log_rows)
    if log_path is None:
        log_path = ENHANCE_LOG_PATH
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(log_path, index=False)
    print(f"enhance log written: {log_path}")

    return log_df


if __name__ == "__main__":
    enhance_dev_images()
