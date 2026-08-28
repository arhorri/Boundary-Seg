"""Step 2 — colour-mode decision.

Decides once, from the dev-split images, whether the pipeline should treat
micrographs as `colour` or `grayscale` going forward. The decision is
written to config/default.yaml; every later stage reads it from there
instead of re-deriving it.

The two-population split used here (Otsu on the value channel) is a
throwaway diagnostic that exists only to give the hue/saturation
comparison two groups of pixels to compare — it is not a segmentation and
nothing downstream may consume the populations themselves. This is the one
place in the pipeline permitted to assume two populations; see CLAUDE.md.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu

sys.path.insert(0, str(Path(__file__).resolve().parent))
import io_utils  # noqa: E402

FIGURE_PATH = io_utils.REPO_ROOT / "data" / "outputs" / "colour_mode_diagnostic.png"

# Cohen's-d-scale "small effect" convention (Cohen, 1988): a standardized
# separation below this is not distinguishable from sensor/quantization
# noise, regardless of how many pixels make the p-value tiny.
DEFAULT_EFFECT_SIZE_THRESHOLD = 0.2


def bgr_uint8_to_hsv_float(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image (as returned by cv2.imread) to floating-point HSV.

    Returns an (H, W, 3) float32 array with H in degrees [0, 360) and
    S, V in [0, 1]. Using float32 throughout avoids OpenCV's 8-bit hue
    quantization (0-179), which would blur small hue differences.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected a 3-channel BGR image, got shape {image.shape}")
    max_val = float(np.iinfo(image.dtype).max)
    bgr_float = image.astype(np.float32) / max_val
    return cv2.cvtColor(bgr_float, cv2.COLOR_BGR2HSV)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized mean difference between two 1-D samples (pooled std)."""
    n_a, n_b = len(a), len(b)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(b) - np.mean(a)) / pooled_std)


def circular_stats(hue_deg: np.ndarray) -> tuple[float, float]:
    """Circular mean (degrees) and circular standard deviation (radians, Mardia)."""
    angles = np.deg2rad(hue_deg)
    c, s = np.mean(np.cos(angles)), np.mean(np.sin(angles))
    mean_angle = np.rad2deg(np.arctan2(s, c)) % 360.0
    resultant_length = np.hypot(c, s)
    circ_std = np.sqrt(-2.0 * np.log(resultant_length)) if resultant_length > 0 else np.inf
    return float(mean_angle), float(circ_std)


def circular_effect_size(hue_a_deg: np.ndarray, hue_b_deg: np.ndarray) -> tuple[float, float]:
    """Circular analogue of Cohen's d for hue (a circular quantity in degrees).

    Hue wraps at 360 degrees, so a plain difference-of-means would treat
    e.g. 2 degrees and 358 degrees as far apart when they are visually
    adjacent reds. This measures the angular separation between the two
    groups' circular means, normalized by their pooled circular spread.

    Returns (effect_size, angular_difference_degrees).
    """
    mean_a, std_a = circular_stats(hue_a_deg)
    mean_b, std_b = circular_stats(hue_b_deg)
    diff_rad = np.angle(np.exp(1j * np.deg2rad(mean_b - mean_a)))
    diff_deg = float(np.rad2deg(diff_rad))
    pooled_std = np.sqrt((std_a**2 + std_b**2) / 2.0)
    if not np.isfinite(pooled_std) or pooled_std == 0:
        return 0.0, diff_deg
    return float(abs(diff_rad) / pooled_std), diff_deg


def analyse_magnification(
    magnification: int,
    filenames: list[str],
    manifest: pd.DataFrame,
    raw_dir: Path,
    effect_size_threshold: float,
) -> dict:
    """Run the colour-mode diagnostic on all dev images at one magnification.

    Pixels from every dev image at this magnification are pooled before the
    Otsu split, so the threshold and effect sizes are computed on the full
    per-magnification sample rather than voted per image.
    """
    h_all, s_all, v_all = [], [], []
    for filename in filenames:
        image = io_utils.load_image(filename, manifest, raw_dir=raw_dir)
        hsv = bgr_uint8_to_hsv_float(image)
        h_all.append(hsv[..., 0].ravel())
        s_all.append(hsv[..., 1].ravel())
        v_all.append(hsv[..., 2].ravel())
    h_all = np.concatenate(h_all)
    s_all = np.concatenate(s_all)
    v_all = np.concatenate(v_all)

    threshold_v = float(threshold_otsu(v_all))
    low_mask = v_all <= threshold_v

    effect_h, diff_h_deg = circular_effect_size(h_all[low_mask], h_all[~low_mask])
    effect_s = cohens_d(s_all[low_mask], s_all[~low_mask])

    decision = "colour" if (effect_h > effect_size_threshold or abs(effect_s) > effect_size_threshold) else "grayscale"

    return {
        "magnification": magnification,
        "filenames": filenames,
        "n_pixels": int(v_all.size),
        "threshold_v": threshold_v,
        "effect_size_h": effect_h,
        "hue_diff_deg": diff_h_deg,
        "effect_size_s": effect_s,
        "decision": decision,
        "h_low": h_all[low_mask],
        "h_high": h_all[~low_mask],
        "s_low": s_all[low_mask],
        "s_high": s_all[~low_mask],
        "v_low": v_all[low_mask],
        "v_high": v_all[~low_mask],
    }


def plot_diagnostic(results: list[dict], figure_path: Path) -> None:
    """Save H/S/V histograms per Otsu population, one row per magnification."""
    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows), squeeze=False)

    channel_specs = [
        ("h_low", "h_high", "Hue (deg)", 0, 360),
        ("s_low", "s_high", "Saturation", 0, 1),
        ("v_low", "v_high", "Value", 0, 1),
    ]

    for row, result in enumerate(results):
        for col, (low_key, high_key, label, lo, hi) in enumerate(channel_specs):
            ax = axes[row][col]
            bins = np.linspace(lo, hi, 60)
            ax.hist(result[low_key], bins=bins, alpha=0.6, density=True, label="low-V population")
            ax.hist(result[high_key], bins=bins, alpha=0.6, density=True, label="high-V population")
            ax.set_xlabel(label)
            ax.set_ylabel("density")
            if col == 0:
                ax.set_ylabel(f"{result['magnification']}X\ndensity")
            if row == 0:
                ax.set_title(label)
            if col == 0 and row == 0:
                ax.legend(fontsize=8)

        axes[row][0].annotate(
            f"effect_size_H={result['effect_size_h']:.3f}  effect_size_S={result['effect_size_s']:.3f}"
            f"  decision={result['decision']}",
            xy=(0.02, 0.95),
            xycoords="axes fraction",
            fontsize=8,
            va="top",
        )

    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)


def _write_colour_mode_config(config_path: Path, decision: str, effect_size_threshold: float) -> None:
    """Insert or replace the `colour_mode:` block in config_path, in place.

    Edits the file as text rather than round-tripping through yaml.safe_dump
    so hand-written comments and key ordering elsewhere in the file survive.
    """
    block = (
        "colour_mode:\n"
        "  # Decided in Step 2 (src/colour_mode.py) from dev-split HSV\n"
        "  # statistics; downstream stages must read this, not re-decide it.\n"
        f"  decision: {decision}\n"
        f"  effect_size_threshold: {effect_size_threshold}\n"
    )
    text = config_path.read_text()
    pattern = re.compile(r"^colour_mode:\n(?:[ \t].*\n?)*", re.MULTILINE)
    if pattern.search(text):
        text = pattern.sub(block, text)
    else:
        if not text.endswith("\n"):
            text += "\n"
        text += "\n" + block
    config_path.write_text(text)


def decide_colour_mode(
    manifest_path: Path = io_utils.MANIFEST_PATH,
    raw_dir: Path = io_utils.RAW_DIR,
    config_path: Path = io_utils.CONFIG_PATH,
    figure_path: Path = FIGURE_PATH,
    effect_size_threshold: Optional[float] = None,
) -> str:
    """Run the colour-mode diagnostic across all dev magnifications.

    Writes `colour_mode.decision` into config_path only if every dev
    magnification agrees; otherwise reports the disagreement and leaves the
    config untouched.

    Returns the decision string, or "disagreement" if magnifications
    disagreed.
    """
    if effect_size_threshold is None:
        effect_size_threshold = DEFAULT_EFFECT_SIZE_THRESHOLD

    manifest = io_utils.load_manifest(manifest_path)
    dev = manifest[manifest["split"] == "dev"]
    if dev.empty:
        raise ValueError("No dev-split images in manifest; run Step 1 first.")

    results = []
    for magnification, group in dev.groupby("magnification"):
        result = analyse_magnification(
            magnification=int(magnification),
            filenames=sorted(group["filename"]),
            manifest=manifest,
            raw_dir=raw_dir,
            effect_size_threshold=effect_size_threshold,
        )
        results.append(result)
        print(
            f"{magnification}X ({result['n_pixels']} pixels, {len(result['filenames'])} images): "
            f"effect_size_H={result['effect_size_h']:.3f} (Δhue={result['hue_diff_deg']:.1f} deg), "
            f"effect_size_S={result['effect_size_s']:.3f}, Otsu V threshold={result['threshold_v']:.3f} "
            f"-> {result['decision']}"
        )

    plot_diagnostic(results, figure_path)
    print(f"diagnostic figure written: {figure_path}")

    decisions = {r["decision"] for r in results}
    if len(decisions) > 1:
        print("DISAGREEMENT across magnifications — not writing a decision:")
        for r in results:
            print(f"  {r['magnification']}X -> {r['decision']}")
        return "disagreement"

    decision = decisions.pop()
    print(f"decision: {decision} (effect_size_threshold={effect_size_threshold})")

    _write_colour_mode_config(config_path, decision, effect_size_threshold)
    print(f"decision written to {config_path}")

    return decision


if __name__ == "__main__":
    decide_colour_mode()
