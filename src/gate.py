"""Step 6 — QC re-check gate.

Re-runs the exact Step 4 metric code (src/quality.py) on the Step 5
enhanced (CLAHE-final) dev images, writes data/outputs/quality_after.csv,
and gates each image pass/fail against its OWN before-enhancement numbers
from data/outputs/quality_before.csv.

Every threshold is a relative change (before -> after), never an absolute
value: illumination non-uniformity must have decreased substantially,
noise must not have increased, focus must not have decreased. Relative
thresholds are what keep the gate valid across magnifications without a
separate threshold set per magnification -- see config/default.yaml's
`gate:` section for why none of these three needs an absolute,
magnification-keyed fallback.

This module does not auto-tune anything: it reports pass/fail against
whatever is in config/default.yaml. A human reads the verdicts, edits the
config, and re-runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import enhance  # noqa: E402
import io_utils  # noqa: E402
import quality  # noqa: E402

QUALITY_AFTER_CSV_PATH = io_utils.REPO_ROOT / "data" / "outputs" / "quality_after.csv"
GATE_COMPARISON_CSV_PATH = io_utils.REPO_ROOT / "data" / "outputs" / "quality_gate.csv"
HEATMAP_AFTER_DIR = io_utils.REPO_ROOT / "data" / "outputs" / "quality_heatmaps_after"

DEFAULT_ILLUMINATION_RELATIVE_IMPROVEMENT_MIN = 0.5
DEFAULT_NOISE_RELATIVE_INCREASE_MAX = 0.0
DEFAULT_FOCUS_RELATIVE_DECREASE_MAX = 0.0


def illumination_relative_decrease(before: float, after: float) -> float:
    """Fraction by which illumination_range_frac decreased; positive = improved."""
    if before == 0:
        return 0.0 if after <= 0 else -float("inf")
    return (before - after) / before


def noise_relative_increase(before: float, after: float) -> float:
    """Fraction by which noise_mad_sigma increased; positive = worse."""
    if before == 0:
        return 0.0 if after <= 0 else float("inf")
    return (after - before) / before


def focus_relative_decrease(before: float, after: float) -> float:
    """Fraction by which focus_variance_of_laplacian decreased; positive = worse."""
    if before == 0:
        return 0.0 if after >= 0 else float("inf")
    return (before - after) / before


def gate_image(before_row: pd.Series, after_row: pd.Series, gate_cfg: dict) -> dict:
    """Compare one image's before/after whole-image metrics against the gate thresholds."""
    illum_decrease = illumination_relative_decrease(
        before_row["illumination_range_frac"], after_row["illumination_range_frac"]
    )
    noise_increase = noise_relative_increase(before_row["noise_mad_sigma"], after_row["noise_mad_sigma"])
    focus_decrease = focus_relative_decrease(
        before_row["focus_variance_of_laplacian"], after_row["focus_variance_of_laplacian"]
    )

    illum_pass = illum_decrease >= gate_cfg["illumination_relative_improvement_min"]
    noise_pass = noise_increase <= gate_cfg["noise_relative_increase_max"]
    focus_pass = focus_decrease <= gate_cfg["focus_relative_decrease_max"]

    return {
        "filename": before_row["filename"],
        "magnification": int(before_row["magnification"]),
        "illumination_range_frac_before": before_row["illumination_range_frac"],
        "illumination_range_frac_after": after_row["illumination_range_frac"],
        "illumination_relative_decrease": illum_decrease,
        "illumination_pass": illum_pass,
        "noise_mad_sigma_before": before_row["noise_mad_sigma"],
        "noise_mad_sigma_after": after_row["noise_mad_sigma"],
        "noise_relative_increase": noise_increase,
        "noise_pass": noise_pass,
        "focus_variance_of_laplacian_before": before_row["focus_variance_of_laplacian"],
        "focus_variance_of_laplacian_after": after_row["focus_variance_of_laplacian"],
        "focus_relative_decrease": focus_decrease,
        "focus_pass": focus_pass,
        "gate_pass": bool(illum_pass and noise_pass and focus_pass),
    }


def run_qc_gate(
    quality_before_csv_path: Path = quality.QUALITY_CSV_PATH,
    enhance_log_path: Path = enhance.ENHANCE_LOG_PATH,
    config_path: Path = io_utils.CONFIG_PATH,
    quality_after_csv_path: Path = QUALITY_AFTER_CSV_PATH,
    gate_comparison_csv_path: Path = GATE_COMPARISON_CSV_PATH,
    heatmap_dir: Path = HEATMAP_AFTER_DIR,
) -> pd.DataFrame:
    """Re-run quality metrics on enhanced images and gate each against its own before numbers."""
    if not quality_before_csv_path.exists():
        raise FileNotFoundError(f"{quality_before_csv_path} not found; run src/quality.py (Step 4) first.")
    if not enhance_log_path.exists():
        raise FileNotFoundError(f"{enhance_log_path} not found; run src/enhance.py (Step 5) first.")

    config = io_utils.load_config(config_path)
    params = quality.resolve_quality_params(config)
    gate_cfg_raw = config.get("gate")
    if gate_cfg_raw is None:
        raise KeyError("config/default.yaml has no `gate:` section; run Step 6 config setup first.")
    gate_cfg = {
        "illumination_relative_improvement_min": gate_cfg_raw.get(
            "illumination_relative_improvement_min", DEFAULT_ILLUMINATION_RELATIVE_IMPROVEMENT_MIN
        ),
        "noise_relative_increase_max": gate_cfg_raw.get(
            "noise_relative_increase_max", DEFAULT_NOISE_RELATIVE_INCREASE_MAX
        ),
        "focus_relative_decrease_max": gate_cfg_raw.get(
            "focus_relative_decrease_max", DEFAULT_FOCUS_RELATIVE_DECREASE_MAX
        ),
    }

    print(
        f"gate thresholds (relative change, before->after): "
        f"illumination_relative_improvement_min={gate_cfg['illumination_relative_improvement_min']}  "
        f"noise_relative_increase_max={gate_cfg['noise_relative_increase_max']}  "
        f"focus_relative_decrease_max={gate_cfg['focus_relative_decrease_max']}  "
        f"config hash={io_utils.config_hash(config_path)}"
    )

    quality_before_df = pd.read_csv(quality_before_csv_path)
    before_whole = quality_before_df[quality_before_df["scope"] == "image"].set_index("filename", drop=False)

    enhance_log = pd.read_csv(enhance_log_path)
    quality_after_df = quality.assess_quality_from_entries(
        enhance_log,
        "clahe_output_path",
        params,
        quality_after_csv_path,
        heatmap_dir,
        heatmap_suffix="_quality_heatmap_after.png",
    )
    after_whole = quality_after_df[quality_after_df["scope"] == "image"].set_index("filename", drop=False)

    missing_before = set(after_whole.index) - set(before_whole.index)
    if missing_before:
        raise ValueError(
            f"No Step 4 quality_before.csv entry for: {sorted(missing_before)}; "
            "before/after images must match 1:1."
        )

    verdict_rows = []
    for filename in sorted(after_whole.index):
        result = gate_image(before_whole.loc[filename], after_whole.loc[filename], gate_cfg)
        verdict_rows.append(result)
        verdict = "PASS" if result["gate_pass"] else "FAIL"
        print(
            f"{filename} ({result['magnification']}X): [{verdict}]  "
            f"illumination_range_frac {result['illumination_range_frac_before']:.3f} -> "
            f"{result['illumination_range_frac_after']:.3f} "
            f"(decrease {result['illumination_relative_decrease']:+.1%}, "
            f"need >= {gate_cfg['illumination_relative_improvement_min']:.0%}) "
            f"[{'ok' if result['illumination_pass'] else 'FAIL'}]  |  "
            f"noise_mad_sigma {result['noise_mad_sigma_before']:.4f} -> {result['noise_mad_sigma_after']:.4f} "
            f"(change {result['noise_relative_increase']:+.1%}, "
            f"need <= {gate_cfg['noise_relative_increase_max']:.0%}) "
            f"[{'ok' if result['noise_pass'] else 'FAIL'}]  |  "
            f"focus {result['focus_variance_of_laplacian_before']:.3f} -> "
            f"{result['focus_variance_of_laplacian_after']:.3f} "
            f"(change {-result['focus_relative_decrease']:+.1%}, "
            f"need >= {-gate_cfg['focus_relative_decrease_max']:.0%}) "
            f"[{'ok' if result['focus_pass'] else 'FAIL'}]"
        )

    comparison_df = pd.DataFrame(verdict_rows)
    gate_comparison_csv_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(gate_comparison_csv_path, index=False)
    print(f"gate comparison table written: {gate_comparison_csv_path}")

    n_pass = int(comparison_df["gate_pass"].sum())
    print(f"gate summary: {n_pass}/{len(comparison_df)} images passed. No auto-tuning -- adjust config and re-run.")

    return comparison_df


if __name__ == "__main__":
    run_qc_gate()
