"""Step 8 — benchmark runner.

Runs all four candidate boundary models (Cellpose, SAM, PiDiNet-or-
substitute, watershed) on the same dev tiles, timed, and produces one
comparison figure plus a timing table -- both broken down by
magnification, since a model can do well at one magnification and poorly
at another (itself a finding worth reporting, not something to average
away).

A model that fails to load (missing weights/checkpoint/package) or fails
to predict on a given tile is reported as such in the timing table and
shown as "FAILED" in the figure, rather than crashing the whole run --
useful because these four candidates have very different dependency
weights (watershed needs nothing beyond this repo's existing stack; SAM
needs a multi-hundred-MB checkpoint most environments won't have staged
by default).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import io_utils  # noqa: E402
import normalise  # noqa: E402
import tiling  # noqa: E402
from models.base import validate_boundary_map  # noqa: E402
from models.cellpose_model import CellposeBoundaryModel  # noqa: E402
from models.pidinet_model import PidiNetBoundaryModel  # noqa: E402
from models.sam_model import SAMBoundaryModel  # noqa: E402
from models.watershed_model import WatershedBoundaryModel  # noqa: E402

TIMING_CSV_PATH = io_utils.REPO_ROOT / "data" / "outputs" / "benchmark_timing.csv"
FIGURE_PATH = io_utils.REPO_ROOT / "data" / "outputs" / "benchmark_comparison.png"

DEFAULT_TILES_PER_MAGNIFICATION = 2

MODEL_CONSTRUCTORS = [
    ("cellpose", CellposeBoundaryModel),
    ("sam", SAMBoundaryModel),
    ("pidinet", PidiNetBoundaryModel),
    ("watershed", WatershedBoundaryModel),
]


def select_benchmark_tiles(tile_log: pd.DataFrame, tiles_per_magnification: int, seed: int) -> pd.DataFrame:
    """Deterministically pick `tiles_per_magnification` tiles from each magnification present."""
    selected = []
    for magnification, group in tile_log.groupby("magnification"):
        group_sorted = group.sort_values("tile_filename").reset_index(drop=True)
        rng = np.random.default_rng(seed + int(magnification))
        n = min(tiles_per_magnification, len(group_sorted))
        idx = np.sort(rng.choice(len(group_sorted), size=n, replace=False))
        selected.append(group_sorted.iloc[idx])
    return pd.concat(selected, ignore_index=True)


def build_models_for_magnification(magnification: int, config_path: Path) -> dict:
    """Construct one instance of every candidate model for this magnification.

    A model whose constructor raises (missing package, missing checkpoint)
    is stored as the exception itself rather than propagating -- the
    caller reports it per-model instead of aborting the whole benchmark.
    """
    models = {}
    for name, cls in MODEL_CONSTRUCTORS:
        try:
            models[name] = cls(magnification=magnification, config_path=config_path)
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any load failure is reportable
            models[name] = exc
    return models


def _plot_comparison(tile_results: list[dict], figure_path: Path) -> None:
    model_names = [name for name, _ in MODEL_CONSTRUCTORS]
    n_tiles = len(tile_results)
    n_cols = 1 + len(model_names)
    fig, axes = plt.subplots(n_tiles, n_cols, figsize=(4 * n_cols, 4 * n_tiles), squeeze=False)

    for row_idx, result in enumerate(tile_results):
        row = result["row"]
        tile = result["tile"]
        cmap_tile = "gray" if tile.ndim == 2 else None
        axes[row_idx][0].imshow(normalise.to_uint8(tile), cmap=cmap_tile)
        axes[row_idx][0].set_title(f"{row['tile_filename']}\n{row['magnification']}X input", fontsize=8)
        axes[row_idx][0].axis("off")

        for col_idx, name in enumerate(model_names, start=1):
            ax = axes[row_idx][col_idx]
            boundary_map = result["maps"][name]
            if boundary_map is None:
                ax.text(0.5, 0.5, "FAILED", ha="center", va="center", fontsize=11, color="red")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.imshow(boundary_map, cmap="magma", vmin=0, vmax=1)
                ax.axis("off")
            if row_idx == 0:
                ax.set_title(name, fontsize=10)

    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=100)
    plt.close(fig)


def run_benchmark(
    tile_log_path: Path = tiling.TILE_LOG_PATH,
    config_path: Path = io_utils.CONFIG_PATH,
    timing_csv_path: Path = TIMING_CSV_PATH,
    figure_path: Path = FIGURE_PATH,
) -> pd.DataFrame:
    """Run every candidate model on the same sample of dev tiles from every magnification."""
    config = io_utils.load_config(config_path)
    benchmark_cfg = config.get("models", {}).get("benchmark", {})
    tiles_per_magnification = benchmark_cfg.get("tiles_per_magnification", DEFAULT_TILES_PER_MAGNIFICATION)
    seed = config.get("random_seed", 42)

    tile_log = pd.read_csv(tile_log_path)
    if tile_log.empty:
        raise ValueError(f"{tile_log_path} is empty; run src/tiling.py (Step 7) first.")

    selected_tiles = select_benchmark_tiles(tile_log, tiles_per_magnification, seed)
    magnifications = sorted(int(m) for m in selected_tiles["magnification"].unique())
    print(f"selected {len(selected_tiles)} tiles across magnifications {magnifications}")

    models_by_magnification = {mag: build_models_for_magnification(mag, config_path) for mag in magnifications}
    for mag, models in models_by_magnification.items():
        for name, m in models.items():
            status = "ready" if not isinstance(m, Exception) else f"FAILED TO LOAD ({type(m).__name__}): {m}"
            print(f"  {mag}X {name}: {status}")

    timing_rows = []
    tile_results = []

    for _, row in selected_tiles.iterrows():
        magnification = int(row["magnification"])
        tile = np.load(io_utils.REPO_ROOT / row["tile_path"])
        models = models_by_magnification[magnification]

        maps_for_tile = {}
        for name, model in models.items():
            if isinstance(model, Exception):
                timing_rows.append(
                    {
                        "tile_filename": row["tile_filename"],
                        "source_filename": row["source_filename"],
                        "magnification": magnification,
                        "model": name,
                        "elapsed_seconds": None,
                        "status": "load_failed",
                        "detail": str(model),
                    }
                )
                maps_for_tile[name] = None
                continue
            try:
                t0 = time.perf_counter()
                boundary_map = model.predict(tile)
                elapsed = time.perf_counter() - t0
                validate_boundary_map(tile, boundary_map)
                timing_rows.append(
                    {
                        "tile_filename": row["tile_filename"],
                        "source_filename": row["source_filename"],
                        "magnification": magnification,
                        "model": name,
                        "elapsed_seconds": elapsed,
                        "status": "ok",
                        "detail": "",
                    }
                )
                maps_for_tile[name] = boundary_map
            except Exception as exc:  # noqa: BLE001 - reportable per-model, not fatal
                timing_rows.append(
                    {
                        "tile_filename": row["tile_filename"],
                        "source_filename": row["source_filename"],
                        "magnification": magnification,
                        "model": name,
                        "elapsed_seconds": None,
                        "status": "predict_failed",
                        "detail": str(exc),
                    }
                )
                maps_for_tile[name] = None

        tile_results.append({"row": row, "tile": tile, "maps": maps_for_tile})
        status_line = "  ".join(f"{n}={'ok' if maps_for_tile[n] is not None else 'FAIL'}" for n in maps_for_tile)
        print(f"{row['tile_filename']} ({magnification}X): {status_line}")

    timing_df = pd.DataFrame(timing_rows)
    timing_csv_path.parent.mkdir(parents=True, exist_ok=True)
    timing_df.to_csv(timing_csv_path, index=False)
    print(f"timing table written: {timing_csv_path}")

    ok = timing_df[timing_df["status"] == "ok"]
    if not ok.empty:
        print("\ntiming summary (mean seconds, by magnification x model):")
        print(ok.pivot_table(index="magnification", columns="model", values="elapsed_seconds", aggfunc="mean"))

    _plot_comparison(tile_results, figure_path)
    print(f"comparison figure written: {figure_path}")

    return timing_df


if __name__ == "__main__":
    run_benchmark()
