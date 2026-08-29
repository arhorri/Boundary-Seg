"""Step 9a — full-tile inference across all four candidate models.

Step 8's benchmark ran all four models on a handful of sample tiles.
This step runs each model against every dev tile in data/interim/tile_log.csv
(~416 tiles), so Step 9b has full coverage to stitch. Test-split tiles are
out of scope here, same as everywhere else in this pipeline: tiling.py
only ever tiles dev-split enhanced images.

Resumable by construction: a tile/model pair is skipped if its output
.npy already exists on disk (the file itself is the source of truth, not
the log), and every tile's output is written immediately after it's
computed -- a mid-run disconnect loses at most the tile in flight. A
per-tile progress log is appended to (not rewritten) after every tile, so
a resumed run can report "X/416 done" without re-deriving it, and so a
crash mid-write loses at most one log row.

SAM is materially slower than the other three, so models run in a fixed
order with SAM last (see DEFAULT_MODEL_ORDER) -- if a session drops
during SAM, the three fast models are already fully done and are skipped
instantly on the next run. run_model_inference() also runs any single
model on its own, so SAM can be resumed independently of the others
without touching them at all.
"""

from __future__ import annotations

import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import io_utils  # noqa: E402
import tiling  # noqa: E402
from benchmark import MODEL_CONSTRUCTORS  # noqa: E402
from models.base import validate_boundary_map  # noqa: E402

PREDICTIONS_DIR = io_utils.REPO_ROOT / "data" / "interim" / "predictions"
INFERENCE_LOG_PATH = PREDICTIONS_DIR / "inference_log.csv"

LOG_FIELDNAMES = [
    "timestamp",
    "model",
    "tile_filename",
    "source_filename",
    "magnification",
    "status",
    "elapsed_seconds",
    "detail",
]

# Fastest-to-slowest (per Step 8's benchmark): SAM last, so a dropped
# session always leaves the three fast models fully complete.
DEFAULT_MODEL_ORDER = ["watershed", "pidinet", "cellpose", "sam"]

MODEL_CLASSES = dict(MODEL_CONSTRUCTORS)

MAX_FAILURES_PRINTED = 10


def prediction_path(model_name: str, source_filename: str, tile_filename: str, predictions_dir: Path = PREDICTIONS_DIR) -> Path:
    """Where one model's prediction for one tile lives on disk."""
    stem = Path(source_filename).stem
    return Path(predictions_dir) / model_name / stem / tile_filename


def _append_log_row(log_path: Path, row: dict) -> None:
    """Append one row to the incremental progress log, writing the header only once."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists() or log_path.stat().st_size == 0
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def run_model_inference(
    model_name: str,
    tile_log_path: Path = tiling.TILE_LOG_PATH,
    config_path: Path = io_utils.CONFIG_PATH,
    predictions_dir: Path = PREDICTIONS_DIR,
    log_path: Path = INFERENCE_LOG_PATH,
) -> dict:
    """Run one model across every dev tile, skipping tiles already predicted.

    A model that fails to construct for a given magnification (missing
    weights/checkpoint/package) is only attempted once per magnification;
    the resulting exception is cached and reapplied to every remaining
    tile at that magnification so a hard-broken model fails fast instead
    of re-attempting a doomed load hundreds of times.
    """
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model {model_name!r}; expected one of {sorted(MODEL_CLASSES)}")

    tile_log = pd.read_csv(tile_log_path)
    if tile_log.empty:
        raise ValueError(f"{tile_log_path} is empty; run src/tiling.py (Step 7) first.")

    predictions_dir = Path(predictions_dir)
    total = len(tile_log)

    to_run = []
    already_done = 0
    for _, row in tile_log.iterrows():
        out_path = prediction_path(model_name, row["source_filename"], row["tile_filename"], predictions_dir)
        if out_path.exists():
            already_done += 1
        else:
            to_run.append(row)

    print(f"[{model_name}] {already_done}/{total} already done, {len(to_run)} remaining")
    if not to_run:
        print(f"[{model_name}] nothing to do")
        return {
            "model": model_name,
            "total": total,
            "already_done": already_done,
            "ran": 0,
            "ok": 0,
            "failed": 0,
            "wall_time_seconds": 0.0,
            "failures": [],
        }

    model_cls = MODEL_CLASSES[model_name]
    models_by_magnification: dict[int, object] = {}
    n_ok = 0
    n_failed = 0
    failures = []
    t_start = time.perf_counter()

    for i, row in enumerate(to_run, start=1):
        magnification = int(row["magnification"])
        if magnification not in models_by_magnification:
            try:
                models_by_magnification[magnification] = model_cls(
                    magnification=magnification, config_path=config_path
                )
            except Exception as exc:  # noqa: BLE001 - cached and reported, not fatal
                models_by_magnification[magnification] = exc
        model_or_exc = models_by_magnification[magnification]

        out_path = prediction_path(model_name, row["source_filename"], row["tile_filename"], predictions_dir)
        t0 = time.perf_counter()
        try:
            if isinstance(model_or_exc, Exception):
                raise model_or_exc
            tile = np.load(io_utils.REPO_ROOT / row["tile_path"])
            boundary_map = model_or_exc.predict(tile)
            validate_boundary_map(tile, boundary_map)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, boundary_map)
            elapsed = time.perf_counter() - t0
            n_ok += 1
            status, detail = "ok", ""
        except Exception as exc:  # noqa: BLE001 - logged and skipped, not fatal
            elapsed = time.perf_counter() - t0
            n_failed += 1
            status, detail = "failed", f"{type(exc).__name__}: {exc}"
            failures.append(
                {
                    "tile_filename": row["tile_filename"],
                    "source_filename": row["source_filename"],
                    "magnification": magnification,
                    "detail": detail,
                }
            )

        _append_log_row(
            log_path,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": model_name,
                "tile_filename": row["tile_filename"],
                "source_filename": row["source_filename"],
                "magnification": magnification,
                "status": status,
                "elapsed_seconds": elapsed,
                "detail": detail,
            },
        )

        if i % 20 == 0 or i == len(to_run):
            print(f"[{model_name}] {already_done + i}/{total} done ({i}/{len(to_run)} this run, {n_failed} failed)")

    wall_time = time.perf_counter() - t_start
    print(f"[{model_name}] this run: {n_ok} ok, {n_failed} failed, wall time {wall_time:.1f}s")
    if failures:
        print(f"[{model_name}] FAILED TILES ({len(failures)}):")
        for f in failures[:MAX_FAILURES_PRINTED]:
            print(f"    {f['source_filename']} / {f['tile_filename']} ({f['magnification']}X): {f['detail']}")
        if len(failures) > MAX_FAILURES_PRINTED:
            print(f"    ... and {len(failures) - MAX_FAILURES_PRINTED} more (see {log_path})")

    return {
        "model": model_name,
        "total": total,
        "already_done": already_done,
        "ran": len(to_run),
        "ok": n_ok,
        "failed": n_failed,
        "wall_time_seconds": wall_time,
        "failures": failures,
    }


def run_all_inference(
    models: Optional[list[str]] = None,
    tile_log_path: Path = tiling.TILE_LOG_PATH,
    config_path: Path = io_utils.CONFIG_PATH,
    predictions_dir: Path = PREDICTIONS_DIR,
    log_path: Path = INFERENCE_LOG_PATH,
) -> dict[str, dict]:
    """Run every model in `models` (default DEFAULT_MODEL_ORDER, SAM last) across all dev tiles."""
    if models is None:
        models = DEFAULT_MODEL_ORDER

    summaries = {}
    for model_name in models:
        summaries[model_name] = run_model_inference(
            model_name, tile_log_path, config_path, predictions_dir, log_path
        )

    print("\n=== inference completion summary ===")
    for model_name, summary in summaries.items():
        present = summary["already_done"] + summary["ok"]
        print(
            f"{model_name}: {present}/{summary['total']} predictions present  "
            f"({summary['ran']} attempted this run: {summary['ok']} ok, {summary['failed']} failed)  "
            f"wall time {summary['wall_time_seconds']:.1f}s"
        )

    return summaries


if __name__ == "__main__":
    run_all_inference()
