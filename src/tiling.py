"""Step 7 — tiling.

Extracts fixed-size, overlapping tiles from every Step 5 enhanced
(CLAHE-final) dev image, at every magnification. Tile size is bounded by
GPU memory, not the specimen -- the one spatial parameter in this whole
pipeline that does NOT go through scale_for_magnification. A 200x tile
simply contains fewer grains than a 100x tile; that is correct and must
not be compensated for (see the grain-count sanity check below, which
reports this rather than hiding it).

Also provides the coordinate inverse (tile-local position -> full-image
position) that Step 9's stitching will need.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import enhance  # noqa: E402
import io_utils  # noqa: E402

OUTPUT_DIR = io_utils.REPO_ROOT / "data" / "interim" / "tiles"
TILE_LOG_PATH = io_utils.REPO_ROOT / "data" / "interim" / "tile_log.csv"

DEFAULT_TILE_SIZE_PX = 512
DEFAULT_OVERLAP_FRACTION = 0.5


def _tile_origins(dimension_px: int, tile_size_px: int, stride_px: int) -> list[int]:
    """Top-left origins along one axis: regular stride, with a final tile flush to the edge.

    Never pads: every returned origin is a valid full-size tile position.
    Raises if the image is smaller than one tile (padding/resizing would
    be required, and both are disallowed).
    """
    if dimension_px < tile_size_px:
        raise ValueError(
            f"dimension {dimension_px}px is smaller than tile_size {tile_size_px}px; "
            "cannot tile without padding or resizing, both disallowed."
        )
    last_origin = dimension_px - tile_size_px
    origins = list(range(0, last_origin + 1, stride_px))
    if origins[-1] != last_origin:
        origins.append(last_origin)  # flush-to-edge tile, not a padded partial tile
    return origins


def extract_tiles(image: np.ndarray, tile_size_px: int, stride_px: int) -> list[dict]:
    """Extract all tiles from `image`, covering it fully with no resizing or padding.

    Returns a list of dicts: tile_row_idx, tile_col_idx, row0, col0, tile
    (always exactly tile_size_px x tile_size_px in its first two axes).
    """
    height, width = image.shape[0], image.shape[1]
    row_origins = _tile_origins(height, tile_size_px, stride_px)
    col_origins = _tile_origins(width, tile_size_px, stride_px)

    tiles = []
    for tile_row_idx, row0 in enumerate(row_origins):
        for tile_col_idx, col0 in enumerate(col_origins):
            tile = image[row0 : row0 + tile_size_px, col0 : col0 + tile_size_px]
            tiles.append(
                {
                    "tile_row_idx": tile_row_idx,
                    "tile_col_idx": tile_col_idx,
                    "row0": row0,
                    "col0": col0,
                    "tile": tile,
                }
            )
    return tiles


def tile_box(row0: int, col0: int, tile_size_px: int) -> tuple[int, int, int, int]:
    """A tile's (row0, row1, col0, col1) placement box in the full image -- the Step 9 inverse."""
    return row0, row0 + tile_size_px, col0, col0 + tile_size_px


def local_to_global(row0: int, col0: int, local_row, local_col):
    """Map a coordinate local to a tile back to the full source image's coordinate space."""
    return row0 + local_row, col0 + local_col


def expected_grain_count_per_tile(tile_size_px: int, grain_diameter_px: float) -> float:
    """Rough geometric sanity check: tile area / expected grain area (circle of grain_diameter_px).

    Not a detection or a count of anything measured -- purely tile_area /
    grain_area from the pipeline's own declared feature scale, to confirm
    tile sizing behaves as expected across magnifications (fewer grains
    per tile at higher magnification is correct, not a defect).
    """
    grain_area_px2 = (np.pi / 4.0) * grain_diameter_px**2
    return (tile_size_px**2) / grain_area_px2


def reconstruct_image_from_tiles(tile_log: pd.DataFrame, source_filename: str) -> np.ndarray:
    """Reassemble a source image from its logged tiles (the round-trip check).

    Overlapping tiles carry identical redundant pixel data (they're crops
    of the same source array), so a simple overwrite-as-you-go placement
    reproduces the original exactly -- no blending needed.
    """
    rows = tile_log[tile_log["source_filename"] == source_filename]
    if rows.empty:
        raise ValueError(f"No tiles logged for {source_filename!r}")

    height = int(rows["source_height_px"].iloc[0])
    width = int(rows["source_width_px"].iloc[0])
    first_tile = np.load(io_utils.REPO_ROOT / rows.iloc[0]["tile_path"])
    canvas = np.zeros((height, width) + first_tile.shape[2:], dtype=first_tile.dtype)

    for _, row in rows.iterrows():
        tile = np.load(io_utils.REPO_ROOT / row["tile_path"])
        row0, col0 = int(row["row0"]), int(row["col0"])
        canvas[row0 : row0 + tile.shape[0], col0 : col0 + tile.shape[1]] = tile

    return canvas


def extract_dev_tiles(
    enhance_log_path: Path = enhance.ENHANCE_LOG_PATH,
    config_path: Path = io_utils.CONFIG_PATH,
    output_dir: Path = OUTPUT_DIR,
    log_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Tile every Step 5 enhanced dev image and write tiles + a coordinate log to disk."""
    config = io_utils.load_config(config_path)
    tiling_cfg = config.get("tiling", {})
    tile_size_px = tiling_cfg.get("tile_size_px", DEFAULT_TILE_SIZE_PX)
    overlap_fraction = tiling_cfg.get("overlap_fraction", DEFAULT_OVERLAP_FRACTION)
    stride_px = int(round(tile_size_px * (1.0 - overlap_fraction)))

    grain_diameter_100x = config.get("enhance", {}).get(
        "grain_diameter_at_100x_px", enhance.DEFAULT_GRAIN_DIAMETER_AT_100X_PX
    )

    enhance_log = pd.read_csv(enhance_log_path)
    if enhance_log.empty:
        raise ValueError(f"{enhance_log_path} is empty; run src/enhance.py (Step 5) first.")

    print(
        f"tile_size_px={tile_size_px} (fixed, not scaled)  overlap_fraction={overlap_fraction}  "
        f"stride_px={stride_px}  config hash={io_utils.config_hash(config_path)}"
    )

    output_dir = Path(output_dir)
    log_rows = []
    for _, entry in enhance_log.sort_values("filename").iterrows():
        filename = entry["filename"]
        magnification = int(entry["magnification"])
        image = np.load(io_utils.REPO_ROOT / entry["clahe_output_path"])
        height, width = image.shape[0], image.shape[1]

        grain_diameter_px = io_utils.scale_for_magnification(grain_diameter_100x, magnification)
        grains_per_tile = expected_grain_count_per_tile(tile_size_px, grain_diameter_px)

        tiles = extract_tiles(image, tile_size_px, stride_px)
        n_rows = max(t["tile_row_idx"] for t in tiles) + 1
        n_cols = max(t["tile_col_idx"] for t in tiles) + 1

        stem = Path(filename).stem
        image_out_dir = output_dir / stem
        image_out_dir.mkdir(parents=True, exist_ok=True)

        for t in tiles:
            tile_filename = f"{stem}_r{t['row0']:05d}_c{t['col0']:05d}.npy"
            tile_path = image_out_dir / tile_filename
            np.save(tile_path, t["tile"])
            try:
                tile_path_str = str(tile_path.relative_to(io_utils.REPO_ROOT))
            except ValueError:
                tile_path_str = str(tile_path)

            log_rows.append(
                {
                    "tile_filename": tile_filename,
                    "source_filename": filename,
                    "magnification": magnification,
                    "tile_row_idx": t["tile_row_idx"],
                    "tile_col_idx": t["tile_col_idx"],
                    "row0": t["row0"],
                    "col0": t["col0"],
                    "tile_size_px": tile_size_px,
                    "source_height_px": height,
                    "source_width_px": width,
                    "tile_path": tile_path_str,
                }
            )

        print(
            f"{filename} ({magnification}X): {len(tiles)} tiles ({n_rows}x{n_cols} grid, "
            f"stride={stride_px}px)  grain_diameter={grain_diameter_px:.1f}px  "
            f"expected grains/tile ~= {grains_per_tile:.0f}  -> {image_out_dir}"
        )

    log_df = pd.DataFrame(log_rows)
    if log_path is None:
        log_path = TILE_LOG_PATH
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(log_path, index=False)
    print(f"tile log written: {log_path} ({len(log_df)} tiles)")

    return log_df


if __name__ == "__main__":
    extract_dev_tiles()
