"""Step 8 — classical baseline: gradient magnitude -> h-minima suppression -> watershed.

No deep learning. This is a required deliverable, not a formality: if it
wins against the learned models on a given magnification, that is a
legitimate and reportable result (see src/benchmark.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from skimage.filters import gaussian, sobel
from skimage.measure import label
from skimage.morphology import h_minima, remove_small_objects
from skimage.segmentation import watershed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import io_utils  # noqa: E402
import quality  # noqa: E402
from models.base import BoundaryModel, labels_to_boundary_probability  # noqa: E402

DEFAULT_GRADIENT_SMOOTHING_SIGMA_AT_100X_PX = 1.0
DEFAULT_H_MINIMA_DEPTH_AT_100X = 0.02
DEFAULT_MIN_REGION_AREA_IN_GRAIN_AREAS = 0.1


class WatershedBoundaryModel(BoundaryModel):
    """Gradient magnitude -> h-minima marker suppression -> watershed -> boundaries.

    h-minima depth and the minimum-surviving-region area both scale with
    magnification: `h` controls how many local minima merge into a single
    watershed basin, which sets the granularity (effective grain size) of
    the result, so it is scaled like a spatial parameter to keep
    segmentation granularity comparable across magnifications even though
    it is nominally an intensity threshold rather than a pixel length.
    """

    name = "watershed"

    def __init__(
        self,
        magnification: float,
        config_path: Path = io_utils.CONFIG_PATH,
    ) -> None:
        config = io_utils.load_config(config_path)
        watershed_cfg = config.get("models", {}).get("watershed", {})
        grain_diameter_100x = config.get("enhance", {}).get(
            "grain_diameter_at_100x_px", 40.0
        )

        self.magnification = magnification
        self.gradient_smoothing_sigma_px = io_utils.scale_for_magnification(
            watershed_cfg.get(
                "gradient_smoothing_sigma_at_100x_px", DEFAULT_GRADIENT_SMOOTHING_SIGMA_AT_100X_PX
            ),
            magnification,
        )
        self.h_minima_depth = io_utils.scale_for_magnification(
            watershed_cfg.get("h_minima_depth_at_100x", DEFAULT_H_MINIMA_DEPTH_AT_100X),
            magnification,
        )
        grain_diameter_px = io_utils.scale_for_magnification(grain_diameter_100x, magnification)
        grain_area_px2 = (np.pi / 4.0) * grain_diameter_px**2
        min_region_area_fraction = watershed_cfg.get(
            "min_region_area_in_grain_areas", DEFAULT_MIN_REGION_AREA_IN_GRAIN_AREAS
        )
        self.min_region_area_px2 = grain_area_px2 * min_region_area_fraction

    def predict(self, tile: np.ndarray) -> np.ndarray:
        luminance = quality.to_luminance(tile)

        smoothed = gaussian(luminance, sigma=self.gradient_smoothing_sigma_px, preserve_range=True)
        gradient = sobel(smoothed)

        minima_markers = h_minima(gradient, h=self.h_minima_depth)
        markers = label(minima_markers)
        labels = watershed(gradient, markers=markers)
        labels = remove_small_objects(labels, min_size=int(round(self.min_region_area_px2)))

        return labels_to_boundary_probability(labels, sigma_px=1.0)
