"""Step 8 — Cellpose candidate boundary model.

Uses Cellpose's newest available generalist pretrained model (zero-shot,
no fine-tuning). `diameter` is the single most important parameter: it is
in pixels and MUST be resolved via scale_for_magnification, reusing
enhance.grain_diameter_at_100x_px (the same physical quantity as every
other grain-size reference in this pipeline, not a second source of
truth). A fixed diameter across magnifications is the most likely way
this model silently produces superficially-reasonable-looking garbage --
Cellpose resizes its internal receptive field around the given diameter,
so an unscaled value is correctly tuned for exactly one magnification and
subtly wrong at every other.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import io_utils  # noqa: E402
import quality  # noqa: E402
from models.base import BoundaryModel, labels_to_boundary_probability  # noqa: E402

DEFAULT_MODEL_TYPE = "cyto3"
DEFAULT_BOUNDARY_GAUSSIAN_SIGMA_PX = 1.0


class CellposeBoundaryModel(BoundaryModel):
    name = "cellpose"

    def __init__(
        self,
        magnification: float,
        config_path: Path = io_utils.CONFIG_PATH,
        gpu: bool | None = None,
    ) -> None:
        from cellpose import models

        config = io_utils.load_config(config_path)
        cellpose_cfg = config.get("models", {}).get("cellpose", {})
        grain_diameter_100x = config.get("enhance", {}).get("grain_diameter_at_100x_px", 40.0)

        self.magnification = magnification
        self.diameter_px = io_utils.scale_for_magnification(grain_diameter_100x, magnification)
        self.boundary_gaussian_sigma_px = cellpose_cfg.get(
            "boundary_gaussian_sigma_px", DEFAULT_BOUNDARY_GAUSSIAN_SIGMA_PX
        )
        model_type = cellpose_cfg.get("model_type", DEFAULT_MODEL_TYPE)

        if gpu is None:
            import torch

            gpu = torch.cuda.is_available()

        self._model = models.CellposeModel(gpu=gpu, model_type=model_type)

    def predict(self, tile: np.ndarray) -> np.ndarray:
        # Grain-boundary structure in these micrographs is a luminance
        # phenomenon, not a colour one, and Cellpose's diameter-based
        # scaling is defined per single channel -- feeding it luminance
        # sidesteps channel-convention ambiguity across Cellpose versions.
        luminance = quality.to_luminance(tile)
        result = self._model.eval(luminance, diameter=self.diameter_px, channels=[0, 0])
        masks = result[0]  # (masks, flows, styles[, diams]) across Cellpose versions
        return labels_to_boundary_probability(masks, sigma_px=self.boundary_gaussian_sigma_px)
