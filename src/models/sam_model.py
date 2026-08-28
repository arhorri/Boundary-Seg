"""Step 8 — Segment Anything (SAM) candidate boundary model.

WARNING: this is the slow candidate -- automatic mask generation is
roughly 10-20 seconds per 512x512 tile on a T4, and much slower on CPU.
Budget accordingly in src/benchmark.py (it deliberately runs SAM over
only a handful of tiles, not the full tile set).

Unlike Cellpose, SAM has no notion of an expected object diameter -- it
segments at whatever scale its grid of prompt points happens to land on.
`points_per_side` and `points_per_batch` are therefore the parameters
that matter here and are read from config (models.sam), not scaled by
magnification (they control prompt density/throughput, not a
specimen-scale length).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import io_utils  # noqa: E402
import normalise  # noqa: E402
from models.base import BoundaryModel, labels_to_boundary_probability  # noqa: E402

DEFAULT_MODEL_TYPE = "vit_b"
DEFAULT_POINTS_PER_SIDE = 32
DEFAULT_POINTS_PER_BATCH = 64
DEFAULT_BOUNDARY_GAUSSIAN_SIGMA_PX = 1.0


class SAMBoundaryModel(BoundaryModel):
    name = "sam"

    def __init__(
        self,
        magnification: float,
        config_path: Path = io_utils.CONFIG_PATH,
        device: str | None = None,
    ) -> None:
        config = io_utils.load_config(config_path)
        sam_cfg = config.get("models", {}).get("sam", {})

        # Checked before the (heavy, fragile-in-some-environments) segment_anything
        # import so a missing checkpoint always fails with this specific, actionable
        # error rather than being masked by an unrelated import-time failure.
        checkpoint_path = sam_cfg.get("checkpoint_path")
        if not checkpoint_path:
            raise FileNotFoundError(
                "config models.sam.checkpoint_path is not set; download a SAM checkpoint "
                "(e.g. sam_vit_b_01ec64.pth from Meta's official release) and set its path."
            )

        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        self.magnification = magnification
        self.boundary_gaussian_sigma_px = sam_cfg.get(
            "boundary_gaussian_sigma_px", DEFAULT_BOUNDARY_GAUSSIAN_SIGMA_PX
        )
        model_type = sam_cfg.get("model_type", DEFAULT_MODEL_TYPE)
        points_per_side = sam_cfg.get("points_per_side", DEFAULT_POINTS_PER_SIDE)
        points_per_batch = sam_cfg.get("points_per_batch", DEFAULT_POINTS_PER_BATCH)

        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"

        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        self._generator = SamAutomaticMaskGenerator(
            sam, points_per_side=points_per_side, points_per_batch=points_per_batch
        )

    def predict(self, tile: np.ndarray) -> np.ndarray:
        if tile.ndim == 2:
            image_uint8 = normalise.to_uint8(np.stack([tile, tile, tile], axis=-1))
        else:
            image_uint8 = normalise.to_uint8(tile)

        masks = self._generator.generate(image_uint8)
        labels = np.zeros(image_uint8.shape[:2], dtype=np.int32)
        # Paint largest-first so smaller, more specific masks aren't overwritten by larger ones.
        for i, mask in enumerate(sorted(masks, key=lambda m: m["area"], reverse=True), start=1):
            labels[mask["segmentation"]] = i

        return labels_to_boundary_probability(labels, sigma_px=self.boundary_gaussian_sigma_px)
