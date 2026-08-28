"""Step 8 — pretrained edge detector candidate.

Genuine PiDiNet weights are distributed from the original authors' repo
as a manual Google Drive download -- awkward and unreliable to fetch in a
disposable Colab/Kaggle session. This module tries a substitute that
sidesteps that: `controlnet_aux`'s PidiNetDetector, which is a PyTorch
implementation of the *same* PiDiNet architecture with weights hosted on
the Hugging Face Hub (`lllyasviel/Annotators`) and fetched automatically
via `from_pretrained` -- no manual download step. If that package or
download is unavailable in a given environment, this falls back to a
Canny-based soft edge probability map, documented clearly below and
tagged in the model's `name` and `substituted` attribute so the benchmark
report never silently presents it as the real thing.

Either way, the output is already a probability map (no instance-mask
conversion needed, unlike Cellpose/SAM/watershed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import io_utils  # noqa: E402
import quality  # noqa: E402
from models.base import BoundaryModel  # noqa: E402

DEFAULT_CANNY_SIGMA_AT_100X_PX = 1.0
DEFAULT_CANNY_LOW_THRESHOLD_PERCENTILE = 60
DEFAULT_CANNY_HIGH_THRESHOLD_PERCENTILE = 90


class PidiNetBoundaryModel(BoundaryModel):
    """PiDiNet via controlnet_aux (Hugging Face weights) if available, else a documented substitute."""

    name = "pidinet"

    def __init__(self, magnification: float, config_path: Path = io_utils.CONFIG_PATH) -> None:
        self.magnification = magnification
        self.substituted = True
        self.substitution_reason = ""
        self._detector = None

        config = io_utils.load_config(config_path)
        self._pidinet_cfg = config.get("models", {}).get("pidinet", {})

        try:
            from controlnet_aux import PidiNetDetector

            self._detector = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
            self.substituted = False
        except Exception as exc:  # noqa: BLE001 - any failure here means "use the fallback"
            self.substitution_reason = (
                f"controlnet_aux PidiNetDetector unavailable ({type(exc).__name__}: {exc}); "
                "falling back to a Canny-based soft edge probability map. This is NOT PiDiNet "
                "output -- see src/models/pidinet_model.py docstring."
            )

        canny_cfg = self._pidinet_cfg
        self.canny_sigma_100x = canny_cfg.get("canny_sigma_at_100x_px", DEFAULT_CANNY_SIGMA_AT_100X_PX)
        self.canny_low_pct = canny_cfg.get(
            "canny_low_threshold_percentile", DEFAULT_CANNY_LOW_THRESHOLD_PERCENTILE
        )
        self.canny_high_pct = canny_cfg.get(
            "canny_high_threshold_percentile", DEFAULT_CANNY_HIGH_THRESHOLD_PERCENTILE
        )

    def _predict_pidinet(self, tile: np.ndarray) -> np.ndarray:
        from PIL import Image

        import normalise

        rgb = tile if tile.ndim == 3 else np.stack([tile, tile, tile], axis=-1)
        image = Image.fromarray(normalise.to_uint8(rgb))
        result = self._detector(image, safe=True, output_type="np")
        edge_map = np.asarray(result)
        if edge_map.ndim == 3:
            edge_map = edge_map.mean(axis=-1)
        edge_map = edge_map.astype(np.float32)
        max_val = float(edge_map.max())
        if max_val > 1.0:
            edge_map = edge_map / 255.0
        return np.clip(edge_map, 0.0, 1.0).astype(np.float32)

    def _predict_canny_fallback(self, tile: np.ndarray) -> np.ndarray:
        from skimage.feature import canny
        from skimage.filters import gaussian

        luminance = quality.to_luminance(tile)
        sigma_px = io_utils.scale_for_magnification(self.canny_sigma_100x, self.magnification)
        low = float(np.percentile(luminance, self.canny_low_pct))
        high = float(np.percentile(luminance, self.canny_high_pct))

        edges = canny(luminance, sigma=sigma_px, low_threshold=low, high_threshold=high).astype(np.float32)
        # Canny is binary; smooth it into a soft probability-like map so it's
        # visually and numerically comparable to the other models' output.
        soft = gaussian(edges, sigma=1.0, preserve_range=True)
        max_val = float(soft.max())
        if max_val > 0:
            soft = soft / max_val
        return np.clip(soft, 0.0, 1.0).astype(np.float32)

    def predict(self, tile: np.ndarray) -> np.ndarray:
        if self._detector is not None:
            return self._predict_pidinet(tile)
        return self._predict_canny_fallback(tile)
