"""Step 8 — abstract interface every candidate boundary model implements.

Every model returns the SAME representation regardless of how it works
internally (instance segmentation, automatic mask generation, a
pretrained edge net, or classical watershed): a float32 boundary
probability map in [0, 1], same H x W as the input tile. This is what
makes src/benchmark.py able to compare them directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BoundaryModel(ABC):
    """Common interface for all Step 8 candidate boundary models."""

    name: str

    @abstractmethod
    def predict(self, tile: np.ndarray) -> np.ndarray:
        """Return a float32 boundary probability map in [0, 1], same HxW as `tile`.

        Args:
            tile: (H, W) or (H, W, C) array, float32, values in [0, 1]
                (a Step 7 tile, or anything with the same contract).

        Returns:
            (H, W) float32 array, values in [0, 1]. High values indicate
            higher confidence that the pixel sits on a boundary (grain
            boundary or phase interface, unlabelled either way).
        """
        raise NotImplementedError


def validate_boundary_map(tile: np.ndarray, boundary_map: np.ndarray) -> None:
    """Enforce the shared BoundaryModel output contract; raise loudly on violation.

    Used by src/benchmark.py against every model's output so a model that
    silently breaks the contract (wrong shape, wrong dtype, out-of-range
    values) fails immediately rather than producing a comparison figure
    that looks superficially fine.
    """
    expected_shape = tile.shape[:2]
    if boundary_map.shape != expected_shape:
        raise ValueError(
            f"boundary map shape {boundary_map.shape} does not match tile HxW {expected_shape}"
        )
    if boundary_map.dtype != np.float32:
        raise ValueError(f"boundary map dtype must be float32, got {boundary_map.dtype}")
    if boundary_map.min() < 0.0 or boundary_map.max() > 1.0:
        raise ValueError(
            f"boundary map values must be in [0, 1], got range "
            f"[{boundary_map.min():.4f}, {boundary_map.max():.4f}]"
        )


def labels_to_boundary_probability(labels: np.ndarray, sigma_px: float = 1.0) -> np.ndarray:
    """Shared instance-mask -> boundary-probability conversion.

    Used by every instance-segmentation-based model (Cellpose, SAM,
    watershed): region perimeters via skimage.segmentation.find_boundaries,
    then a small Gaussian to turn the binary boundary mask into a smooth
    probability-like map in [0, 1].
    """
    from skimage.filters import gaussian
    from skimage.segmentation import find_boundaries

    binary_boundaries = find_boundaries(labels, mode="thick").astype(np.float32)
    smoothed = gaussian(binary_boundaries, sigma=sigma_px, preserve_range=True)
    max_val = float(smoothed.max())
    if max_val > 0:
        smoothed = smoothed / max_val
    return np.clip(smoothed, 0.0, 1.0).astype(np.float32)
