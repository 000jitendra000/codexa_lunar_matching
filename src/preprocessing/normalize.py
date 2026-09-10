"""
src/preprocessing/normalize.py

Intensity normalization operations for lunar imagery.
Each function returns a NEW array; the original is never modified.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def minmax_normalize(image: np.ndarray, out_range: tuple = (0.0, 1.0)) -> np.ndarray:
    """
    Min-max normalization to [out_min, out_max].

    Handles constant / near-constant images safely (no divide-by-zero).

    Args:
        image:     Input numpy array (any numeric dtype).
        out_range: Desired output range as (min, max). Default (0.0, 1.0).

    Returns:
        float32 numpy array with values in out_range.
    """
    if image is None:
        raise ValueError("image must not be None.")

    out_min, out_max = float(out_range[0]), float(out_range[1])
    img = image.astype(np.float32)

    i_min = float(img.min())
    i_max = float(img.max())

    if i_max - i_min < 1e-8:
        # Constant image — map entire image to out_min
        logger.warning(
            "Image is constant or near-constant (min=%.4f, max=%.4f). "
            "Returning out_min (%.4f) for all pixels.", i_min, i_max, out_min
        )
        return np.full_like(img, out_min, dtype=np.float32)

    normalized = (img - i_min) / (i_max - i_min)      # → [0, 1]
    normalized = normalized * (out_max - out_min) + out_min
    return normalized.astype(np.float32)


# Registry allows future normalization strategies to be added by name
_NORMALIZERS = {
    "minmax": minmax_normalize,
}


def normalize(image: np.ndarray, method: str = "minmax", **kwargs) -> np.ndarray:
    """
    Dispatch normalization by method name.

    Args:
        image:  Input array.
        method: Name of normalization method ('minmax', …).
        **kwargs: Extra args forwarded to the chosen method.

    Returns:
        Normalized float32 array.
    """
    if method not in _NORMALIZERS:
        raise ValueError(
            f"Unknown normalization method '{method}'. "
            f"Available: {list(_NORMALIZERS)}"
        )
    return _NORMALIZERS[method](image, **kwargs)
