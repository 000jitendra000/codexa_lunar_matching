"""
src/preprocessing/clahe.py

CLAHE (Contrast Limited Adaptive Histogram Equalization) for lunar imagery.
Operates on uint8 grayscale arrays. Returns a NEW array.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

DEFAULT_CLIP_LIMIT = 2.0
DEFAULT_TILE_GRID_SIZE = (8, 8)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = DEFAULT_CLIP_LIMIT,
    tile_grid_size: tuple = DEFAULT_TILE_GRID_SIZE,
) -> np.ndarray:
    """
    Apply CLAHE to a grayscale image.

    The input is converted to uint8 if needed. The original array is
    never modified — a new array is returned.

    Args:
        image:          2-D (H, W) or 3-D (H, W, 1) numpy array.
        clip_limit:     Threshold for contrast limiting. Higher → more contrast.
        tile_grid_size: (columns, rows) of the grid cells.

    Returns:
        uint8 numpy array of the same shape as the input.

    Raises:
        ValueError: If the image is not grayscale (2D or single-channel).
    """
    if image is None:
        raise ValueError("image must not be None.")

    # Accept (H, W, 1) as well as (H, W)
    if image.ndim == 3:
        if image.shape[2] == 1:
            img = image[:, :, 0]
        else:
            raise ValueError(
                f"CLAHE expects a grayscale (2D or HxWx1) image, "
                f"got shape {image.shape}. Convert to grayscale first."
            )
    elif image.ndim == 2:
        img = image
    else:
        raise ValueError(f"Unsupported image ndim: {image.ndim}")

    # Convert to uint8 for OpenCV CLAHE
    if img.dtype != np.uint8:
        # Normalize to 0-255 range
        i_min, i_max = float(img.min()), float(img.max())
        if i_max - i_min < 1e-8:
            img_u8 = np.zeros_like(img, dtype=np.uint8)
        else:
            img_u8 = ((img - i_min) / (i_max - i_min) * 255).astype(np.uint8)
    else:
        img_u8 = img.copy()

    clahe_obj = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=tuple(tile_grid_size),
    )
    result = clahe_obj.apply(img_u8)

    # Re-expand (H,W,1) shape if original was 3-D
    if image.ndim == 3:
        result = result[:, :, np.newaxis]

    return result
