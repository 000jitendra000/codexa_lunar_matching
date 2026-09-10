"""
src/preprocessing/pyramid.py

Multi-scale Gaussian image pyramid.

Level 0  → working (input) scale
Level 1  → ½ of level 0
Level 2  → ¼ of level 0
…

The implementation stops automatically before any level drops below
MIN_DIM pixels on the shorter side to avoid useless tiny images.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

MIN_DIM = 16  # Do not generate pyramid levels smaller than this


def build_pyramid(
    image: np.ndarray,
    num_levels: int = 4,
    scale_factor: float = 0.5,
) -> list:
    """
    Build a multi-scale image pyramid.

    Args:
        image:        Input numpy array (H,W) or (H,W,C).
        num_levels:   Total number of levels including level 0 (the original).
        scale_factor: Downscale factor between consecutive levels (default 0.5).

    Returns:
        List of numpy arrays ordered from finest (level 0) to coarsest.
        The list may be shorter than num_levels if the image would become
        too small.

    Raises:
        ValueError: For invalid arguments.
    """
    if image is None:
        raise ValueError("image must not be None.")
    if num_levels < 1:
        raise ValueError(f"num_levels must be ≥ 1, got {num_levels}.")
    if not (0.0 < scale_factor < 1.0):
        raise ValueError(
            f"scale_factor must be in (0, 1), got {scale_factor}."
        )

    pyramid = [image.copy()]  # level 0 — original scale
    current = image

    for lvl in range(1, num_levels):
        h, w = current.shape[:2]
        new_w = max(1, int(round(w * scale_factor)))
        new_h = max(1, int(round(h * scale_factor)))

        if min(new_w, new_h) < MIN_DIM:
            logger.debug(
                "Stopping pyramid at level %d: next size %dx%d < MIN_DIM %d",
                lvl, new_w, new_h, MIN_DIM,
            )
            break

        downscaled = cv2.resize(
            current, (new_w, new_h), interpolation=cv2.INTER_AREA
        )
        pyramid.append(downscaled)
        current = downscaled
        logger.debug("Pyramid level %d: %dx%d", lvl, new_w, new_h)

    return pyramid
