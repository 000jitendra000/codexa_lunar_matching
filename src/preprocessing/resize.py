"""
src/preprocessing/resize.py

Scale / resize utilities for lunar imagery.

IMPORTANT DISTINCTION:
- This module handles *pixel-space* resizing only.
- Physical GSD (Ground Sampling Distance) normalization is a separate concern
  and is NOT handled here.  GSD metadata lives in dataset_meta.json and will
  be addressed in a later phase when sensor-aware scale invariance is built.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def resize_image(
    image: np.ndarray,
    target_size: tuple,
    keep_aspect_ratio: bool = True,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """
    Resize an image to target_size (width, height).

    Args:
        image:            Input numpy array (H, W) or (H, W, C).
        target_size:      Desired (width, height) in pixels.
        keep_aspect_ratio: If True, fit inside target_size while preserving
                           the aspect ratio.  The result may be smaller than
                           target_size on one axis.
        interpolation:    OpenCV interpolation flag.
                          INTER_AREA is recommended when shrinking.

    Returns:
        Resized numpy array. Original is NOT modified.

    Raises:
        ValueError: For invalid inputs.
    """
    if image is None:
        raise ValueError("image must not be None.")
    if len(target_size) != 2:
        raise ValueError("target_size must be (width, height).")

    target_w, target_h = int(target_size[0]), int(target_size[1])
    if target_w <= 0 or target_h <= 0:
        raise ValueError(f"target_size must be positive, got ({target_w}, {target_h}).")

    h, w = image.shape[:2]

    if keep_aspect_ratio:
        scale = min(target_w / w, target_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
    else:
        new_w, new_h = target_w, target_h

    if new_w == w and new_h == h:
        return image.copy()

    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    logger.debug("Resized %dx%d → %dx%d", w, h, new_w, new_h)
    return resized


def resize_by_scale_factor(
    image: np.ndarray,
    scale: float,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """
    Resize by a scalar factor (e.g. 0.5 → half size, 2.0 → double).

    Args:
        image:         Input numpy array.
        scale:         Positive float scale factor.
        interpolation: OpenCV interpolation flag.

    Returns:
        Resized numpy array.
    """
    if image is None:
        raise ValueError("image must not be None.")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}.")

    h, w = image.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)
