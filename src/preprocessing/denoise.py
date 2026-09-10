"""
src/preprocessing/denoise.py

Optional denoising for lunar imagery.

Conservative defaults are intentional: aggressive denoising removes crater
edges and fine terrain structure that crater detection relies on.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

SUPPORTED_METHODS = ("gaussian", "median", "bilateral", "none")


def denoise(
    image: np.ndarray,
    method: str = "none",
    **kwargs,
) -> np.ndarray:
    """
    Apply an optional denoising operation to a grayscale or colour image.

    Args:
        image:  Input numpy array (H,W) or (H,W,C).
        method: One of 'gaussian', 'median', 'bilateral', or 'none'.
        **kwargs: Method-specific parameters (see below).

    Keyword args per method
    -----------------------
    gaussian:
        ksize (int, default 3) — kernel size (must be positive odd integer)
        sigma (float, default 0) — Gaussian sigma; 0 lets OpenCV choose
    median:
        ksize (int, default 3) — aperture size (positive odd integer ≥ 3)
    bilateral:
        d         (int, default 9)   — diameter of pixel neighbourhood
        sigma_color (float, default 75) — filter sigma in colour space
        sigma_space (float, default 75) — filter sigma in coordinate space

    Returns:
        Denoised numpy array (same dtype as input). Original is NOT modified.

    Raises:
        ValueError: For unknown method or None image.
    """
    if image is None:
        raise ValueError("image must not be None.")
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unknown denoising method '{method}'. "
            f"Supported: {SUPPORTED_METHODS}"
        )

    if method == "none":
        return image.copy()

    if method == "gaussian":
        ksize = int(kwargs.get("ksize", 3))
        sigma = float(kwargs.get("sigma", 0))
        if ksize % 2 == 0:
            raise ValueError(f"gaussian ksize must be odd, got {ksize}.")
        return cv2.GaussianBlur(image, (ksize, ksize), sigma)

    if method == "median":
        ksize = int(kwargs.get("ksize", 3))
        if ksize % 2 == 0:
            raise ValueError(f"median ksize must be odd, got {ksize}.")
        return cv2.medianBlur(image, ksize)

    if method == "bilateral":
        d = int(kwargs.get("d", 9))
        sigma_color = float(kwargs.get("sigma_color", 75))
        sigma_space = float(kwargs.get("sigma_space", 75))
        # bilateral requires uint8 or float32
        if image.dtype not in (np.uint8, np.float32):
            image = image.astype(np.float32)
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
