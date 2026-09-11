"""
src/visualization/registration_overlay.py

Generates Registered Alpha Composite Overlay and Checkerboard visual comparison images.
Reuses existing RegistrationEngine warping and transform conventions.
"""

import logging
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np

from src.matching.transformation import SimilarityTransform2D
from src.registration.register import register_image

logger = logging.getLogger(__name__)


def generate_registration_overlay(
    image_a: np.ndarray,
    image_b: np.ndarray,
    transform: SimilarityTransform2D,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Generate alpha composite overlay and checkerboard comparison images.

    Args:
        image_a: Reference lunar image.
        image_b: Query lunar image.
        transform: Verified SimilarityTransform2D mapping Image A to Image B.
        config: Optional visualization configuration dictionary.

    Returns:
        Tuple of (registration_overlay_rgb, checkerboard_image_rgb).
    """
    config = config or {}
    alpha = float(config.get("overlay_alpha", 0.5))
    include_checkerboard = bool(config.get("include_checkerboard", True))
    max_display_dim = int(config.get("max_display_dim", 1200))

    if image_a is None or image_b is None or image_a.size == 0 or image_b.size == 0:
        raise ValueError("Input images for registration overlay must not be empty.")
    if not isinstance(transform, SimilarityTransform2D):
        raise TypeError(f"Expected SimilarityTransform2D, got {type(transform)}.")

    # 1. Warp Image B into Image A's coordinate frame using existing register_image function
    registered_b, valid_mask, _ = register_image(image_a, image_b, transform)

    # 2. Downsample for memory safety if exceeding max_display_dim
    gray_a = _to_uint8_gray(image_a)
    gray_b = _to_uint8_gray(registered_b)
    mask = _to_uint8_gray(valid_mask)

    gray_a, scale = _downsample_if_needed(gray_a, max_display_dim)
    if scale < 1.0:
        gray_b = cv2.resize(gray_b, (gray_a.shape[1], gray_a.shape[0]), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (gray_a.shape[1], gray_a.shape[0]), interpolation=cv2.INTER_NEAREST)

    h, w = gray_a.shape[:2]

    # 3. Create False-Color Red/Cyan Alignment Overlay
    # Channel R = Image A, Channels G & B = Warped Image B
    overlay_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    overlay_rgb[:, :, 0] = gray_a  # Red channel = Reference Image A
    overlay_rgb[:, :, 1] = np.where(mask > 0, gray_b, 0)  # Green channel = Query Image B
    overlay_rgb[:, :, 2] = np.where(mask > 0, gray_b, 0)  # Blue channel = Query Image B

    # Blend with standard weighted alpha overlay in non-overlap regions
    rgb_a = cv2.cvtColor(gray_a, cv2.COLOR_GRAY2RGB)
    rgb_b = cv2.cvtColor(gray_b, cv2.COLOR_GRAY2RGB)
    standard_alpha = cv2.addWeighted(rgb_a, alpha, rgb_b, 1.0 - alpha, 0)

    # Combine false-color in valid overlap regions, standard alpha outside
    mask_3d = (mask > 0)[:, :, np.newaxis]
    final_overlay = np.where(mask_3d, overlay_rgb, standard_alpha).astype(np.uint8)

    # 4. Generate Checkerboard Comparison Image (if enabled)
    checkerboard_rgb: Optional[np.ndarray] = None
    if include_checkerboard:
        checkerboard_rgb = _generate_checkerboard(gray_a, gray_b, mask, num_squares=8)

    return final_overlay, checkerboard_rgb


def _generate_checkerboard(
    gray_a: np.ndarray,
    gray_b: np.ndarray,
    mask: np.ndarray,
    num_squares: int = 8,
) -> np.ndarray:
    """Generate alternating checkerboard pattern comparing Image A and Warped Image B."""
    h, w = gray_a.shape[:2]
    tile_h = max(1, h // num_squares)
    tile_w = max(1, w // num_squares)

    canvas_gray = gray_a.copy()

    for i in range(num_squares):
        for j in range(num_squares):
            if (i + j) % 2 == 1:
                y1, y2 = i * tile_h, (i + 1) * tile_h if i < num_squares - 1 else h
                x1, x2 = j * tile_w, (j + 1) * tile_w if j < num_squares - 1 else w
                
                # Apply warped Image B inside valid mask region
                tile_mask = mask[y1:y2, x1:x2] > 0
                canvas_gray[y1:y2, x1:x2] = np.where(
                    tile_mask,
                    gray_b[y1:y2, x1:x2],
                    gray_a[y1:y2, x1:x2],
                )

    checker_rgb = cv2.cvtColor(canvas_gray, cv2.COLOR_GRAY2RGB)

    # Draw subtle grid lines demarcation
    for i in range(1, num_squares):
        cv2.line(checker_rgb, (0, i * tile_h), (w, i * tile_h), (255, 255, 0), 1)
        cv2.line(checker_rgb, (i * tile_w, 0), (i * tile_w, h), (255, 255, 0), 1)

    return checker_rgb


def _downsample_if_needed(img: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
    """Downsample image if exceeding max dimension."""
    h, w = img.shape[:2]
    max_hw = max(h, w)
    if max_hw > max_dim and max_dim > 0:
        scale = float(max_dim) / float(max_hw)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale
    return img, 1.0


def _to_uint8_gray(img: np.ndarray) -> np.ndarray:
    """Convert arbitrary input array to uint8 2D grayscale."""
    arr = np.asarray(img)
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
        elif arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            arr = arr[:, :, 0]

    if arr.dtype == np.uint8:
        return arr
    elif np.issubdtype(arr.dtype, np.floating):
        max_val = np.max(arr) if arr.size > 0 else 1.0
        if max_val <= 1.0:
            return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            return np.clip(arr, 0.0, 255.0).astype(np.uint8)
    else:
        return cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
