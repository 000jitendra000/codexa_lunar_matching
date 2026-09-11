"""
src/visualization/confidence_map.py

Generates spatial Match Confidence / Matching Evidence Maps.
Visualizes spatial distribution of verified matching evidence overlaying lunar image context.
"""

import logging
from typing import List, Dict, Any, Optional
import cv2
import numpy as np

from src.matching.correspondence_fusion import Correspondence

logger = logging.getLogger(__name__)


def generate_confidence_map(
    image: np.ndarray,
    correspondences: List[Correspondence],
    inlier_indices: Optional[List[int]] = None,
    is_target_b: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Generate a Match Confidence / Matching Evidence Map overlaying the lunar image.

    Args:
        image: Grayscale or RGB lunar input image (H, W) or (H, W, C).
        correspondences: List of Correspondence objects from hybrid matcher.
        inlier_indices: Optional indices of verified RANSAC inliers.
        is_target_b: If True, uses point_b coordinates; if False, uses point_a.
        config: Optional visualization configuration dictionary.

    Returns:
        uint8 RGB image array (H_out, W_out, 3) containing evidence heatmap overlay.
    """
    config = config or {}
    sigma = float(config.get("confidence_sigma", 15.0))
    alpha = float(config.get("overlay_alpha", 0.45))
    max_display_dim = int(config.get("max_display_dim", 1200))
    colormap_type = config.get("colormap", "TURBO")

    if image is None or image.size == 0:
        raise ValueError("Input image for confidence map must not be empty.")

    # 1. Normalize base image to 2D uint8
    base_gray = _to_uint8_gray(image)
    orig_h, orig_w = base_gray.shape[:2]

    # 2. Downsample for display memory safety if exceeding max_display_dim
    scale = 1.0
    max_dim = max(orig_h, orig_w)
    if max_dim > max_display_dim and max_display_dim > 0:
        scale = float(max_display_dim) / float(max_dim)
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        base_gray = cv2.resize(base_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        sigma = max(3.0, sigma * scale)

    h, w = base_gray.shape[:2]
    base_rgb = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2RGB)

    # 3. Filter primary matching evidence points (prefer verified RANSAC inliers)
    selected_corrs: List[Correspondence] = []
    if inlier_indices is not None and len(inlier_indices) > 0:
        for idx in inlier_indices:
            if 0 <= idx < len(correspondences):
                selected_corrs.append(correspondences[idx])

    # Fallback to all correspondences if no inliers available
    if not selected_corrs and len(correspondences) > 0:
        selected_corrs = correspondences

    if not selected_corrs:
        # Return base image if no correspondences exist
        return base_rgb

    # 4. Accumulate spatial Gaussian evidence
    evidence_grid = np.zeros((h, w), dtype=np.float32)

    # Kernel radius (3 * sigma)
    k_radius = max(3, int(round(3.0 * sigma)))
    k_size = 2 * k_radius + 1
    
    # Pre-generate 1D Gaussian kernel
    ax = np.arange(-k_radius, k_radius + 1, dtype=np.float32)
    gauss_1d = np.exp(-0.5 * (ax / sigma) ** 2)
    gauss_2d = np.outer(gauss_1d, gauss_1d)

    for corr in selected_corrs:
        pt = corr.point_b if is_target_b else corr.point_a
        cx = int(round(float(pt[0]) * scale))
        cy = int(round(float(pt[1]) * scale))

        if 0 <= cx < w and 0 <= cy < h:
            conf = max(0.1, float(corr.confidence))

            # Bounding box in target image
            x1 = max(0, cx - k_radius)
            x2 = min(w, cx + k_radius + 1)
            y1 = max(0, cy - k_radius)
            y2 = min(h, cy + k_radius + 1)

            # Corresponding bounding box in Gaussian kernel
            kx1 = x1 - (cx - k_radius)
            kx2 = kx1 + (x2 - x1)
            ky1 = y1 - (cy - k_radius)
            ky2 = ky1 + (y2 - y1)

            evidence_grid[y1:y2, x1:x2] += conf * gauss_2d[ky1:ky2, kx1:kx2]

    # 5. Normalize spatial evidence field to [0, 1]
    max_ev = float(np.max(evidence_grid))
    if max_ev > 1e-6:
        evidence_grid = evidence_grid / max_ev

    # 6. Apply colormap and blend
    ev_uint8 = (evidence_grid * 255.0).clip(0, 255).astype(np.uint8)

    cv_cmap = cv2.COLORMAP_TURBO if colormap_type.upper() == "TURBO" else cv2.COLORMAP_JET
    heatmap_bgr = cv2.applyColorMap(ev_uint8, cv_cmap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Mask for low-evidence background preservation
    mask = (evidence_grid > 0.02)[:, :, np.newaxis]

    blended = cv2.addWeighted(base_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)
    output_rgb = np.where(mask, blended, base_rgb)

    return output_rgb.astype(np.uint8)


def _to_uint8_gray(img: np.ndarray) -> np.ndarray:
    """Convert arbitrary image array to uint8 2D grayscale."""
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
