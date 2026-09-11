"""
src/visualization/correspondence_plot.py

Generates side-by-side correspondence line plot visualizations.
Visually connects correspondences between Image A and Image B with inlier/outlier differentiation.
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
import cv2
import numpy as np

from src.matching.correspondence_fusion import Correspondence

logger = logging.getLogger(__name__)


def plot_correspondences(
    image_a: np.ndarray,
    image_b: np.ndarray,
    correspondences: List[Correspondence],
    inlier_indices: Optional[List[int]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Generate side-by-side correspondence line plot between Image A and Image B.

    Args:
        image_a: Reference lunar image array.
        image_b: Query lunar image array.
        correspondences: List of Correspondence objects.
        inlier_indices: Optional indices of verified RANSAC inliers.
        config: Optional visualization configuration dictionary.

    Returns:
        uint8 RGB image array containing side-by-side visualization with drawn matches.
    """
    config = config or {}
    max_matches = int(config.get("max_visualized_matches", 100))
    include_outliers = bool(config.get("include_outliers", True))
    max_display_dim = int(config.get("max_display_dim", 1200))

    if image_a is None or image_b is None or image_a.size == 0 or image_b.size == 0:
        raise ValueError("Input images for correspondence plot must not be empty.")

    # 1. Normalize and downsample images for memory safety
    gray_a = _to_uint8_gray(image_a)
    gray_b = _to_uint8_gray(image_b)

    scale_a, gray_a = _downsample_if_needed(gray_a, max_display_dim)
    scale_b, gray_b = _downsample_if_needed(gray_b, max_display_dim)

    h_a, w_a = gray_a.shape[:2]
    h_b, w_b = gray_b.shape[:2]

    # Target side-by-side canvas height
    canvas_h = max(h_a, h_b)

    # Pad images vertically to canvas_h if needed
    rgb_a = cv2.cvtColor(_pad_vertical(gray_a, canvas_h), cv2.COLOR_GRAY2RGB)
    rgb_b = cv2.cvtColor(_pad_vertical(gray_b, canvas_h), cv2.COLOR_GRAY2RGB)

    canvas = np.hstack([rgb_a, rgb_b])

    if not correspondences:
        _draw_legend(canvas, inliers_count=0, outliers_count=0)
        return canvas

    # 2. Inlier index set
    inlier_set: Set[int] = set(inlier_indices) if inlier_indices is not None else set()

    # 3. Deterministic match selection: Prioritize inliers, then top confidence outliers
    inliers_list: List[Tuple[int, Correspondence]] = []
    outliers_list: List[Tuple[int, Correspondence]] = []

    for idx, corr in enumerate(correspondences):
        if idx in inlier_set:
            inliers_list.append((idx, corr))
        else:
            outliers_list.append((idx, corr))

    # Sort deterministically by confidence descending, then coordinates
    inliers_list.sort(
        key=lambda x: (-float(x[1].confidence), float(x[1].point_a[0]), float(x[1].point_a[1]))
    )
    outliers_list.sort(
        key=lambda x: (-float(x[1].confidence), float(x[1].point_a[0]), float(x[1].point_a[1]))
    )

    selected_inliers = inliers_list[:max_matches]
    rem_capacity = max_matches - len(selected_inliers)

    selected_outliers = []
    if include_outliers and rem_capacity > 0:
        selected_outliers = outliers_list[:rem_capacity]

    # 4. Draw correspondence lines
    # Draw outliers first (underneath inliers)
    for idx, corr in selected_outliers:
        pt_a = (
            int(round(float(corr.point_a[0]) * scale_a)),
            int(round(float(corr.point_a[1]) * scale_a)),
        )
        pt_b = (
            int(round(float(corr.point_b[0]) * scale_b)) + w_a,
            int(round(float(corr.point_b[1]) * scale_b)),
        )
        cv2.line(canvas, pt_a, pt_b, (200, 60, 60), 1, cv2.LINE_AA)
        cv2.circle(canvas, pt_a, 3, (220, 80, 80), -1)
        cv2.circle(canvas, pt_b, 3, (220, 80, 80), -1)

    # Draw inliers on top
    for idx, corr in selected_inliers:
        pt_a = (
            int(round(float(corr.point_a[0]) * scale_a)),
            int(round(float(corr.point_a[1]) * scale_a)),
        )
        pt_b = (
            int(round(float(corr.point_b[0]) * scale_b)) + w_a,
            int(round(float(corr.point_b[1]) * scale_b)),
        )
        cv2.line(canvas, pt_a, pt_b, (0, 230, 0), 2, cv2.LINE_AA)
        cv2.circle(canvas, pt_a, 4, (0, 255, 0), -1)
        cv2.circle(canvas, pt_b, 4, (0, 255, 0), -1)

    # 5. Draw legend and header titles
    _draw_legend(
        canvas,
        inliers_count=len(inliers_list),
        outliers_count=len(outliers_list),
        w_a=w_a,
    )

    return canvas


def _downsample_if_needed(img: np.ndarray, max_dim: int) -> Tuple[float, np.ndarray]:
    """Downsample image if exceeding max dimension while preserving scale factor."""
    h, w = img.shape[:2]
    max_hw = max(h, w)
    if max_hw > max_dim and max_dim > 0:
        scale = float(max_dim) / float(max_hw)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return scale, resized
    return 1.0, img


def _pad_vertical(img: np.ndarray, target_h: int) -> np.ndarray:
    """Pad image vertically with zeros to match target height."""
    h, w = img.shape[:2]
    if h >= target_h:
        return img
    pad_bottom = target_h - h
    return cv2.copyMakeBorder(img, 0, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)


def _draw_legend(canvas: np.ndarray, inliers_count: int, outliers_count: int, w_a: int = 200):
    """Draw titles and color legend overlay on canvas."""
    # Image headers
    cv2.putText(canvas, "IMAGE A (Reference)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, "IMAGE B (Query)", (w_a + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Legend box
    leg_x = 20
    leg_y = canvas.shape[0] - 70
    leg_w = 260
    leg_h = 60

    if leg_y > 40:
        sub = canvas[leg_y:leg_y+leg_h, leg_x:leg_x+leg_w]
        rect = np.full_like(sub, 30)
        cv2.addWeighted(sub, 0.3, rect, 0.7, 0, dst=sub)

        cv2.line(canvas, (leg_x + 10, leg_y + 20), (leg_x + 40, leg_y + 20), (0, 230, 0), 2)
        cv2.putText(canvas, f"RANSAC Inlier ({inliers_count})", (leg_x + 50, leg_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.line(canvas, (leg_x + 10, leg_y + 42), (leg_x + 40, leg_y + 42), (200, 60, 60), 1)
        cv2.putText(canvas, f"Outlier ({outliers_count})", (leg_x + 50, leg_y + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 100, 100), 1)


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
