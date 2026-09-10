"""
src/registration/refinement.py

Sub-pixel / local correspondence refinement module for Milestone B (Registration & Quality Engine).
Refines the spatial locations of verified corresponding points using rotation/scale-aware
local template matching and continuous 2D parabolic peak interpolation.

ARCHITECTURAL RULES:
- Strictly optional (enabled=True/False in config).
- Rotation- and scale-aware: samples patches from Image B using the coarse similarity transform.
- Continuous sub-pixel precision via parabolic peak interpolation over normalized cross-correlation.
- Graceful per-point fallback: if correlation is weak (< min_correlation), near boundary, or
  shift exceeds max_shift_px, the original coordinates are preserved.
- Never discards or aborts the entire registration because an individual point fails refinement.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import copy
import cv2
import numpy as np

from src.matching.correspondence_fusion import Correspondence
from src.matching.transformation import SimilarityTransform2D


@dataclass
class RefinementResult:
    """
    Result container for sub-pixel / local tie-point refinement.

    Attributes:
        refined_points_a: (M, 2) float64 array of coordinates in Image A (unchanged reference).
        refined_points_b: (M, 2) float64 array of refined coordinates in Image B.
        refined_correspondences: List of updated Correspondence objects.
        num_attempted: Total points submitted for refinement.
        num_refined: Points successfully refined.
        num_failed: Points where refinement failed (original coordinates retained).
        mean_shift_px: Mean displacement in pixels among successfully refined points.
        max_shift_px: Maximum displacement in pixels among successfully refined points.
        shifts_px: (M,) array of displacement magnitudes in Image B pixels.
        metadata: Diagnostic details regarding refinement execution.
    """
    refined_points_a: np.ndarray
    refined_points_b: np.ndarray
    refined_correspondences: List[Correspondence]
    num_attempted: int
    num_refined: int
    num_failed: int
    mean_shift_px: float
    max_shift_px: float
    shifts_px: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize refinement result to primitive dictionary."""
        return {
            "num_attempted": int(self.num_attempted),
            "num_refined": int(self.num_refined),
            "num_failed": int(self.num_failed),
            "mean_shift_px": round(float(self.mean_shift_px), 4),
            "max_shift_px": round(float(self.max_shift_px), 4),
            "shifts_px": [round(float(s), 4) for s in self.shifts_px.tolist()],
            "metadata": self.metadata,
        }


def _parabolic_subpixel_offset(
    corr_map: np.ndarray,
    u_max: int,
    v_max: int,
) -> Tuple[float, float]:
    """
    Compute 2D continuous sub-pixel offset from the discrete peak of a correlation map
    using independent parabolic (quadratic) interpolation along each axis.
    """
    h, w = corr_map.shape
    du, dv = 0.0, 0.0
    eps = 1e-7

    # Horizontal sub-pixel offset
    if 0 < u_max < w - 1:
        left = float(corr_map[v_max, u_max - 1])
        center = float(corr_map[v_max, u_max])
        right = float(corr_map[v_max, u_max + 1])
        denom = 2.0 * (2.0 * center - left - right)
        if abs(denom) > eps:
            du = (right - left) / denom
            du = float(np.clip(du, -0.5, 0.5))

    # Vertical sub-pixel offset
    if 0 < v_max < h - 1:
        top = float(corr_map[v_max - 1, u_max])
        center = float(corr_map[v_max, u_max])
        bottom = float(corr_map[v_max + 1, u_max])
        denom = 2.0 * (2.0 * center - top - bottom)
        if abs(denom) > eps:
            dv = (bottom - top) / denom
            dv = float(np.clip(dv, -0.5, 0.5))

    return du, dv


def refine_tie_points(
    image_a: np.ndarray,
    image_b: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
    transform: SimilarityTransform2D,
    correspondences: Optional[List[Correspondence]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> RefinementResult:
    """
    Refine corresponding points between Image A and Image B using scale/rotation-aware
    normalized cross-correlation and sub-pixel peak interpolation.

    Args:
        image_a: Reference image (H_A, W_A) or (H_A, W_A, C).
        image_b: Query image (H_B, W_B) or (H_B, W_B, C).
        points_a: (M, 2) float64 array of coordinates in Image A.
        points_b: (M, 2) float64 array of initial coarse coordinates in Image B.
        transform: SimilarityTransform2D mapping Image A coordinates to Image B.
        correspondences: Optional list of Correspondence objects to update.
        config: Optional configuration dictionary. Defaults to Config.SUBPIXEL_REFINEMENT.

    Returns:
        RefinementResult container with updated coordinates and shift metrics.
    """
    cfg: Dict[str, Any] = {}
    if config is not None:
        cfg.update(config)
    else:
        try:
            from configs.default import Config
            cfg.update(getattr(Config, "SUBPIXEL_REFINEMENT", {}))
        except (ImportError, AttributeError):
            pass

    enabled: bool = bool(cfg.get("enabled", True))
    patch_radius: int = max(3, int(cfg.get("patch_radius", 7)))
    search_radius: int = max(1, int(cfg.get("search_radius", 3)))
    min_correlation: float = float(cfg.get("min_correlation", 0.5))
    max_shift_px: float = float(cfg.get("max_shift_px", 3.0))
    subpixel_interp: bool = bool(cfg.get("subpixel_interpolation", True))

    # Standardize input arrays
    pts_a = np.asarray(points_a, dtype=np.float64).reshape(-1, 2)
    pts_b = np.asarray(points_b, dtype=np.float64).reshape(-1, 2)
    m = len(pts_a)

    if correspondences is not None:
        corrs = [copy.deepcopy(c) for c in correspondences]
    else:
        corrs = [
            Correspondence(
                point_a=pts_a[i],
                point_b=pts_b[i],
                confidence=1.0,
                source="unknown",
                source_index=i,
            )
            for i in range(m)
        ]

    # Handle disabled state or empty input
    if not enabled or m == 0:
        return RefinementResult(
            refined_points_a=pts_a.copy(),
            refined_points_b=pts_b.copy(),
            refined_correspondences=corrs,
            num_attempted=0,
            num_refined=0,
            num_failed=0,
            mean_shift_px=0.0,
            max_shift_px=0.0,
            shifts_px=np.zeros(m, dtype=np.float64),
            metadata={"enabled": False},
        )

    # Convert images to single-channel float32
    if image_a.ndim == 3:
        img_a_gray = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY if image_a.shape[2] == 3 else cv2.COLOR_RGB2GRAY)
    else:
        img_a_gray = image_a
    img_a_f32 = img_a_gray.astype(np.float32)

    if image_b.ndim == 3:
        img_b_gray = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY if image_b.shape[2] == 3 else cv2.COLOR_RGB2GRAY)
    else:
        img_b_gray = image_b
    img_b_f32 = img_b_gray.astype(np.float32)

    h_a, w_a = img_a_f32.shape[:2]
    h_b, w_b = img_b_f32.shape[:2]

    refined_pts_b = pts_b.copy()
    shifts = np.zeros(m, dtype=np.float64)
    num_refined = 0
    num_failed = 0

    patch_size = 2 * patch_radius + 1
    total_search_radius = patch_radius + search_radius
    search_crop_size = 2 * total_search_radius + 1

    scale = float(transform.scale)
    rot_mat = transform.rotation_matrix.astype(np.float64)

    for i in range(m):
        xa, ya = pts_a[i]
        xb, yb = pts_b[i]

        # 1. Boundary check in Image A for reference template
        if (
            xa - patch_radius < 0
            or xa + patch_radius >= w_a
            or ya - patch_radius < 0
            or ya + patch_radius >= h_a
        ):
            num_failed += 1
            continue

        # Extract reference template from Image A using sub-pixel interpolation
        template = cv2.getRectSubPix(
            img_a_f32,
            (patch_size, patch_size),
            (float(xa), float(ya)),
        )

        # Variance check on template
        if float(np.std(template)) < 1e-4:
            num_failed += 1
            continue

        # 2. Extract rotation- and scale-aligned search region from Image B
        # Destination search crop (patch) has center at (total_search_radius, total_search_radius).
        # A source pixel p_B in Image B maps to destination crop coordinates as:
        #   p_patch = center_crop + (s * R)^(-1) * (p_B - [xb, yb]^T)
        #   warp_mat[:2, :2] = (1 / s) * R^T
        #   warp_mat[:2, 2]  = center_crop - (1 / s) * R^T @ [xb, yb]^T
        center_crop = np.array([total_search_radius, total_search_radius], dtype=np.float64)
        m_rot_scale = scale * rot_mat
        inv_m_rot_scale = (1.0 / scale) * rot_mat.T
        t_crop = center_crop - inv_m_rot_scale @ np.array([xb, yb], dtype=np.float64)

        warp_mat = np.zeros((2, 3), dtype=np.float64)
        warp_mat[:2, :2] = inv_m_rot_scale
        warp_mat[:2, 2] = t_crop

        # Warp search region from Image B into Image A's orientation/scale
        search_crop = cv2.warpAffine(
            img_b_f32,
            warp_mat,
            (search_crop_size, search_crop_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

        # Variance check on search crop
        if float(np.std(search_crop)) < 1e-4:
            num_failed += 1
            continue

        # 3. Template matching via Normalized Cross Correlation
        corr_map = cv2.matchTemplate(search_crop, template, cv2.TM_CCOEFF_NORMED)
        # corr_map size: (search_crop_size - patch_size + 1) = (2 * search_radius + 1)
        expected_size = 2 * search_radius + 1
        if corr_map.shape != (expected_size, expected_size):
            num_failed += 1
            continue

        _, max_val, _, max_loc = cv2.minMaxLoc(corr_map)
        u_peak, v_peak = max_loc

        # Reject weak correlation
        if max_val < min_correlation:
            num_failed += 1
            continue

        # Integer displacement relative to nominal center in Image A coordinate frame
        du_int = float(u_peak - search_radius)
        dv_int = float(v_peak - search_radius)

        # Sub-pixel continuous parabolic peak interpolation
        if subpixel_interp:
            sub_u, sub_v = _parabolic_subpixel_offset(corr_map, u_peak, v_peak)
            du = du_int + sub_u
            dv = dv_int + sub_v
        else:
            du = du_int
            dv = dv_int

        # Map displacement into Image B's coordinate space: delta_b = s * R * [du, dv]^T
        disp_b = m_rot_scale @ np.array([du, dv], dtype=np.float64)
        shift_mag = float(np.linalg.norm(disp_b))

        # Check maximum allowed displacement
        if shift_mag > max_shift_px:
            num_failed += 1
            continue

        # Check that refined coordinate remains within Image B bounds
        new_xb = xb + disp_b[0]
        new_yb = yb + disp_b[1]
        if 0 <= new_xb < w_b and 0 <= new_yb < h_b:
            refined_pts_b[i] = [new_xb, new_yb]
            corrs[i].point_b = np.array([new_xb, new_yb], dtype=np.float64)
            corrs[i].metadata["refined"] = True
            corrs[i].metadata["refinement_correlation"] = round(float(max_val), 4)
            corrs[i].metadata["refinement_shift_px"] = round(shift_mag, 4)
            shifts[i] = shift_mag
            num_refined += 1
        else:
            num_failed += 1

    refined_shifts = shifts[shifts > 0.0]
    mean_shift = float(np.mean(refined_shifts)) if len(refined_shifts) > 0 else 0.0
    max_shift = float(np.max(refined_shifts)) if len(refined_shifts) > 0 else 0.0

    metadata = {
        "enabled": True,
        "patch_radius": patch_radius,
        "search_radius": search_radius,
        "min_correlation": min_correlation,
        "max_shift_px": max_shift_px,
        "subpixel_interpolation": subpixel_interp,
    }

    return RefinementResult(
        refined_points_a=pts_a.copy(),
        refined_points_b=refined_pts_b,
        refined_correspondences=corrs,
        num_attempted=m,
        num_refined=num_refined,
        num_failed=num_failed,
        mean_shift_px=round(mean_shift, 4),
        max_shift_px=round(max_shift, 4),
        shifts_px=shifts,
        metadata=metadata,
    )
