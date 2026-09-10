"""
src/evaluation/ground_truth.py

Ground-truth transformation comparison and error metric computations
for Milestone C (Phase 18).

ARCHITECTURAL RULES:
- Computes canonical geometric discrepancies (scale, rotation with wraparound, Euclidean translation).
- Does NOT compare raw affine matrix elements as primary error metrics.
- Handles angular wraparound cleanly in (-180, 180].
"""

import math
from typing import Dict, Any, Optional, Tuple
from src.matching.transformation import SimilarityTransform2D
from src.evaluation.evaluation_types import FailureReason


def compute_ground_truth_errors(
    t_gt: SimilarityTransform2D,
    t_est: SimilarityTransform2D,
) -> Dict[str, float]:
    """
    Compute discrepancies between estimated and ground-truth SimilarityTransform2D.

    Args:
        t_gt: Known ground-truth SimilarityTransform2D (A -> B).
        t_est: Estimated SimilarityTransform2D (A -> B).

    Returns:
        Dictionary containing:
            - scale_error: Absolute difference |s_est - s_gt|
            - relative_scale_error: Relative difference |s_est - s_gt| / s_gt
            - rotation_error_deg: Shortest angular path on circle in degrees [0, 180]
            - translation_error_px: Euclidean distance between translations in pixels
    """
    if not isinstance(t_gt, SimilarityTransform2D) or not isinstance(t_est, SimilarityTransform2D):
        raise TypeError("Both t_gt and t_est must be SimilarityTransform2D instances.")

    # 1. Scale error
    scale_err = abs(float(t_est.scale) - float(t_gt.scale))
    rel_scale_err = scale_err / max(float(t_gt.scale), 1e-9)

    # 2. Rotation error with strict angular wraparound handling
    deg_diff = (float(t_est.rotation_deg) - float(t_gt.rotation_deg) + 180.0) % 360.0 - 180.0
    rot_err_deg = abs(deg_diff)

    # 3. Euclidean translation error
    dx = float(t_est.translation_x) - float(t_gt.translation_x)
    dy = float(t_est.translation_y) - float(t_gt.translation_y)
    trans_err_px = math.hypot(dx, dy)

    return {
        "scale_error": scale_err,
        "relative_scale_error": rel_scale_err,
        "rotation_error_deg": rot_err_deg,
        "translation_error_px": trans_err_px,
    }


def check_ground_truth_compliance(
    t_gt: SimilarityTransform2D,
    t_est: SimilarityTransform2D,
    tolerances: Optional[Dict[str, float]] = None,
) -> Tuple[bool, Optional[str], Dict[str, float]]:
    """
    Verify whether estimated transformation meets ground-truth tolerance thresholds.

    Args:
        t_gt: Ground-truth transform.
        t_est: Estimated transform.
        tolerances: Dict of tolerance thresholds (max_scale_error, max_rotation_error_deg, max_translation_error_px).

    Returns:
        Tuple of (compliant: bool, failure_reason: Optional[str], errors: Dict[str, float]).
    """
    errors = compute_ground_truth_errors(t_gt, t_est)

    tol = {
        "max_scale_error": 0.05,
        "max_rotation_error_deg": 3.0,
        "max_translation_error_px": 5.0,
    }
    if tolerances is not None:
        tol.update(tolerances)

    if errors["scale_error"] > tol["max_scale_error"]:
        return False, FailureReason.TRANSFORM_ERROR_EXCEEDED, errors

    if errors["rotation_error_deg"] > tol["max_rotation_error_deg"]:
        return False, FailureReason.TRANSFORM_ERROR_EXCEEDED, errors

    if errors["translation_error_px"] > tol["max_translation_error_px"]:
        return False, FailureReason.TRANSFORM_ERROR_EXCEEDED, errors

    return True, None, errors
