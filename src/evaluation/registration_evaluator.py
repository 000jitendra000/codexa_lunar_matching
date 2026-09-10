"""
src/evaluation/registration_evaluator.py

Registration accuracy, residual diagnostics, and failure classification evaluator
for Milestone C (Phase 18).

ARCHITECTURAL RULES:
- Evaluates existing HybridMatchResult and RegistrationResult containers.
- Strict diagnostic separation: does not modify matcher or registration algorithms.
- Classifies root causes of failure according to FailureReason taxonomy.
- Seamlessly handles both synthetic cases (with ground truth) and real/blind pairs (without ground truth).
"""

from typing import Dict, Any, Optional
import numpy as np

from src.matching.transformation import SimilarityTransform2D
from src.matching.hybrid_matcher import HybridMatchResult
from src.registration.registration_engine import RegistrationResult
from src.evaluation.evaluation_types import EvaluationMetrics, FailureReason
from src.evaluation.ground_truth import compute_ground_truth_errors, check_ground_truth_compliance


def evaluate_registration(
    registration_result: RegistrationResult,
    hybrid_result: Optional[HybridMatchResult] = None,
    ground_truth: Optional[SimilarityTransform2D] = None,
    tolerances: Optional[Dict[str, Any]] = None,
) -> EvaluationMetrics:
    """
    Evaluate registration performance, compare against ground truth if present,
    and classify success or failure reason.

    Args:
        registration_result: Result produced by RegistrationEngine.
        hybrid_result: Optional HybridMatchResult from HybridMatcher.
        ground_truth: Optional ground-truth SimilarityTransform2D (A -> B).
        tolerances: Optional dictionary with custom tolerance thresholds.

    Returns:
        EvaluationMetrics instance.
    """
    tol = {
        "max_scale_error": 0.05,
        "max_rotation_error_deg": 3.0,
        "max_translation_error_px": 5.0,
        "max_rmse_px": 3.0,
        "min_coverage": 0.15,
        "min_confidence": 0.5,
    }
    if tolerances is not None:
        tol.update(tolerances)

    metrics = EvaluationMetrics()

    # 1. Matching metrics from hybrid result
    if hybrid_result is not None:
        metrics.num_candidates = hybrid_result.num_correspondences
        metrics.num_inliers = hybrid_result.num_inliers
        metrics.inlier_ratio = float(hybrid_result.inlier_ratio)
        metrics.matched = bool(hybrid_result.matched)
    else:
        metrics.matched = bool(registration_result.success)
        metrics.num_inliers = int(registration_result.num_inliers)

    # 2. Registration metrics from registration result
    metrics.registered = bool(registration_result.success)
    metrics.rmse = registration_result.rmse
    metrics.mean_error = registration_result.mean_error
    metrics.median_error = registration_result.median_error
    metrics.max_error = registration_result.max_error
    metrics.coverage = float(registration_result.coverage)
    metrics.confidence = float(registration_result.confidence)
    metrics.quality = str(registration_result.quality)
    metrics.num_tie_points = int(registration_result.num_tie_points)

    if registration_result.valid_mask is not None:
        metrics.valid_overlap_pixels = int(np.count_nonzero(registration_result.valid_mask))
    else:
        metrics.valid_overlap_pixels = 0

    # 3. Ground-truth transformation comparison
    if ground_truth is not None and registration_result.transform is not None:
        gt_errors = compute_ground_truth_errors(ground_truth, registration_result.transform)
        metrics.scale_error = gt_errors["scale_error"]
        metrics.relative_scale_error = gt_errors["relative_scale_error"]
        metrics.rotation_error_deg = gt_errors["rotation_error_deg"]
        metrics.translation_error_px = gt_errors["translation_error_px"]

    # 4. Success and failure classification
    if not metrics.matched:
        metrics.correct_registration = False
        if metrics.num_candidates == 0:
            metrics.failure_reason = FailureReason.NO_VALID_MATCHES
        else:
            metrics.failure_reason = FailureReason.INSUFFICIENT_INLIERS
        return metrics

    if not metrics.registered or registration_result.transform is None:
        metrics.correct_registration = False
        if metrics.coverage < tol["min_coverage"]:
            metrics.failure_reason = FailureReason.LOW_SPATIAL_COVERAGE
        elif metrics.confidence < tol["min_confidence"]:
            metrics.failure_reason = FailureReason.LOW_CONFIDENCE
        elif metrics.rmse is not None and metrics.rmse > tol["max_rmse_px"]:
            metrics.failure_reason = FailureReason.HIGH_REPROJECTION_ERROR
        else:
            metrics.failure_reason = FailureReason.REGISTRATION_FAILURE
        return metrics

    # Matched and registered: check accuracy
    if ground_truth is not None:
        compliant, reason, _ = check_ground_truth_compliance(
            ground_truth,
            registration_result.transform,
            tolerances=tol,
        )
        if compliant:
            metrics.correct_registration = True
            metrics.failure_reason = FailureReason.NONE
        else:
            metrics.correct_registration = False
            metrics.failure_reason = reason or FailureReason.TRANSFORM_ERROR_EXCEEDED
    else:
        # Blind / real imagery evaluation: rely on measurable quality criteria
        is_high_quality = (
            metrics.quality in ("EXCELLENT", "GOOD")
            and (metrics.rmse is not None and metrics.rmse <= tol["max_rmse_px"])
            and (metrics.coverage >= tol["min_coverage"])
            and (metrics.confidence >= tol["min_confidence"])
        )
        if is_high_quality:
            metrics.correct_registration = True
            metrics.failure_reason = FailureReason.NONE
        else:
            metrics.correct_registration = False
            if metrics.coverage < tol["min_coverage"]:
                metrics.failure_reason = FailureReason.LOW_SPATIAL_COVERAGE
            elif metrics.confidence < tol["min_confidence"]:
                metrics.failure_reason = FailureReason.LOW_CONFIDENCE
            elif metrics.rmse is not None and metrics.rmse > tol["max_rmse_px"]:
                metrics.failure_reason = FailureReason.HIGH_REPROJECTION_ERROR
            else:
                metrics.failure_reason = FailureReason.REGISTRATION_FAILURE

    return metrics
