"""
src/registration/quality.py

Registration quality evaluation and classification module for Milestone B.
Computes rigorous quantitative metrics over registered tie points, spatial coverage,
geometric residual errors, image overlap ratio, and bounded confidence scoring.

ARCHITECTURAL RULES:
- Computes transparent, deterministic metrics: RMSE, mean, median, max residual error.
- Evaluates spatial grid coverage and image overlap ratio.
- Bounded confidence score in [0.0, 1.0] derived from measurable quantities.
- Interpretable engineering quality classification: EXCELLENT, GOOD, FAIR, POOR, FAILED.
- Pure Python/NumPy implementation.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import math
import numpy as np

from src.matching.transformation import SimilarityTransform2D


@dataclass
class QualityMetrics:
    """
    Comprehensive quantitative registration quality metrics.

    Attributes:
        rmse: Root-Mean-Square Error across verified tie points (pixels).
        mean_error: Mean residual alignment error (pixels).
        median_error: Median residual alignment error (pixels).
        max_error: Maximum residual alignment error (pixels).
        inlier_ratio: Ratio of verified inliers to total candidate correspondences.
        coverage: Spatial grid occupancy ratio in [0.0, 1.0].
        overlap_ratio: Fraction of destination image area with valid source pixels in [0.0, 1.0].
        confidence: Overall bounded confidence score in [0.0, 1.0].
        quality: Engineering quality classification ('EXCELLENT', 'GOOD', 'FAIR', 'POOR', 'FAILED').
        metadata: Detailed sub-scores and configuration parameters.
    """
    rmse: Optional[float]
    mean_error: Optional[float]
    median_error: Optional[float]
    max_error: Optional[float]
    inlier_ratio: float
    coverage: float
    overlap_ratio: float
    confidence: float
    quality: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize quality metrics to primitive dictionary."""
        return {
            "rmse": round(float(self.rmse), 4) if self.rmse is not None else None,
            "mean_error": round(float(self.mean_error), 4) if self.mean_error is not None else None,
            "median_error": round(float(self.median_error), 4) if self.median_error is not None else None,
            "max_error": round(float(self.max_error), 4) if self.max_error is not None else None,
            "inlier_ratio": round(float(self.inlier_ratio), 4),
            "coverage": round(float(self.coverage), 4),
            "overlap_ratio": round(float(self.overlap_ratio), 4),
            "confidence": round(float(self.confidence), 4),
            "quality": str(self.quality),
            "metadata": self.metadata,
        }


def compute_residuals(
    points_a: np.ndarray,
    points_b: np.ndarray,
    transform: SimilarityTransform2D,
) -> np.ndarray:
    """
    Compute Euclidean reprojection residuals in Image B coordinate space:
        r_i = ||p_{B, i} - T(p_{A, i})||_2
    """
    pts_a = np.asarray(points_a, dtype=np.float64).reshape(-1, 2)
    pts_b = np.asarray(points_b, dtype=np.float64).reshape(-1, 2)

    if len(pts_a) == 0:
        return np.empty((0,), dtype=np.float64)

    predicted_b = transform.apply(pts_a)
    diff = pts_b - predicted_b
    residuals = np.linalg.norm(diff, axis=1)
    return residuals


def compute_registration_quality(
    points_a: np.ndarray,
    points_b: np.ndarray,
    transform: Optional[SimilarityTransform2D],
    num_candidates: int,
    num_inliers: int,
    coverage: float,
    valid_mask: Optional[np.ndarray] = None,
    config: Optional[Dict[str, Any]] = None,
) -> QualityMetrics:
    """
    Compute quantitative registration quality metrics and classification.

    Args:
        points_a: (M, 2) coordinates in Image A for final tie points.
        points_b: (M, 2) coordinates in Image B for final tie points.
        transform: Final SimilarityTransform2D (None if registration failed).
        num_candidates: Total candidate correspondences before RANSAC.
        num_inliers: Total inliers verified by RANSAC.
        coverage: Spatial grid coverage in [0.0, 1.0].
        valid_mask: Optional uint8 binary overlap mask (255 where overlap exists).
        config: Optional configuration dictionary. Defaults to Config.REGISTRATION_QUALITY.

    Returns:
        QualityMetrics instance with bounded confidence and classification.
    """
    cfg: Dict[str, Any] = {}
    if config is not None:
        cfg.update(config)
    else:
        try:
            from configs.default import Config
            cfg.update(getattr(Config, "REGISTRATION_QUALITY", {}))
        except (ImportError, AttributeError):
            pass

    excellent_rmse = float(cfg.get("excellent_rmse", 1.5))
    good_rmse = float(cfg.get("good_rmse", 3.0))
    fair_rmse = float(cfg.get("fair_rmse", 5.0))
    min_inlier_ratio = float(cfg.get("min_inlier_ratio", 0.3))
    min_coverage = float(cfg.get("min_coverage", 0.1))
    min_tie_points = int(cfg.get("min_tie_points", 3))
    target_tie_points = max(1, int(cfg.get("target_tie_points", 10)))
    rmse_sigma = max(0.1, float(cfg.get("rmse_sigma", 3.0)))

    inlier_ratio = float(num_inliers / num_candidates) if num_candidates > 0 else 0.0

    # Calculate overlap ratio from valid_mask
    if valid_mask is not None and valid_mask.size > 0:
        overlap_ratio = float(np.count_nonzero(valid_mask) / valid_mask.size)
    else:
        overlap_ratio = 0.0

    m = len(points_a)

    # Failure condition: no transform or fewer than 2 correspondences
    if transform is None or m < 2:
        return QualityMetrics(
            rmse=None,
            mean_error=None,
            median_error=None,
            max_error=None,
            inlier_ratio=round(inlier_ratio, 4),
            coverage=round(coverage, 4),
            overlap_ratio=round(overlap_ratio, 4),
            confidence=0.0,
            quality="FAILED",
            metadata={"reason": "Insufficient points or missing transform"},
        )

    # Compute residual errors
    residuals = compute_residuals(points_a, points_b, transform)
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mean_err = float(np.mean(residuals))
    median_err = float(np.median(residuals))
    max_err = float(np.max(residuals))

    # Compute bounded confidence score in [0.0, 1.0]
    # S_inlier in [0, 1]
    s_inlier = min(1.0, max(0.0, inlier_ratio))
    # S_rmse in [0, 1] (exponential decay)
    s_rmse = float(math.exp(-rmse / rmse_sigma))
    # S_coverage in [0, 1]
    s_cov = min(1.0, max(0.0, coverage))
    # S_points in [0, 1]
    s_pts = min(1.0, max(0.0, m / target_tie_points))

    confidence = 0.35 * s_inlier + 0.30 * s_rmse + 0.20 * s_cov + 0.15 * s_pts
    confidence = float(np.clip(confidence, 0.0, 1.0))

    # Quality category classification
    if (
        rmse <= excellent_rmse
        and inlier_ratio >= 0.60
        and coverage >= 0.20
        and m >= 6
    ):
        quality = "EXCELLENT"
    elif (
        rmse <= good_rmse
        and inlier_ratio >= min_inlier_ratio
        and coverage >= min_coverage
        and m >= min_tie_points
    ):
        quality = "GOOD"
    elif (
        rmse <= fair_rmse
        and inlier_ratio >= 0.20
        and m >= min_tie_points
    ):
        quality = "FAIR"
    else:
        quality = "POOR"

    meta = {
        "sub_scores": {
            "inlier_score": round(s_inlier, 4),
            "rmse_score": round(s_rmse, 4),
            "coverage_score": round(s_cov, 4),
            "points_score": round(s_pts, 4),
        },
        "num_residuals": len(residuals),
    }

    return QualityMetrics(
        rmse=round(rmse, 4),
        mean_error=round(mean_err, 4),
        median_error=round(median_err, 4),
        max_error=round(max_err, 4),
        inlier_ratio=round(inlier_ratio, 4),
        coverage=round(coverage, 4),
        overlap_ratio=round(overlap_ratio, 4),
        confidence=round(confidence, 4),
        quality=quality,
        metadata=meta,
    )
