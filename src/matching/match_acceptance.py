"""
src/matching/match_acceptance.py

Match Acceptance Engine for Cross-Sensor Lunar Location Matching.

Evaluates already-computed geometric consensus results against rigorous evidence standards
to determine whether a match should be accepted as a true LOCATION MATCH (matched=True).

ARCHITECTURAL RULES:
- RANSAC establishes geometric consensus; MatchAcceptanceEngine determines location match.
- RANSAC success alone does NOT imply location match (matched=True).
- Low inlier counts with small RMSE (e.g., 3 inliers + 0.90 px RMSE) are explicitly REJECTED.
- Pure Python/NumPy implementation with clean dictionary serialization.
- Fully configurable thresholds with deterministic, explainable evaluation.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
import math
import numpy as np

from src.matching.transformation import SimilarityTransform2D


def compute_spatial_coverage(
    points_a: np.ndarray,
    image_shape: Optional[Tuple[int, int]] = None,
    grid_rows: int = 6,
    grid_cols: int = 6,
) -> Optional[float]:
    """
    Compute spatial grid coverage of inlier coordinates over Image A coordinate space.

    Args:
        points_a: (N, 2) array of coordinates in Image A space.
        image_shape: Optional (height, width) of Image A.
        grid_rows: Number of grid rows (default 6).
        grid_cols: Number of grid columns (default 6).

    Returns:
        Fractional grid cell occupancy ratio in [0.0, 1.0], or None if image_shape is unavailable.
    """
    pts = np.asarray(points_a, dtype=np.float64).reshape(-1, 2)
    total_cells = grid_rows * grid_cols

    if len(pts) == 0:
        return 0.0 if image_shape is not None else None

    if image_shape is None or image_shape[0] <= 0 or image_shape[1] <= 0:
        # Check if coordinates are in normalized [0, 1] range
        if np.max(pts) <= 1.0 and np.min(pts) >= 0.0:
            h, w = 1.0, 1.0
        else:
            max_x = float(np.max(pts[:, 0]))
            max_y = float(np.max(pts[:, 1]))
            if max_x > 0 and max_y > 0:
                h, w = max_y * 1.15 + 1.0, max_x * 1.15 + 1.0
            else:
                return None
    else:
        h, w = float(image_shape[0]), float(image_shape[1])

    cell_h = h / max(1, grid_rows)
    cell_w = w / max(1, grid_cols)

    occupied_cells = set()
    for pt in pts:
        x, y = pt[0], pt[1]
        r = int(y / cell_h)
        c = int(x / cell_w)
        r = min(max(r, 0), grid_rows - 1)
        c = min(max(c, 0), grid_cols - 1)
        occupied_cells.add((r, c))

    coverage = float(len(occupied_cells) / total_cells)
    return round(coverage, 4)


@dataclass
class MatchAcceptanceResult:
    """
    Structured result container for Match Acceptance Engine decisions.

    Attributes:
        accepted: Boolean indicating whether the match passed all location acceptance criteria.
        status: Status classification string ('ACCEPTED' or 'REJECTED').
        reason: Human-readable summary explanation of the decision and failed checks.
        checks: Detailed breakdown of each individual criterion check.
        metrics: Input metrics evaluated during decision making.
        acceptance_score: Deterministic evidence/quality score in [0.0, 1.0].
            NOTE: This is a quality score based on physical evidence, NOT a probability.
    """
    accepted: bool
    status: str
    reason: str
    checks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    acceptance_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize match acceptance result to primitive dictionary."""
        return {
            "accepted": bool(self.accepted),
            "status": str(self.status),
            "reason": str(self.reason),
            "acceptance_score": round(float(self.acceptance_score), 4),
            "checks": self.checks,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MatchAcceptanceResult":
        """Deserialize match acceptance result from primitive dictionary."""
        return cls(
            accepted=bool(data.get("accepted", False)),
            status=str(data.get("status", "REJECTED")),
            reason=str(data.get("reason", "")),
            checks=dict(data.get("checks", {})),
            metrics=dict(data.get("metrics", {})),
            acceptance_score=float(data.get("acceptance_score", 0.0)),
        )


class MatchAcceptanceEngine:
    """
    Match Acceptance Engine.

    Evaluates geometric consensus metrics (inliers, inlier ratio, RMSE, spatial coverage,
    pipeline confidence, and transform parameters) against configurable thresholds to determine
    whether a match constitutes a true location match.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg: Dict[str, Any] = {}
        if config is not None:
            cfg.update(config)
        else:
            try:
                from configs.default import Config
                cfg.update(getattr(Config, "MATCH_ACCEPTANCE", {}))
            except (ImportError, AttributeError):
                pass

        self.enabled: bool = bool(cfg.get("enabled", True))
        self.minimum_inliers: int = int(cfg.get("minimum_inliers", 10))
        self.minimum_inlier_ratio: float = float(cfg.get("minimum_inlier_ratio", 0.20))
        self.maximum_rmse_px: float = float(cfg.get("maximum_rmse_px", 3.0))
        self.minimum_coverage: float = float(cfg.get("minimum_coverage", 0.15))
        self.minimum_confidence: float = float(cfg.get("minimum_confidence", 0.50))
        self.require_finite_transform: bool = bool(cfg.get("require_finite_transform", True))
        self.require_positive_scale: bool = bool(cfg.get("require_positive_scale", True))

    def evaluate(
        self,
        num_inliers: int,
        num_candidates: int,
        inlier_ratio: float,
        rmse: Optional[float],
        coverage: Optional[float],
        confidence: Optional[float],
        scale: Optional[float] = None,
        rotation_deg: Optional[float] = None,
        translation_x: Optional[float] = None,
        translation_y: Optional[float] = None,
        transform: Optional[Any] = None,
        ransac_success: bool = True,
        override_minimum_inliers: Optional[int] = None,
        override_minimum_coverage: Optional[float] = None,
        override_minimum_confidence: Optional[float] = None,
    ) -> MatchAcceptanceResult:
        """
        Evaluate location match evidence against acceptance standards.

        Args:
            num_inliers: Number of geometrically verified RANSAC inliers.
            num_candidates: Total candidate correspondences before RANSAC.
            inlier_ratio: Ratio of inliers to total correspondences.
            rmse: Reprojection RMSE in pixels (None if unavailable).
            coverage: Spatial grid occupancy coverage in [0.0, 1.0] (None if unavailable).
            confidence: Overall pipeline confidence score in [0.0, 1.0] (None if unavailable).
            scale: Transform isotropic scale factor (optional if transform object passed).
            rotation_deg: Transform rotation angle in degrees (optional if transform object passed).
            translation_x: Transform translation X in pixels (optional if transform object passed).
            translation_y: Transform translation Y in pixels (optional if transform object passed).
            transform: Optional SimilarityTransform2D object.
            ransac_success: Boolean indicating whether RANSAC succeeded.
            override_minimum_inliers: Optional runtime override for minimum required inliers.
            override_minimum_coverage: Optional runtime override for minimum required coverage.
            override_minimum_confidence: Optional runtime override for minimum required confidence.

        Returns:
            MatchAcceptanceResult container.
        """
        min_inliers_thresh = override_minimum_inliers if override_minimum_inliers is not None else self.minimum_inliers
        min_coverage_thresh = override_minimum_coverage if override_minimum_coverage is not None else self.minimum_coverage
        min_confidence_thresh = override_minimum_confidence if override_minimum_confidence is not None else self.minimum_confidence
        # Extract transform values if transform object passed
        if transform is not None:
            if hasattr(transform, "scale"):
                scale = float(transform.scale)
            if hasattr(transform, "rotation_deg"):
                rotation_deg = float(transform.rotation_deg)
            if hasattr(transform, "translation"):
                tx, ty = transform.translation
                translation_x, translation_y = float(tx), float(ty)

        # Build raw metrics dictionary for provenance
        metrics_dict: Dict[str, Any] = {
            "num_inliers": int(num_inliers),
            "num_candidates": int(num_candidates),
            "inlier_ratio": round(float(inlier_ratio), 4) if inlier_ratio is not None else 0.0,
            "rmse": round(float(rmse), 4) if rmse is not None else None,
            "coverage": round(float(coverage), 4) if coverage is not None else None,
            "confidence": round(float(confidence), 4) if confidence is not None else None,
            "scale": round(float(scale), 6) if scale is not None else None,
            "rotation_deg": round(float(rotation_deg), 4) if rotation_deg is not None else None,
            "translation_x": round(float(translation_x), 4) if translation_x is not None else None,
            "translation_y": round(float(translation_y), 4) if translation_y is not None else None,
            "ransac_success": bool(ransac_success),
        }

        # If engine is disabled, pass-through RANSAC result
        if not self.enabled:
            return MatchAcceptanceResult(
                accepted=bool(ransac_success),
                status="ACCEPTED" if ransac_success else "REJECTED",
                reason="Match Acceptance Engine disabled by configuration",
                checks={},
                metrics=metrics_dict,
                acceptance_score=1.0 if ransac_success else 0.0,
            )

        # Perform individual criterion checks
        checks: Dict[str, Dict[str, Any]] = {}
        failed_reasons: List[str] = []

        # 0. RANSAC Success check
        checks["ransac_consensus"] = {
            "passed": bool(ransac_success),
            "value": "SUCCESS" if ransac_success else "FAILED",
            "threshold": "SUCCESS",
        }
        if not ransac_success:
            failed_reasons.append("RANSAC geometric consensus failed")

        # 1. Minimum Inliers check
        passed_inliers = (num_inliers >= min_inliers_thresh)
        checks["minimum_inliers"] = {
            "passed": passed_inliers,
            "value": int(num_inliers),
            "threshold": min_inliers_thresh,
        }
        if not passed_inliers:
            failed_reasons.append(f"only {num_inliers} inliers (minimum required: {min_inliers_thresh})")

        # 2. Minimum Inlier Ratio check
        passed_ratio = (inlier_ratio is not None and inlier_ratio >= self.minimum_inlier_ratio)
        checks["minimum_inlier_ratio"] = {
            "passed": passed_ratio,
            "value": round(float(inlier_ratio), 4) if inlier_ratio is not None else None,
            "threshold": self.minimum_inlier_ratio,
        }
        if not passed_ratio:
            val_str = f"{inlier_ratio:.1%}" if inlier_ratio is not None else "N/A"
            failed_reasons.append(f"inlier ratio {val_str} below threshold ({self.minimum_inlier_ratio:.0%})")

        # 3. Maximum RMSE check
        if rmse is None:
            passed_rmse = False
            rmse_reason = "RMSE metric unavailable"
        else:
            passed_rmse = (math.isfinite(rmse) and rmse <= self.maximum_rmse_px)
            rmse_reason = f"RMSE {rmse:.2f} px exceeds maximum threshold ({self.maximum_rmse_px:.1f} px)" if not passed_rmse else None

        checks["maximum_rmse"] = {
            "passed": passed_rmse,
            "value": round(float(rmse), 4) if rmse is not None else None,
            "threshold": self.maximum_rmse_px,
        }
        if not passed_rmse:
            failed_reasons.append(rmse_reason or "invalid or missing RMSE metric")

        # 4. Minimum Spatial Coverage check
        if coverage is None:
            passed_cov = True
            cov_reason = None
        else:
            max_possible_coverage = min_inliers_thresh / 36.0
            effective_coverage_thresh = min(min_coverage_thresh, max_possible_coverage)
            passed_cov = (math.isfinite(coverage) and coverage >= (effective_coverage_thresh - 1e-4))
            cov_reason = f"spatial coverage {coverage:.1%} below threshold ({effective_coverage_thresh:.0%})" if not passed_cov else None

        checks["minimum_coverage"] = {
            "passed": passed_cov,
            "value": round(float(coverage), 4) if coverage is not None else None,
            "threshold": min_coverage_thresh,
        }
        if not passed_cov:
            failed_reasons.append(cov_reason or "invalid or missing spatial coverage metric")

        # 5. Minimum Confidence check
        if confidence is None:
            passed_conf = False
            conf_reason = "pipeline confidence metric unavailable"
        else:
            passed_conf = (math.isfinite(confidence) and confidence >= min_confidence_thresh)
            conf_reason = f"confidence {confidence:.1%} below threshold ({min_confidence_thresh:.0%})" if not passed_conf else None

        checks["minimum_confidence"] = {
            "passed": passed_conf,
            "value": round(float(confidence), 4) if confidence is not None else None,
            "threshold": min_confidence_thresh,
        }
        if not passed_conf:
            failed_reasons.append(conf_reason or "invalid or missing confidence metric")

        # 6. Transform Sanity check
        transform_valid = True
        transform_reasons: List[str] = []

        if self.require_finite_transform:
            if scale is not None:
                if not math.isfinite(scale):
                    transform_valid = False
                    transform_reasons.append("non-finite scale")
                elif self.require_positive_scale and scale <= 0:
                    transform_valid = False
                    transform_reasons.append(f"non-positive scale ({scale})")

            if rotation_deg is not None and not math.isfinite(rotation_deg):
                transform_valid = False
                transform_reasons.append("non-finite rotation")

            if translation_x is not None and not math.isfinite(translation_x):
                transform_valid = False
                transform_reasons.append("non-finite translation_x")

            if translation_y is not None and not math.isfinite(translation_y):
                transform_valid = False
                transform_reasons.append("non-finite translation_y")

        checks["transform_sanity"] = {
            "passed": transform_valid,
            "details": transform_reasons if not transform_valid else "VALID",
        }
        if not transform_valid:
            failed_reasons.append(f"transform numerical anomaly ({', '.join(transform_reasons)})")

        # All hard checks must pass for acceptance
        accepted = all(c["passed"] for c in checks.values())

        # Compute deterministic evidence score in [0.0, 1.0]
        s_inliers = min(1.0, max(0.0, num_inliers / 30.0))
        s_ratio = min(1.0, max(0.0, inlier_ratio)) if inlier_ratio is not None else 0.0
        s_rmse = float(math.exp(-rmse / self.maximum_rmse_px)) if (rmse is not None and math.isfinite(rmse)) else 0.0
        s_cov = min(1.0, max(0.0, coverage)) if (coverage is not None and math.isfinite(coverage)) else 0.0
        s_conf = min(1.0, max(0.0, confidence)) if (confidence is not None and math.isfinite(confidence)) else 0.0

        acceptance_score = float(0.25 * s_inliers + 0.25 * s_ratio + 0.20 * s_rmse + 0.15 * s_cov + 0.15 * s_conf)
        acceptance_score = float(np.clip(acceptance_score, 0.0, 1.0))

        if accepted:
            status = "ACCEPTED"
            reason = (
                f"Location match accepted: {num_inliers} inliers ({inlier_ratio:.1%}), "
                f"RMSE {rmse:.2f} px, coverage {coverage:.1%}, confidence {confidence:.1%}."
            )
        else:
            status = "REJECTED"
            reason = f"Insufficient geometric evidence: {'; '.join(failed_reasons)}."

        return MatchAcceptanceResult(
            accepted=accepted,
            status=status,
            reason=reason,
            checks=checks,
            metrics=metrics_dict,
            acceptance_score=round(acceptance_score, 4),
        )
