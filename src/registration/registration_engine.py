"""
src/registration/registration_engine.py

High-level model-side Registration & Quality Engine for Milestone B.
Takes geometrically verified correspondences from HybridMatchResult and produces
a registered image, overlap mask, and quantitative quality assessment report.

ARCHITECTURAL RULES:
- Model-side pure Python/NumPy/OpenCV pipeline.
- Does NOT build web application, UI, database, or network server.
- Composably coordinates:
    1. Inlier extraction
    2. Uniform spatial tie-point selection
    3. Optional sub-pixel refinement
    4. Final similarity transformation refit (Phase 9 Umeyama)
    5. Image warping into Image A coordinate frame (pull-based)
    6. Quantitative quality evaluation & classification
- Lightweight, backend-friendly serialization via to_dict(include_arrays=False).
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np

from src.matching.correspondence_fusion import Correspondence
from src.matching.transformation import SimilarityTransform2D, estimate_similarity_transform
from src.matching.hybrid_matcher import HybridMatchResult
from src.registration.inliers import extract_verified_inliers
from src.registration.tie_points import select_uniform_tie_points
from src.registration.refinement import refine_tie_points
from src.registration.register import register_image
from src.registration.quality import compute_registration_quality, QualityMetrics

logger = logging.getLogger(__name__)


@dataclass
class RegistrationResult:
    """
    Unified result produced by the Registration & Quality Engine.

    Attributes:
        success: Boolean indicating whether registration succeeded.
        registered_image: Aligned Image B in Image A's coordinate space (None if failed).
        valid_mask: Binary uint8 mask (255 where overlap exists, None if failed).
        transform: Final refined SimilarityTransform2D (None if failed).
        selected_tie_points: List of chosen Correspondence objects used for registration.
        num_inliers: Total verified inliers from RANSAC.
        num_tie_points: Number of spatially distributed tie points used.
        rmse: Final reprojection RMSE in pixels (None if failed).
        mean_error: Mean residual alignment error (None if failed).
        median_error: Median residual alignment error (None if failed).
        max_error: Maximum residual alignment error (None if failed).
        coverage: Spatial grid occupancy ratio in [0.0, 1.0].
        confidence: Bounded registration confidence in [0.0, 1.0].
        quality: Quality classification ('EXCELLENT', 'GOOD', 'FAIR', 'POOR', 'FAILED').
        metadata: Detailed diagnostics, configuration, and execution timings.
    """
    success: bool
    registered_image: Optional[np.ndarray]
    valid_mask: Optional[np.ndarray]
    transform: Optional[SimilarityTransform2D]

    selected_tie_points: List[Correspondence]
    num_inliers: int
    num_tie_points: int

    rmse: Optional[float]
    mean_error: Optional[float]
    median_error: Optional[float]
    max_error: Optional[float]

    coverage: float
    confidence: float
    quality: str

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        """
        Serialize registration result to primitive dictionary.

        Args:
            include_arrays: If False (default), image arrays are omitted to avoid
                enormous JSON payloads while preserving all quantitative metadata.
        """
        out: Dict[str, Any] = {
            "success": bool(self.success),
            "quality": str(self.quality),
            "confidence": round(float(self.confidence), 4),
            "rmse": round(float(self.rmse), 4) if self.rmse is not None else None,
            "mean_error": round(float(self.mean_error), 4) if self.mean_error is not None else None,
            "median_error": round(float(self.median_error), 4) if self.median_error is not None else None,
            "max_error": round(float(self.max_error), 4) if self.max_error is not None else None,
            "coverage": round(float(self.coverage), 4),
            "num_inliers": int(self.num_inliers),
            "num_tie_points": int(self.num_tie_points),
            "transform": self.transform.to_dict() if self.transform is not None else None,
            "selected_tie_points": [c.to_dict() for c in self.selected_tie_points],
            "metadata": self.metadata,
        }

        if self.registered_image is not None:
            out["registered_image_shape"] = list(self.registered_image.shape)
        if self.valid_mask is not None:
            out["valid_mask_shape"] = list(self.valid_mask.shape)
            out["valid_overlap_pixels"] = int(np.count_nonzero(self.valid_mask))

        if include_arrays:
            if self.registered_image is not None:
                out["registered_image"] = self.registered_image.tolist()
            if self.valid_mask is not None:
                out["valid_mask"] = self.valid_mask.tolist()

        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegistrationResult":
        """Deserialize registration result from dictionary."""
        tf = SimilarityTransform2D.from_dict(data["transform"]) if data.get("transform") is not None else None
        pts = [Correspondence.from_dict(c) for c in data.get("selected_tie_points", [])]
        reg_img = np.array(data["registered_image"]) if "registered_image" in data else None
        vmask = np.array(data["valid_mask"], dtype=np.uint8) if "valid_mask" in data else None

        return cls(
            success=bool(data["success"]),
            registered_image=reg_img,
            valid_mask=vmask,
            transform=tf,
            selected_tie_points=pts,
            num_inliers=int(data["num_inliers"]),
            num_tie_points=int(data["num_tie_points"]),
            rmse=float(data["rmse"]) if data.get("rmse") is not None else None,
            mean_error=float(data["mean_error"]) if data.get("mean_error") is not None else None,
            median_error=float(data["median_error"]) if data.get("median_error") is not None else None,
            max_error=float(data["max_error"]) if data.get("max_error") is not None else None,
            coverage=float(data.get("coverage", 0.0)),
            confidence=float(data.get("confidence", 0.0)),
            quality=str(data.get("quality", "FAILED")),
            metadata=dict(data.get("metadata", {})),
        )


class RegistrationEngine:
    """
    High-level model-side Registration & Quality Engine.

    Coordinates tie-point selection, refinement, final transform refit,
    image warping, and quantitative quality assessment.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config: Dict[str, Any] = {}
        if config is not None:
            self.config.update(config)

        # Extract subsystem configurations
        try:
            from configs.default import Config
            default_tie = getattr(Config, "TIE_POINT_SELECTION", {})
            default_ref = getattr(Config, "SUBPIXEL_REFINEMENT", {})
            default_reg = getattr(Config, "REGISTRATION", {})
            default_qty = getattr(Config, "REGISTRATION_QUALITY", {})
        except (ImportError, AttributeError):
            default_tie, default_ref, default_reg, default_qty = {}, {}, {}, {}

        self.tie_point_cfg = dict(default_tie)
        self.tie_point_cfg.update(self.config.get("tie_point_selection", {}))

        self.refinement_cfg = dict(default_ref)
        self.refinement_cfg.update(self.config.get("subpixel_refinement", {}))

        self.reg_cfg = dict(default_reg)
        self.reg_cfg.update(self.config.get("registration", {}))

        self.quality_cfg = dict(default_qty)
        self.quality_cfg.update(self.config.get("registration_quality", {}))

    def register(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        hybrid_result: HybridMatchResult,
    ) -> RegistrationResult:
        """
        Execute the full registration and quality assessment pipeline.

        Args:
            image_a: Reference image (Image A).
            image_b: Query image to register (Image B).
            hybrid_result: Verified HybridMatchResult produced by HybridMatcher.

        Returns:
            RegistrationResult container.
        """
        # 1. Validate hybrid match result
        if (
            not isinstance(hybrid_result, HybridMatchResult)
            or not hybrid_result.matched
            or hybrid_result.transform is None
            or len(hybrid_result.inlier_indices) < 2
        ):
            logger.warning("HybridMatchResult invalid or unverified. Registration aborted.")
            return RegistrationResult(
                success=False,
                registered_image=None,
                valid_mask=None,
                transform=None,
                selected_tie_points=[],
                num_inliers=len(hybrid_result.inlier_indices) if isinstance(hybrid_result, HybridMatchResult) else 0,
                num_tie_points=0,
                rmse=None,
                mean_error=None,
                median_error=None,
                max_error=None,
                coverage=0.0,
                confidence=0.0,
                quality="FAILED",
                metadata={"reason": "HybridMatchResult was not matched or had insufficient inliers."},
            )

        # 2. Extract verified inliers
        try:
            inliers = extract_verified_inliers(hybrid_result, min_inliers=2)
        except (ValueError, IndexError) as exc:
            logger.error("Failed to extract inliers: %s", exc)
            return RegistrationResult(
                success=False,
                registered_image=None,
                valid_mask=None,
                transform=None,
                selected_tie_points=[],
                num_inliers=0,
                num_tie_points=0,
                rmse=None,
                mean_error=None,
                median_error=None,
                max_error=None,
                coverage=0.0,
                confidence=0.0,
                quality="FAILED",
                metadata={"reason": f"Inlier extraction failed: {str(exc)}"},
            )

        # 3. Uniform spatial tie-point selection
        h_a, w_a = image_a.shape[:2]
        tie_result = select_uniform_tie_points(
            inliers=inliers,
            image_shape=(h_a, w_a),
            config=self.tie_point_cfg,
        )

        # Fallback to inliers if fewer than 2 tie points were selected
        if tie_result.num_selected < 2:
            tie_points_a = inliers.points_a
            tie_points_b = inliers.points_b
            tie_corrs = inliers.correspondences
        else:
            tie_points_a = tie_result.points_a
            tie_points_b = tie_result.points_b
            tie_corrs = tie_result.selected_correspondences

        # 4. Optional sub-pixel / local refinement
        refinement_result = refine_tie_points(
            image_a=image_a,
            image_b=image_b,
            points_a=tie_points_a,
            points_b=tie_points_b,
            transform=hybrid_result.transform,
            correspondences=tie_corrs,
            config=self.refinement_cfg,
        )

        final_pts_a = refinement_result.refined_points_a
        final_pts_b = refinement_result.refined_points_b
        final_corrs = refinement_result.refined_correspondences

        # 5. Final transformation refit (Phase 9 Umeyama)
        try:
            estimation = estimate_similarity_transform(
                points_a=final_pts_a,
                points_b=final_pts_b,
                allow_reflection=False,
            )
            final_transform = estimation.transform
        except Exception as exc:
            logger.warning("Final transform refit failed (%s), retaining RANSAC transform.", exc)
            final_transform = hybrid_result.transform

        # 6. Image registration / warping
        try:
            registered_img, valid_mask, _ = register_image(
                image_a=image_a,
                image_b=image_b,
                transform=final_transform,
                output_shape=(h_a, w_a),
                interpolation=self.reg_cfg.get("interpolation", "linear"),
                border_mode=self.reg_cfg.get("border_mode", "constant"),
                border_value=float(self.reg_cfg.get("border_value", 0.0)),
            )
        except Exception as exc:
            logger.error("Image warping failed: %s", exc)
            return RegistrationResult(
                success=False,
                registered_image=None,
                valid_mask=None,
                transform=final_transform,
                selected_tie_points=final_corrs,
                num_inliers=inliers.num_inliers,
                num_tie_points=len(final_corrs),
                rmse=None,
                mean_error=None,
                median_error=None,
                max_error=None,
                coverage=tie_result.coverage,
                confidence=0.0,
                quality="FAILED",
                metadata={"reason": f"Image warping failed: {str(exc)}"},
            )

        # 7. Quality evaluation and classification
        quality_metrics = compute_registration_quality(
            points_a=final_pts_a,
            points_b=final_pts_b,
            transform=final_transform,
            num_candidates=hybrid_result.num_correspondences,
            num_inliers=inliers.num_inliers,
            coverage=tie_result.coverage,
            valid_mask=valid_mask,
            config=self.quality_cfg,
        )

        meta = {
            "tie_point_diagnostics": tie_result.metadata,
            "refinement_diagnostics": refinement_result.to_dict(),
            "quality_diagnostics": quality_metrics.metadata,
            "refit_performed": True,
        }

        return RegistrationResult(
            success=True,
            registered_image=registered_img,
            valid_mask=valid_mask,
            transform=final_transform,
            selected_tie_points=final_corrs,
            num_inliers=inliers.num_inliers,
            num_tie_points=len(final_corrs),
            rmse=quality_metrics.rmse,
            mean_error=quality_metrics.mean_error,
            median_error=quality_metrics.median_error,
            max_error=quality_metrics.max_error,
            coverage=quality_metrics.coverage,
            confidence=quality_metrics.confidence,
            quality=quality_metrics.quality,
            metadata=meta,
        )
