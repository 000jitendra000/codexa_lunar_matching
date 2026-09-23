"""
src/matching/hybrid_matcher.py

Hybrid Matching Engine combining classical crater geometry (Phases 5-8) and
learned deep feature matching (LoFTR / Phase 11) with correspondence fusion (Phase 12)
and robust geometric verification via Phase 10 RANSAC.

- HybridMatchResult: Dataclass containing the unified matching result, final transform,
  inliers/outliers, RMSE, confidence, and provenance diagnostics.
- HybridMatcher: High-level model-side entry point suitable for downstream application / API integration.

ARCHITECTURAL RULES:
- Exposes a clean Python-only matching API.
- Fully backward-compatible with Phases 1-10.
- Graceful degradation: operates smoothly even if the learned branch is unavailable.
- Uses reprojection error in Phase 10 RANSAC as the sole geometric arbiter.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np

from src.matching.progress import ProgressEvent

from src.crater_detection.pipeline import CraterDetectionPipeline
from src.crater_graph.builder import build_crater_graph
from src.crater_graph.invariants import (
    build_local_invariant_descriptors,
    build_triangle_descriptors,
)
from src.crater_graph.constellation_matcher import match_crater_constellations, ConstellationMatch
from src.matching.transformation import SimilarityTransform2D
from src.matching.ransac import estimate_robust_similarity_transform, RANSACResult
from src.matching.learned_matcher import (
    BaseLearnedMatcher,
    LoFTRMatcher,
    LearnedMatchResult,
)
from src.matching.correspondence_fusion import (
    Correspondence,
    crater_matches_to_correspondences,
    learned_matches_to_correspondences,
    fuse_correspondences,
    correspondences_to_arrays,
)
from src.matching.match_acceptance import (
    MatchAcceptanceResult,
    MatchAcceptanceEngine,
    compute_spatial_coverage,
)

logger = logging.getLogger(__name__)


@dataclass
class HybridMatchResult:
    """
    Unified result produced by the Hybrid Matching Engine.

    Attributes:
        matched: Boolean indicating whether geometric consensus was achieved.
        correspondences: List of all fused candidate correspondences.
        inlier_indices: List of 0-indexed positions within correspondences classified as inliers.
        outlier_indices: List of 0-indexed positions within correspondences classified as outliers.
        transform: Estimated planar SimilarityTransform2D (None if matched=False).
        confidence: Overall match confidence score in [0.0, 1.0].
        rmse: Reprojection Root-Mean-Square Error across verified inliers (None if matched=False).
        inlier_ratio: Ratio of inliers to total fused correspondences.
        metadata: Comprehensive execution metrics and provenance breakdown.
    """
    matched: bool
    correspondences: List[Correspondence]
    inlier_indices: List[int]
    outlier_indices: List[int]
    transform: Optional[SimilarityTransform2D]
    confidence: float
    rmse: Optional[float]
    inlier_ratio: float
    acceptance: Optional[MatchAcceptanceResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_correspondences(self) -> int:
        return len(self.correspondences)

    @property
    def num_inliers(self) -> int:
        return len(self.inlier_indices)

    @property
    def num_outliers(self) -> int:
        return len(self.outlier_indices)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize hybrid match result to primitive dictionary."""
        out = {
            "matched": bool(self.matched),
            "num_correspondences": len(self.correspondences),
            "num_inliers": len(self.inlier_indices),
            "num_outliers": len(self.outlier_indices),
            "inlier_indices": [int(i) for i in self.inlier_indices],
            "outlier_indices": [int(i) for i in self.outlier_indices],
            "transform": self.transform.to_dict() if self.transform is not None else None,
            "confidence": round(float(self.confidence), 4),
            "rmse": round(float(self.rmse), 4) if self.rmse is not None else None,
            "inlier_ratio": round(float(self.inlier_ratio), 4),
            "correspondences": [c.to_dict() for c in self.correspondences],
            "metadata": self.metadata,
        }
        if self.acceptance is not None:
            out["acceptance"] = self.acceptance.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridMatchResult":
        """Deserialize hybrid match result from dictionary."""
        tf = SimilarityTransform2D.from_dict(data["transform"]) if data.get("transform") is not None else None
        corrs = [Correspondence.from_dict(c) for c in data.get("correspondences", [])]
        acc = MatchAcceptanceResult.from_dict(data["acceptance"]) if data.get("acceptance") is not None else None
        return cls(
            matched=bool(data["matched"]),
            correspondences=corrs,
            inlier_indices=[int(i) for i in data.get("inlier_indices", [])],
            outlier_indices=[int(i) for i in data.get("outlier_indices", [])],
            transform=tf,
            confidence=float(data.get("confidence", 0.0)),
            rmse=float(data["rmse"]) if data.get("rmse") is not None else None,
            inlier_ratio=float(data.get("inlier_ratio", 0.0)),
            acceptance=acc,
            metadata=dict(data.get("metadata", {})),
        )


class HybridMatcher:
    """
    High-level model-side Hybrid Matching Engine.

    Unifies:
    1. Classical crater branch (Crater detection -> Graph -> Invariants -> Constellation matching)
    2. Learned deep feature branch (LoFTR / Mock)
    3. Correspondence fusion (deduplication, normalization, source balancing)
    4. Robust geometric verification via Phase 10 RANSAC
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        learned_matcher: Optional[BaseLearnedMatcher] = None,
        crater_pipeline: Optional[CraterDetectionPipeline] = None,
    ):
        self.config: Dict[str, Any] = {}
        if config is not None:
            self.config.update(config)
        else:
            try:
                from configs.default import Config
                self.config.update(getattr(Config, "HYBRID_MATCHING", {}))
            except (ImportError, AttributeError):
                pass

        self.enable_crater: bool = bool(self.config.get("enable_crater_branch", True))
        self.enable_learned: bool = bool(self.config.get("enable_learned_branch", True))
        self.min_inliers: int = int(self.config.get("min_inliers", 3))
        self.reprojection_threshold: float = float(self.config.get("reprojection_threshold", 3.0))

        # Initialize or inject crater pipeline
        if self.enable_crater:
            self.crater_pipeline = crater_pipeline or CraterDetectionPipeline()
        else:
            self.crater_pipeline = None

        # Initialize or inject learned matcher
        if self.enable_learned:
            self.learned_matcher = learned_matcher or LoFTRMatcher()
        else:
            self.learned_matcher = None

        # Initialize Match Acceptance Engine
        acc_cfg = self.config.get("match_acceptance", self.config)
        self.acceptance_engine = MatchAcceptanceEngine(acc_cfg)

    def match(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        progress_callback: Optional[Any] = None,
        **kwargs,
    ) -> HybridMatchResult:
        """
        Execute full end-to-end hybrid matching between Image A and Image B.

        Args:
            image_a: Grayscale image A (H, W) or (H, W, C).
            image_b: Grayscale image B (H, W) or (H, W, C).
            progress_callback: Optional callback receiving ProgressEvent objects.
            **kwargs: Overrides for runtime parameters.

        Returns:
            HybridMatchResult.
        """
        crater_corrs: List[Correspondence] = []
        crater_meta: Dict[str, Any] = {"enabled": self.enable_crater}

        def _notify(evt: ProgressEvent):
            if progress_callback is not None:
                try:
                    progress_callback(evt)
                except Exception as p_err:
                    logger.debug("Progress callback error: %s", p_err)

        # 1. Classical Crater Branch
        if self.enable_crater and self.crater_pipeline is not None:
            try:
                _notify(ProgressEvent(stage="crater_detection", current=1, total=2, progress=0.5, message="Detecting craters in image A"))
                craters_a = self.crater_pipeline.detect(image_a)
                _notify(ProgressEvent(stage="crater_detection", current=2, total=2, progress=1.0, message="Detecting craters in image B"))
                craters_b = self.crater_pipeline.detect(image_b)

                graph_a = build_crater_graph(craters_a)
                graph_b = build_crater_graph(craters_b)

                local_a = build_local_invariant_descriptors(graph_a)
                local_b = build_local_invariant_descriptors(graph_b)

                tri_a = build_triangle_descriptors(graph_a)
                tri_b = build_triangle_descriptors(graph_b)

                match_result = match_crater_constellations(
                    graph_a=graph_a,
                    graph_b=graph_b,
                    descriptors_a=local_a,
                    descriptors_b=local_b,
                    triangles_a=tri_a,
                    triangles_b=tri_b,
                )

                crater_corrs = crater_matches_to_correspondences(
                    graph_a=graph_a,
                    graph_b=graph_b,
                    matches=match_result.matches,
                )
                crater_meta.update({
                    "craters_a": len(craters_a),
                    "craters_b": len(craters_b),
                    "raw_crater_matches": len(match_result.matches),
                    "status": "success",
                })
            except Exception as e:
                logger.warning("Crater branch execution failed: %s", e)
                crater_meta.update({"status": "failed", "reason": str(e)})

        # 2. Learned Image Branch
        learned_corrs: List[Correspondence] = []
        learned_meta: Dict[str, Any] = {"enabled": self.enable_learned}

        if self.enable_learned and self.learned_matcher is not None:
            try:
                # Check if learned_matcher supports progress_callback
                import inspect
                sig = inspect.signature(self.learned_matcher.match)
                if "progress_callback" in sig.parameters:
                    learned_res = self.learned_matcher.match(image_a, image_b, progress_callback=progress_callback)
                else:
                    learned_res = self.learned_matcher.match(image_a, image_b)

                learned_corrs = learned_matches_to_correspondences(learned_res)
                learned_meta.update({
                    "available": learned_res.available,
                    "backend": learned_res.backend,
                    "raw_learned_matches": len(learned_corrs),
                    "status": "success" if learned_res.available else "unavailable",
                    "details": learned_res.metadata,
                })
            except Exception as e:
                logger.warning("Learned branch execution failed: %s", e)
                learned_meta.update({"status": "failed", "reason": str(e)})

        # 3. Fuse Correspondences
        _notify(ProgressEvent(stage="correspondence_fusion", current=1, total=1, progress=1.0, message="Fusing crater and learned correspondences"))
        shape_a = image_a.shape[:2] if hasattr(image_a, "shape") else None
        shape_b = image_b.shape[:2] if hasattr(image_b, "shape") else None

        fused = fuse_correspondences(
            crater_correspondences=crater_corrs,
            learned_correspondences=learned_corrs,
            image_shape_a=shape_a,
            image_shape_b=shape_b,
        )

        _notify(ProgressEvent(stage="geometric_verification", current=1, total=1, progress=1.0, message="Executing RANSAC geometric verification"))
        return self._run_geometric_verification(
            fused_correspondences=fused,
            num_crater_raw=len(crater_corrs),
            num_learned_raw=len(learned_corrs),
            crater_meta=crater_meta,
            learned_meta=learned_meta,
            image_shape_a=shape_a,
            image_shape_b=shape_b,
            **kwargs,
        )

    def match_from_correspondences(
        self,
        crater_correspondences: List[Correspondence],
        learned_correspondences: List[Correspondence],
        image_shape_a: Optional[Tuple[int, int]] = None,
        image_shape_b: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> HybridMatchResult:
        """
        Execute fusion and RANSAC directly on pre-extracted correspondences.
        Useful for synthetic experiments, headless evaluation, and test suites.
        """
        fused = fuse_correspondences(
            crater_correspondences=crater_correspondences,
            learned_correspondences=learned_correspondences,
            image_shape_a=image_shape_a,
            image_shape_b=image_shape_b,
        )
        return self._run_geometric_verification(
            fused_correspondences=fused,
            num_crater_raw=len(crater_correspondences),
            num_learned_raw=len(learned_correspondences),
            crater_meta={"precomputed": True},
            learned_meta={"precomputed": True},
            image_shape_a=image_shape_a,
            image_shape_b=image_shape_b,
            **kwargs,
        )

    def _run_geometric_verification(
        self,
        fused_correspondences: List[Correspondence],
        num_crater_raw: int,
        num_learned_raw: int,
        crater_meta: Dict[str, Any],
        learned_meta: Dict[str, Any],
        **kwargs,
    ) -> HybridMatchResult:
        """Run Phase 10 RANSAC on fused correspondences and construct HybridMatchResult."""
        total_fused = len(fused_correspondences)
        base_meta = {
            "num_crater_matches": num_crater_raw,
            "num_learned_matches": num_learned_raw,
            "num_fused_matches": total_fused,
            "crater_branch": crater_meta,
            "learned_branch": learned_meta,
        }

        min_inliers = int(kwargs.get("min_inliers", self.min_inliers))
        reproj_thresh = float(kwargs.get("reprojection_threshold", self.reprojection_threshold))

        # Check minimum correspondence threshold
        if total_fused < min_inliers:
            base_meta["reason"] = f"Insufficient fused correspondences: {total_fused} < min_inliers ({min_inliers})."
            acc_res = self.acceptance_engine.evaluate(
                num_inliers=0,
                num_candidates=total_fused,
                inlier_ratio=0.0,
                rmse=None,
                coverage=0.0,
                confidence=0.0,
                ransac_success=False,
            )
            base_meta["acceptance"] = acc_res.to_dict()
            return HybridMatchResult(
                matched=False,
                correspondences=fused_correspondences,
                inlier_indices=[],
                outlier_indices=list(range(total_fused)),
                transform=None,
                confidence=0.0,
                rmse=None,
                inlier_ratio=0.0,
                acceptance=acc_res,
                metadata=base_meta,
            )

        pts_a, pts_b, weights = correspondences_to_arrays(fused_correspondences)

        try:
            ransac_kwargs = dict(kwargs)
            ransac_kwargs.setdefault("reprojection_threshold", reproj_thresh)
            ransac_kwargs.setdefault("min_inliers", min_inliers)

            ransac_res: RANSACResult = estimate_robust_similarity_transform(
                points_a=pts_a,
                points_b=pts_b,
                weights=weights,
                **ransac_kwargs,
            )

            # Inlier source provenance counts
            crater_inliers = sum(1 for idx in ransac_res.inlier_indices if fused_correspondences[idx].source == "crater")
            learned_inliers = sum(1 for idx in ransac_res.inlier_indices if fused_correspondences[idx].source == "learned")

            base_meta.update({
                "ransac_iterations": ransac_res.num_iterations,
                "ransac_rmse": ransac_res.rmse,
                "crater_inliers": crater_inliers,
                "learned_inliers": learned_inliers,
                "ransac_metadata": ransac_res.metadata,
            })

            # Calculate confidence combining inlier ratio and RMSE decay
            conf_scale = max(0.0, 1.0 - (ransac_res.rmse / (reproj_thresh * 2.0)))
            hybrid_confidence = float(min(1.0, max(0.0, ransac_res.inlier_ratio * 0.7 + conf_scale * 0.3)))

            # Compute spatial coverage over Image A
            img_shape_a = kwargs.get("image_shape_a")
            inlier_pts_a = pts_a[ransac_res.inlier_indices] if len(ransac_res.inlier_indices) > 0 else np.empty((0, 2))
            coverage = compute_spatial_coverage(inlier_pts_a, image_shape=img_shape_a)
            if coverage is None and "coverage" in kwargs:
                coverage = float(kwargs["coverage"])

            # Determine acceptance overrides if custom min_inliers passed
            override_min_inliers = None
            if "minimum_inliers" in kwargs:
                override_min_inliers = int(kwargs["minimum_inliers"])
            elif "min_inliers" in kwargs:
                override_min_inliers = int(kwargs["min_inliers"])
            elif "min_inliers" in self.config and "match_acceptance" not in self.config:
                override_min_inliers = int(self.config["min_inliers"])
            elif hasattr(self, "min_inliers") and "match_acceptance" not in self.config:
                override_min_inliers = int(self.min_inliers)

            override_coverage = kwargs.get("minimum_coverage")
            override_confidence = kwargs.get("minimum_confidence")

            acc_res = self.acceptance_engine.evaluate(
                num_inliers=len(ransac_res.inlier_indices),
                num_candidates=total_fused,
                inlier_ratio=ransac_res.inlier_ratio,
                rmse=ransac_res.rmse,
                coverage=coverage,
                confidence=hybrid_confidence,
                transform=ransac_res.transform,
                ransac_success=True,
                override_minimum_inliers=override_min_inliers,
                override_minimum_coverage=override_coverage,
                override_minimum_confidence=override_confidence,
            )

            base_meta["acceptance"] = acc_res.to_dict()
            base_meta["coverage"] = coverage

            return HybridMatchResult(
                matched=acc_res.accepted,
                correspondences=fused_correspondences,
                inlier_indices=ransac_res.inlier_indices,
                outlier_indices=ransac_res.outlier_indices,
                transform=ransac_res.transform,
                confidence=hybrid_confidence,
                rmse=ransac_res.rmse,
                inlier_ratio=ransac_res.inlier_ratio,
                acceptance=acc_res,
                metadata=base_meta,
            )

        except ValueError as e:
            base_meta["reason"] = f"RANSAC consensus verification failed: {str(e)}"
            acc_res = self.acceptance_engine.evaluate(
                num_inliers=0,
                num_candidates=total_fused,
                inlier_ratio=0.0,
                rmse=None,
                coverage=0.0,
                confidence=0.0,
                ransac_success=False,
            )
            base_meta["acceptance"] = acc_res.to_dict()
            return HybridMatchResult(
                matched=False,
                correspondences=fused_correspondences,
                inlier_indices=[],
                outlier_indices=list(range(total_fused)),
                transform=None,
                confidence=0.0,
                rmse=None,
                inlier_ratio=0.0,
                acceptance=acc_res,
                metadata=base_meta,
            )
