"""
src/evaluation/evaluation_types.py

Data structures, taxonomy, and serializable containers for Milestone C
(Phases 18 + 19) Evaluation & Robustness Engine.

ARCHITECTURAL RULES:
- Strictly model-side Python; no UI or web dependencies.
- Clear, explicit failure taxonomy without hiding errors.
- Clean JSON-serializable to_dict() and from_dict() interfaces.
- Lightweight: does NOT serialize full image matrices into report dictionaries.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np

from src.matching.transformation import SimilarityTransform2D
from src.matching.hybrid_matcher import HybridMatchResult
from src.registration.registration_engine import RegistrationResult


class FailureReason:
    """Standardized failure taxonomy for cross-sensor location matching & registration."""
    NONE = "NONE"
    NO_VALID_MATCHES = "NO_VALID_MATCHES"
    INSUFFICIENT_INLIERS = "INSUFFICIENT_INLIERS"
    DEGENERATE_GEOMETRY = "DEGENERATE_GEOMETRY"
    LOW_SPATIAL_COVERAGE = "LOW_SPATIAL_COVERAGE"
    HIGH_REPROJECTION_ERROR = "HIGH_REPROJECTION_ERROR"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    REGISTRATION_FAILURE = "REGISTRATION_FAILURE"
    TRANSFORM_ERROR_EXCEEDED = "TRANSFORM_ERROR_EXCEEDED"
    INVALID_INPUT = "INVALID_INPUT"
    UNKNOWN_FAILURE = "UNKNOWN_FAILURE"

    ALL_REASONS = [
        NONE,
        NO_VALID_MATCHES,
        INSUFFICIENT_INLIERS,
        DEGENERATE_GEOMETRY,
        LOW_SPATIAL_COVERAGE,
        HIGH_REPROJECTION_ERROR,
        LOW_CONFIDENCE,
        REGISTRATION_FAILURE,
        TRANSFORM_ERROR_EXCEEDED,
        INVALID_INPUT,
        UNKNOWN_FAILURE,
    ]


@dataclass
class EvaluationCase:
    """
    Encapsulates a benchmark case (image pair, optional ground truth, parameters).
    Designed to easily accept real lunar imagery metadata when available.
    """
    case_name: str
    image_a: np.ndarray
    image_b: np.ndarray
    ground_truth: Optional[SimilarityTransform2D] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_behavior: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationMetrics:
    """
    Structured metrics across matching, transformation accuracy, and registration quality.
    """
    # Matching metrics
    num_candidates: int = 0
    num_inliers: int = 0
    inlier_ratio: float = 0.0

    # Ground-truth transformation accuracy (populated only when GT available)
    scale_error: Optional[float] = None
    relative_scale_error: Optional[float] = None
    rotation_error_deg: Optional[float] = None
    translation_error_px: Optional[float] = None

    # Registration quality metrics
    rmse: Optional[float] = None
    mean_error: Optional[float] = None
    median_error: Optional[float] = None
    max_error: Optional[float] = None
    coverage: float = 0.0
    confidence: float = 0.0
    quality: str = "FAILED"
    num_tie_points: int = 0
    valid_overlap_pixels: int = 0

    # Evaluation outcomes
    matched: bool = False
    registered: bool = False
    correct_registration: bool = False
    failure_reason: str = FailureReason.NONE

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics to JSON-compatible dictionary."""
        return {
            "num_candidates": int(self.num_candidates),
            "num_inliers": int(self.num_inliers),
            "inlier_ratio": round(float(self.inlier_ratio), 4),
            "scale_error": round(float(self.scale_error), 6) if self.scale_error is not None else None,
            "relative_scale_error": round(float(self.relative_scale_error), 6) if self.relative_scale_error is not None else None,
            "rotation_error_deg": round(float(self.rotation_error_deg), 4) if self.rotation_error_deg is not None else None,
            "translation_error_px": round(float(self.translation_error_px), 4) if self.translation_error_px is not None else None,
            "rmse": round(float(self.rmse), 4) if self.rmse is not None else None,
            "mean_error": round(float(self.mean_error), 4) if self.mean_error is not None else None,
            "median_error": round(float(self.median_error), 4) if self.median_error is not None else None,
            "max_error": round(float(self.max_error), 4) if self.max_error is not None else None,
            "coverage": round(float(self.coverage), 4),
            "confidence": round(float(self.confidence), 4),
            "quality": str(self.quality),
            "num_tie_points": int(self.num_tie_points),
            "valid_overlap_pixels": int(self.valid_overlap_pixels),
            "matched": bool(self.matched),
            "registered": bool(self.registered),
            "correct_registration": bool(self.correct_registration),
            "failure_reason": str(self.failure_reason),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationMetrics":
        """Deserialize metrics from dictionary."""
        return cls(
            num_candidates=int(data.get("num_candidates", 0)),
            num_inliers=int(data.get("num_inliers", 0)),
            inlier_ratio=float(data.get("inlier_ratio", 0.0)),
            scale_error=float(data["scale_error"]) if data.get("scale_error") is not None else None,
            relative_scale_error=float(data["relative_scale_error"]) if data.get("relative_scale_error") is not None else None,
            rotation_error_deg=float(data["rotation_error_deg"]) if data.get("rotation_error_deg") is not None else None,
            translation_error_px=float(data["translation_error_px"]) if data.get("translation_error_px") is not None else None,
            rmse=float(data["rmse"]) if data.get("rmse") is not None else None,
            mean_error=float(data["mean_error"]) if data.get("mean_error") is not None else None,
            median_error=float(data["median_error"]) if data.get("median_error") is not None else None,
            max_error=float(data["max_error"]) if data.get("max_error") is not None else None,
            coverage=float(data.get("coverage", 0.0)),
            confidence=float(data.get("confidence", 0.0)),
            quality=str(data.get("quality", "FAILED")),
            num_tie_points=int(data.get("num_tie_points", 0)),
            valid_overlap_pixels=int(data.get("valid_overlap_pixels", 0)),
            matched=bool(data.get("matched", False)),
            registered=bool(data.get("registered", False)),
            correct_registration=bool(data.get("correct_registration", False)),
            failure_reason=str(data.get("failure_reason", FailureReason.NONE)),
        )


@dataclass
class EvaluationCaseResult:
    """
    Result of evaluating a single test case.
    """
    case_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_behavior: str = ""
    metrics: EvaluationMetrics = field(default_factory=EvaluationMetrics)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Optional references to intermediate engine results
    hybrid_result: Optional[HybridMatchResult] = None
    registration_result: Optional[RegistrationResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize case result to JSON dictionary."""
        out = {
            "case_name": str(self.case_name),
            "parameters": self.parameters,
            "expected_behavior": str(self.expected_behavior),
            "metrics": self.metrics.to_dict(),
            "metadata": self.metadata,
        }
        if self.hybrid_result is not None:
            out["hybrid_match"] = self.hybrid_result.to_dict()
        if self.registration_result is not None:
            out["registration"] = self.registration_result.to_dict(include_arrays=False)
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationCaseResult":
        """Deserialize case result from dictionary."""
        metrics = EvaluationMetrics.from_dict(data.get("metrics", {}))
        reg_result = None
        if "registration" in data and data["registration"] is not None:
            reg_result = RegistrationResult.from_dict(data["registration"])

        return cls(
            case_name=str(data.get("case_name", "")),
            parameters=dict(data.get("parameters", {})),
            expected_behavior=str(data.get("expected_behavior", "")),
            metrics=metrics,
            metadata=dict(data.get("metadata", {})),
            registration_result=reg_result,
        )


@dataclass
class EvaluationSummary:
    """
    Aggregated evaluation report over a set or suite of test cases.
    """
    total_cases: int = 0
    successful_matches: int = 0
    successful_registrations: int = 0
    correct_registrations: int = 0
    failure_count: int = 0

    mean_rmse: Optional[float] = None
    median_rmse: Optional[float] = None
    mean_inlier_ratio: float = 0.0
    mean_coverage: float = 0.0
    mean_confidence: float = 0.0

    quality_distribution: Dict[str, int] = field(default_factory=dict)
    failure_distribution: Dict[str, int] = field(default_factory=dict)

    # GT transformation accuracy aggregate summary
    mean_scale_error: Optional[float] = None
    median_scale_error: Optional[float] = None
    mean_rotation_error: Optional[float] = None
    median_rotation_error: Optional[float] = None
    mean_translation_error: Optional[float] = None
    median_translation_error: Optional[float] = None

    case_results: List[EvaluationCaseResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize summary to JSON dictionary."""
        return {
            "total_cases": int(self.total_cases),
            "successful_matches": int(self.successful_matches),
            "successful_registrations": int(self.successful_registrations),
            "correct_registrations": int(self.correct_registrations),
            "failure_count": int(self.failure_count),
            "mean_rmse": round(float(self.mean_rmse), 4) if self.mean_rmse is not None else None,
            "median_rmse": round(float(self.median_rmse), 4) if self.median_rmse is not None else None,
            "mean_inlier_ratio": round(float(self.mean_inlier_ratio), 4),
            "mean_coverage": round(float(self.mean_coverage), 4),
            "mean_confidence": round(float(self.mean_confidence), 4),
            "quality_distribution": {k: int(v) for k, v in self.quality_distribution.items()},
            "failure_distribution": {k: int(v) for k, v in self.failure_distribution.items()},
            "mean_scale_error": round(float(self.mean_scale_error), 6) if self.mean_scale_error is not None else None,
            "median_scale_error": round(float(self.median_scale_error), 6) if self.median_scale_error is not None else None,
            "mean_rotation_error": round(float(self.mean_rotation_error), 4) if self.mean_rotation_error is not None else None,
            "median_rotation_error": round(float(self.median_rotation_error), 4) if self.median_rotation_error is not None else None,
            "mean_translation_error": round(float(self.mean_translation_error), 4) if self.mean_translation_error is not None else None,
            "median_translation_error": round(float(self.median_translation_error), 4) if self.median_translation_error is not None else None,
            "case_results": [cr.to_dict() for cr in self.case_results],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationSummary":
        """Deserialize summary from dictionary."""
        case_results = [EvaluationCaseResult.from_dict(c) for c in data.get("case_results", [])]
        return cls(
            total_cases=int(data.get("total_cases", 0)),
            successful_matches=int(data.get("successful_matches", 0)),
            successful_registrations=int(data.get("successful_registrations", 0)),
            correct_registrations=int(data.get("correct_registrations", 0)),
            failure_count=int(data.get("failure_count", 0)),
            mean_rmse=float(data["mean_rmse"]) if data.get("mean_rmse") is not None else None,
            median_rmse=float(data["median_rmse"]) if data.get("median_rmse") is not None else None,
            mean_inlier_ratio=float(data.get("mean_inlier_ratio", 0.0)),
            mean_coverage=float(data.get("mean_coverage", 0.0)),
            mean_confidence=float(data.get("mean_confidence", 0.0)),
            quality_distribution=dict(data.get("quality_distribution", {})),
            failure_distribution=dict(data.get("failure_distribution", {})),
            mean_scale_error=float(data["mean_scale_error"]) if data.get("mean_scale_error") is not None else None,
            median_scale_error=float(data["median_scale_error"]) if data.get("median_scale_error") is not None else None,
            mean_rotation_error=float(data["mean_rotation_error"]) if data.get("mean_rotation_error") is not None else None,
            median_rotation_error=float(data["median_rotation_error"]) if data.get("median_rotation_error") is not None else None,
            mean_translation_error=float(data["mean_translation_error"]) if data.get("mean_translation_error") is not None else None,
            median_translation_error=float(data["median_translation_error"]) if data.get("median_translation_error") is not None else None,
            case_results=case_results,
        )
