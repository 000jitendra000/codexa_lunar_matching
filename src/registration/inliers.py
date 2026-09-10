"""
src/registration/inliers.py

Verified inlier extraction module for Milestone B (Registration & Quality Engine).
Extracts, validates, and formats geometrically verified correspondences from HybridMatchResult
using strictly the RANSAC inlier indices.

ARCHITECTURAL RULES:
- Only uses correspondences identified by result.inlier_indices (never outliers).
- Validates indices, finite coordinates, and minimum required point count.
- Preserves provenance (crater vs learned) and per-point confidence.
- Model-side pure Python/NumPy implementation.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np

from src.matching.correspondence_fusion import Correspondence
from src.matching.hybrid_matcher import HybridMatchResult


@dataclass
class ExtractedInliers:
    """
    Container for verified RANSAC inliers extracted from a HybridMatchResult.

    Attributes:
        correspondences: List of verified inlier Correspondence objects.
        points_a: (N, 2) float64 array of coordinates in Image A.
        points_b: (N, 2) float64 array of coordinates in Image B.
        confidences: (N,) float64 array of correspondence confidence scores.
        sources: List of provenance labels ('crater', 'learned', etc.).
        metadata: Diagnostic details regarding extraction.
    """
    correspondences: List[Correspondence]
    points_a: np.ndarray
    points_b: np.ndarray
    confidences: np.ndarray
    sources: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.points_a = np.asarray(self.points_a, dtype=np.float64)
        self.points_b = np.asarray(self.points_b, dtype=np.float64)
        self.confidences = np.asarray(self.confidences, dtype=np.float64)

        n = len(self.correspondences)
        if self.points_a.shape != (n, 2):
            raise ValueError(f"points_a shape {self.points_a.shape} does not match (N={n}, 2).")
        if self.points_b.shape != (n, 2):
            raise ValueError(f"points_b shape {self.points_b.shape} does not match (N={n}, 2).")
        if self.confidences.shape != (n,):
            raise ValueError(f"confidences shape {self.confidences.shape} does not match (N={n},).")
        if len(self.sources) != n:
            raise ValueError(f"sources length {len(self.sources)} does not match N={n}.")

    @property
    def num_inliers(self) -> int:
        return len(self.correspondences)

    @property
    def num_crater_inliers(self) -> int:
        return sum(1 for s in self.sources if s == "crater")

    @property
    def num_learned_inliers(self) -> int:
        return sum(1 for s in self.sources if s == "learned")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize extracted inliers metadata to primitive dictionary."""
        return {
            "num_inliers": self.num_inliers,
            "num_crater_inliers": self.num_crater_inliers,
            "num_learned_inliers": self.num_learned_inliers,
            "sources": list(self.sources),
            "confidences": [round(float(c), 4) for c in self.confidences.tolist()],
            "metadata": self.metadata,
        }


def extract_verified_inliers(
    result: HybridMatchResult,
    min_inliers: int = 2,
) -> ExtractedInliers:
    """
    Extract verified RANSAC inliers from a HybridMatchResult.

    Args:
        result: HybridMatchResult produced by HybridMatcher.
        min_inliers: Minimum number of required inlier points (default 2 for similarity).

    Returns:
        ExtractedInliers container with coordinate arrays and metadata.

    Raises:
        ValueError: If match was unsuccessful, inliers are missing, or fewer than min_inliers exist.
        IndexError: If an inlier index is out of bounds with respect to result.correspondences.
    """
    if not isinstance(result, HybridMatchResult):
        raise TypeError(f"Expected HybridMatchResult instance, got {type(result)}.")

    if not result.matched:
        raise ValueError(
            f"Cannot extract inliers from unsuccessful HybridMatchResult (matched=False). "
            f"Message: {result.metadata.get('message', 'No message')}"
        )

    if not result.inlier_indices:
        raise ValueError("No inlier indices present in HybridMatchResult.")

    num_corrs = len(result.correspondences)
    inlier_corrs: List[Correspondence] = []

    for idx in result.inlier_indices:
        if idx < 0 or idx >= num_corrs:
            raise IndexError(
                f"Inlier index {idx} out of range for correspondences list of length {num_corrs}."
            )
        inlier_corrs.append(result.correspondences[idx])

    if len(inlier_corrs) < min_inliers:
        raise ValueError(
            f"Extracted {len(inlier_corrs)} inliers, which is fewer than required min_inliers={min_inliers}."
        )

    # Convert coordinates and validate
    pts_a = np.array([c.point_a for c in inlier_corrs], dtype=np.float64)
    pts_b = np.array([c.point_b for c in inlier_corrs], dtype=np.float64)
    confs = np.array([c.confidence for c in inlier_corrs], dtype=np.float64)
    sources = [c.source for c in inlier_corrs]

    if not np.all(np.isfinite(pts_a)):
        raise ValueError("Non-finite coordinates detected in points_a of extracted inliers.")
    if not np.all(np.isfinite(pts_b)):
        raise ValueError("Non-finite coordinates detected in points_b of extracted inliers.")

    meta = {
        "original_total_correspondences": num_corrs,
        "num_inliers_extracted": len(inlier_corrs),
        "inlier_ratio": float(result.inlier_ratio),
    }

    return ExtractedInliers(
        correspondences=inlier_corrs,
        points_a=pts_a,
        points_b=pts_b,
        confidences=confs,
        sources=sources,
        metadata=meta,
    )
