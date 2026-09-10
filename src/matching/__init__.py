"""
src/matching package.

Phase 4, Phase 9, Phase 10 & Milestone A (Phases 11-13) matching modules:
- Feature detectors (SIFT, AKAZE)
- Classical descriptor matcher & RANSAC geometric verification
- Planar similarity transformation estimation (Phase 9)
- Robust similarity transformation estimation with RANSAC (Phase 10)
- Learned deep feature matching & LoFTR integration (Phase 11)
- Correspondence fusion & deduplication (Phase 12)
- Hybrid matching engine (Phase 13 / Milestone A)
"""

from src.matching.transformation import (
    SimilarityTransform2D,
    TransformationEstimationResult,
    estimate_similarity_transform,
    estimate_transform_from_matches,
)
from src.matching.ransac import (
    RANSACResult,
    evaluate_hypothesis,
    estimate_robust_similarity_transform,
    estimate_robust_transform_from_matches,
)
from src.matching.learned_matcher import (
    LearnedCorrespondence,
    LearnedMatchResult,
    BaseLearnedMatcher,
    MockLearnedMatcher,
    LoFTRMatcher,
)
from src.matching.correspondence_fusion import (
    Correspondence,
    crater_matches_to_correspondences,
    learned_matches_to_correspondences,
    fuse_correspondences,
    correspondences_to_arrays,
)
from src.matching.hybrid_matcher import (
    HybridMatchResult,
    HybridMatcher,
)

__all__ = [
    "SimilarityTransform2D",
    "TransformationEstimationResult",
    "estimate_similarity_transform",
    "estimate_transform_from_matches",
    "RANSACResult",
    "evaluate_hypothesis",
    "estimate_robust_similarity_transform",
    "estimate_robust_transform_from_matches",
    "LearnedCorrespondence",
    "LearnedMatchResult",
    "BaseLearnedMatcher",
    "MockLearnedMatcher",
    "LoFTRMatcher",
    "Correspondence",
    "crater_matches_to_correspondences",
    "learned_matches_to_correspondences",
    "fuse_correspondences",
    "correspondences_to_arrays",
    "HybridMatchResult",
    "HybridMatcher",
]


