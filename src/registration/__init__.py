"""
src/registration

Registration & Quality Engine package for Milestone B (Combined Phases 14 + 15 + 16 + 17).
Provides:
- Inlier extraction from HybridMatchResult
- Uniform spatial tie-point selection
- Optional sub-pixel local refinement
- Image registration (pull-based inverse warping)
- Quantitative quality metrics & classification
- Unified RegistrationResult and high-level RegistrationEngine
"""

from src.registration.inliers import (
    ExtractedInliers,
    extract_verified_inliers,
)
from src.registration.tie_points import (
    TiePointSelectionResult,
    select_uniform_tie_points,
)
from src.registration.refinement import (
    RefinementResult,
    refine_tie_points,
)
from src.registration.register import (
    register_image,
    get_affine_matrix_from_similarity,
)
from src.registration.quality import (
    QualityMetrics,
    compute_registration_quality,
    compute_residuals,
)
from src.registration.registration_engine import (
    RegistrationResult,
    RegistrationEngine,
)

__all__ = [
    "ExtractedInliers",
    "extract_verified_inliers",
    "TiePointSelectionResult",
    "select_uniform_tie_points",
    "RefinementResult",
    "refine_tie_points",
    "register_image",
    "get_affine_matrix_from_similarity",
    "QualityMetrics",
    "compute_registration_quality",
    "compute_residuals",
    "RegistrationResult",
    "RegistrationEngine",
]
