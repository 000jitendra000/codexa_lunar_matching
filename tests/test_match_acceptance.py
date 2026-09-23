"""
tests/test_match_acceptance.py

Comprehensive test suite for the Match Acceptance Engine (Scenarios A through O).
Ensures evidence-based location matching decisions, transform sanity checks, spatial coverage,
serialization fidelity, and integration with HybridMatcher and API schemas.
"""

import math
import numpy as np
import pytest

from src.matching.match_acceptance import (
    MatchAcceptanceResult,
    MatchAcceptanceEngine,
    compute_spatial_coverage,
)
from src.matching.transformation import SimilarityTransform2D
from src.matching.correspondence_fusion import Correspondence
from src.matching.hybrid_matcher import HybridMatcher
from api.schemas import MatchResultSummary, MatchAcceptanceSchema, TransformSchema, TranslationSchema


# ---------------------------------------------------------------------------
# Scenario A: All criteria passed -> ACCEPTED
# ---------------------------------------------------------------------------
def test_scenario_a_all_criteria_passed():
    """Scenario A: High inliers, high ratio, low RMSE, good coverage, high confidence -> ACCEPTED."""
    engine = MatchAcceptanceEngine({
        "minimum_inliers": 10,
        "minimum_inlier_ratio": 0.20,
        "maximum_rmse_px": 3.0,
        "minimum_coverage": 0.15,
        "minimum_confidence": 0.50,
    })

    tf = SimilarityTransform2D(scale=1.0, rotation_rad=0.1, translation_x=10.0, translation_y=20.0)
    res = engine.evaluate(
        num_inliers=25,
        num_candidates=50,
        inlier_ratio=0.50,
        rmse=1.2,
        coverage=0.40,
        confidence=0.85,
        transform=tf,
        ransac_success=True,
    )

    assert res.accepted is True
    assert res.status == "ACCEPTED"
    assert "Location match accepted" in res.reason
    assert res.acceptance_score > 0.6
    assert res.checks["minimum_inliers"]["passed"] is True
    assert res.checks["minimum_inlier_ratio"]["passed"] is True
    assert res.checks["maximum_rmse"]["passed"] is True
    assert res.checks["minimum_coverage"]["passed"] is True
    assert res.checks["minimum_confidence"]["passed"] is True
    assert res.checks["transform_sanity"]["passed"] is True


# ---------------------------------------------------------------------------
# Scenario B: Low inliers (3 inliers, 0.8 px RMSE) -> REJECTED
# ---------------------------------------------------------------------------
def test_scenario_b_low_inliers_rejected():
    """Scenario B: 3 inliers, 0.8 px RMSE (accidental consensus) -> REJECTED."""
    engine = MatchAcceptanceEngine({"minimum_inliers": 10})

    res = engine.evaluate(
        num_inliers=3,
        num_candidates=35,
        inlier_ratio=0.086,
        rmse=0.80,
        coverage=0.20,
        confidence=0.70,
        ransac_success=True,
    )

    assert res.accepted is False
    assert res.status == "REJECTED"
    assert res.checks["minimum_inliers"]["passed"] is False
    assert "only 3 inliers" in res.reason


# ---------------------------------------------------------------------------
# Scenario C: Low inlier ratio -> REJECTED
# ---------------------------------------------------------------------------
def test_scenario_c_low_inlier_ratio_rejected():
    """Scenario C: 30 inliers out of 600 candidates (5.0% ratio < 20.0%) -> REJECTED."""
    engine = MatchAcceptanceEngine({"minimum_inliers": 10, "minimum_inlier_ratio": 0.20})

    res = engine.evaluate(
        num_inliers=30,
        num_candidates=600,
        inlier_ratio=0.05,
        rmse=1.5,
        coverage=0.30,
        confidence=0.60,
        ransac_success=True,
    )

    assert res.accepted is False
    assert res.status == "REJECTED"
    assert res.checks["minimum_inlier_ratio"]["passed"] is False
    assert "inlier ratio" in res.reason


# ---------------------------------------------------------------------------
# Scenario D: High RMSE -> REJECTED
# ---------------------------------------------------------------------------
def test_scenario_d_high_rmse_rejected():
    """Scenario D: 30 inliers but RMSE 6.5 px (> 3.0 px) -> REJECTED."""
    engine = MatchAcceptanceEngine({"maximum_rmse_px": 3.0})

    res = engine.evaluate(
        num_inliers=30,
        num_candidates=50,
        inlier_ratio=0.60,
        rmse=6.5,
        coverage=0.40,
        confidence=0.80,
        ransac_success=True,
    )

    assert res.accepted is False
    assert res.status == "REJECTED"
    assert res.checks["maximum_rmse"]["passed"] is False
    assert "RMSE 6.50 px exceeds maximum threshold" in res.reason


# ---------------------------------------------------------------------------
# Scenario E: Low spatial coverage -> REJECTED
# ---------------------------------------------------------------------------
def test_scenario_e_low_coverage_rejected():
    """Scenario E: 30 inliers clustered in single corner cell (coverage 0.027 < 0.15) -> REJECTED."""
    engine = MatchAcceptanceEngine({"minimum_coverage": 0.15})

    res = engine.evaluate(
        num_inliers=30,
        num_candidates=40,
        inlier_ratio=0.75,
        rmse=1.0,
        coverage=0.027,
        confidence=0.80,
        ransac_success=True,
    )

    assert res.accepted is False
    assert res.status == "REJECTED"
    assert res.checks["minimum_coverage"]["passed"] is False
    assert "spatial coverage" in res.reason


# ---------------------------------------------------------------------------
# Scenario F: Low pipeline confidence -> REJECTED
# ---------------------------------------------------------------------------
def test_scenario_f_low_confidence_rejected():
    """Scenario F: Low overall confidence (0.25 < 0.50) -> REJECTED."""
    engine = MatchAcceptanceEngine({"minimum_confidence": 0.50})

    res = engine.evaluate(
        num_inliers=20,
        num_candidates=40,
        inlier_ratio=0.50,
        rmse=1.5,
        coverage=0.30,
        confidence=0.25,
        ransac_success=True,
    )

    assert res.accepted is False
    assert res.status == "REJECTED"
    assert res.checks["minimum_confidence"]["passed"] is False
    assert "confidence" in res.reason


# ---------------------------------------------------------------------------
# Scenario G: NaN / Inf scale -> REJECTED
# ---------------------------------------------------------------------------
def test_scenario_g_nan_scale_rejected():
    """Scenario G: Non-finite scale factor (NaN) -> REJECTED."""
    engine = MatchAcceptanceEngine()

    res = engine.evaluate(
        num_inliers=25,
        num_candidates=50,
        inlier_ratio=0.50,
        rmse=1.0,
        coverage=0.30,
        confidence=0.80,
        scale=float("nan"),
        rotation_deg=0.0,
        translation_x=0.0,
        translation_y=0.0,
        ransac_success=True,
    )

    assert res.accepted is False
    assert res.checks["transform_sanity"]["passed"] is False
    assert "non-finite scale" in str(res.checks["transform_sanity"]["details"])


# ---------------------------------------------------------------------------
# Scenario H: Negative scale factor -> REJECTED
# ---------------------------------------------------------------------------
def test_scenario_h_negative_scale_rejected():
    """Scenario H: Negative scale factor (-1.5) -> REJECTED."""
    engine = MatchAcceptanceEngine({"require_positive_scale": True})

    res = engine.evaluate(
        num_inliers=25,
        num_candidates=50,
        inlier_ratio=0.50,
        rmse=1.0,
        coverage=0.30,
        confidence=0.80,
        scale=-1.5,
        rotation_deg=0.0,
        translation_x=0.0,
        translation_y=0.0,
        ransac_success=True,
    )

    assert res.accepted is False
    assert res.checks["transform_sanity"]["passed"] is False
    assert "non-positive scale" in str(res.checks["transform_sanity"]["details"])


# ---------------------------------------------------------------------------
# Scenario I: Infinite translation -> REJECTED
# ---------------------------------------------------------------------------
def test_scenario_i_inf_translation_rejected():
    """Scenario I: Infinite translation parameter (inf) -> REJECTED."""
    engine = MatchAcceptanceEngine()

    res = engine.evaluate(
        num_inliers=25,
        num_candidates=50,
        inlier_ratio=0.50,
        rmse=1.0,
        coverage=0.30,
        confidence=0.80,
        scale=1.0,
        rotation_deg=0.0,
        translation_x=float("inf"),
        translation_y=0.0,
        ransac_success=True,
    )

    assert res.accepted is False
    assert res.checks["transform_sanity"]["passed"] is False
    assert "non-finite translation_x" in str(res.checks["transform_sanity"]["details"])


# ---------------------------------------------------------------------------
# Scenario J: RANSAC failure -> REJECTED
# ---------------------------------------------------------------------------
def test_scenario_j_ransac_failure_rejected():
    """Scenario J: Geometric verification failed (ransac_success=False) -> REJECTED."""
    engine = MatchAcceptanceEngine()

    res = engine.evaluate(
        num_inliers=0,
        num_candidates=5,
        inlier_ratio=0.0,
        rmse=None,
        coverage=0.0,
        confidence=0.0,
        ransac_success=False,
    )

    assert res.accepted is False
    assert res.status == "REJECTED"
    assert res.checks["ransac_consensus"]["passed"] is False
    assert "RANSAC geometric consensus failed" in res.reason


# ---------------------------------------------------------------------------
# Scenario K: Spatial grid coverage computation
# ---------------------------------------------------------------------------
def test_scenario_k_spatial_coverage_computation():
    """Scenario K: Verify 6x6 spatial grid occupancy ratio computation."""
    # Image size: 600 x 600 -> cell size 100 x 100
    # Place points in 9 distinct cells out of 36
    pts = np.array([
        [10.0, 10.0],   # cell (0,0)
        [150.0, 10.0],  # cell (0,1)
        [250.0, 10.0],  # cell (0,2)
        [10.0, 150.0],  # cell (1,0)
        [150.0, 150.0], # cell (1,1)
        [250.0, 150.0], # cell (1,2)
        [10.0, 250.0],  # cell (2,0)
        [150.0, 250.0], # cell (2,1)
        [250.0, 250.0], # cell (2,2)
    ])

    cov = compute_spatial_coverage(pts, image_shape=(600, 600), grid_rows=6, grid_cols=6)
    assert cov == pytest.approx(9 / 36, abs=1e-3)

    # Empty points return 0.0
    empty_cov = compute_spatial_coverage(np.empty((0, 2)), image_shape=(600, 600))
    assert empty_cov == 0.0

    # None image shape estimates bounds if coords > 1.0
    none_cov = compute_spatial_coverage(pts, image_shape=None)
    assert none_cov == pytest.approx(0.25, abs=1e-3)


# ---------------------------------------------------------------------------
# Scenario L: Engine disabled pass-through
# ---------------------------------------------------------------------------
def test_scenario_l_disabled_engine_passthrough():
    """Scenario L: Disabled engine passes through RANSAC status."""
    engine = MatchAcceptanceEngine({"enabled": False})

    res_succ = engine.evaluate(num_inliers=2, num_candidates=5, inlier_ratio=0.4, rmse=0.5, coverage=0.05, confidence=0.2, ransac_success=True)
    assert res_succ.accepted is True
    assert res_succ.status == "ACCEPTED"
    assert "disabled" in res_succ.reason

    res_fail = engine.evaluate(num_inliers=0, num_candidates=5, inlier_ratio=0.0, rmse=None, coverage=0.0, confidence=0.0, ransac_success=False)
    assert res_fail.accepted is False
    assert res_fail.status == "REJECTED"


# ---------------------------------------------------------------------------
# Scenario M: Serialization round-trip fidelity
# ---------------------------------------------------------------------------
def test_scenario_m_serialization_round_trip():
    """Scenario M: Verify MatchAcceptanceResult to_dict() and from_dict() fidelity."""
    engine = MatchAcceptanceEngine()
    tf = SimilarityTransform2D(scale=1.1, rotation_rad=0.05, translation_x=5.0, translation_y=-10.0)

    res = engine.evaluate(
        num_inliers=15,
        num_candidates=30,
        inlier_ratio=0.50,
        rmse=1.2,
        coverage=0.25,
        confidence=0.80,
        transform=tf,
        ransac_success=True,
    )

    d = res.to_dict()
    res_restored = MatchAcceptanceResult.from_dict(d)

    assert res_restored.accepted == res.accepted
    assert res_restored.status == res.status
    assert res_restored.reason == res.reason
    assert res_restored.acceptance_score == pytest.approx(res.acceptance_score, abs=1e-4)
    assert res_restored.checks == res.checks
    assert res_restored.metrics == res.metrics


# ---------------------------------------------------------------------------
# Scenario N: Integration with HybridMatcher
# ---------------------------------------------------------------------------
def test_scenario_n_hybrid_matcher_acceptance_rejection():
    """Scenario N: HybridMatcher where RANSAC succeeds but Acceptance fails due to minimum_inliers."""
    # Configure HybridMatcher with RANSAC min_inliers=3, but Acceptance minimum_inliers=10
    config = {
        "min_inliers": 3,
        "match_acceptance": {
            "enabled": True,
            "minimum_inliers": 10,
            "minimum_inlier_ratio": 0.20,
            "maximum_rmse_px": 3.0,
            "minimum_coverage": 0.10,
            "minimum_confidence": 0.40,
        }
    }
    matcher = HybridMatcher(config=config)

    # 4 clean correspondences -> RANSAC finds 4 inliers, but Acceptance requires 10
    tf = SimilarityTransform2D(scale=1.0, rotation_rad=0.0, translation_x=10.0, translation_y=10.0)
    pts_a = np.array([[50.0, 50.0], [100.0, 50.0], [100.0, 100.0], [50.0, 100.0]])
    pts_b = tf.apply(pts_a)

    corrs = [Correspondence(point_a=pts_a[i], point_b=pts_b[i], confidence=0.9, source="learned", source_index=i) for i in range(4)]

    res = matcher.match_from_correspondences([], corrs)

    # Geometric RANSAC found 4 inliers, but Location Match is False due to acceptance failure
    assert res.matched is False
    assert res.acceptance is not None
    assert res.acceptance.accepted is False
    assert res.acceptance.status == "REJECTED"
    assert "only 4 inliers" in res.acceptance.reason


# ---------------------------------------------------------------------------
# Scenario O: API Schema validation with acceptance field
# ---------------------------------------------------------------------------
def test_scenario_o_api_schema_validation():
    """Scenario O: Verify MatchResultSummary includes acceptance schema field."""
    acc_schema = MatchAcceptanceSchema(
        accepted=True,
        status="ACCEPTED",
        reason="Location match accepted: 25 inliers.",
        acceptance_score=0.85,
        checks={"minimum_inliers": {"passed": True, "value": 25, "threshold": 10}},
        metrics={"num_inliers": 25},
    )

    summary = MatchResultSummary(
        matched=True,
        correspondences=50,
        inliers=25,
        inlier_ratio=0.50,
        confidence=0.85,
        rmse=1.2,
        coverage=0.40,
        quality="EXCELLENT",
        transform=TransformSchema(scale=1.0, rotation_deg=0.0, translation=TranslationSchema(x=10.0, y=20.0)),
        acceptance=acc_schema,
        failure_reason=None,
    )

    summary_dict = summary.model_dump()
    assert "acceptance" in summary_dict
    assert summary_dict["acceptance"]["accepted"] is True
    assert summary_dict["acceptance"]["status"] == "ACCEPTED"
    assert summary_dict["acceptance"]["acceptance_score"] == 0.85
