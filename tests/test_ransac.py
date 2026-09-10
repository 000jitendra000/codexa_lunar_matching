"""
tests/test_ransac.py

Unit and regression test suite for Phase 10: Robust Similarity Transformation Estimation (RANSAC).
Verifies:
1. Clean data recovery (all inliers, numerical precision).
2. Single outlier rejection (exact Phase 9 corrupted correspondence).
3. Multiple outliers rejection (dominant consensus recovery).
4. Distractor false correspondence rejection.
5. Small coordinate noise robustness.
6. Minimal sample size = 2 hypothesis generation and coincidence rejection.
7. Degeneracy gatekeeper (numerical zero variance).
8. Deterministic reproducibility (same seed yields bit-identical results).
9. Seed stability on clean data.
10. Reprojection threshold sensitivity.
11. Minimum inlier failure handling (clear ValueError).
12. Adaptive stopping rule (early termination on high inlier ratio).
13. Optional confidence-guided sampling.
14. RANSACResult serialization round-trip.
15. High-level adapter with CraterGraph and ConstellationMatch.
16. Input validation (shapes, lengths, NaN/Inf).
"""

import math
import numpy as np
import pytest

from src.crater_detection.types import CraterCandidate
from src.crater_graph.graph import CraterGraph
from src.crater_graph.builder import build_crater_graph
from src.crater_graph.constellation_matcher import ConstellationMatch
from src.matching.transformation import SimilarityTransform2D, estimate_similarity_transform
from src.matching.ransac import (
    RANSACResult,
    evaluate_hypothesis,
    estimate_robust_similarity_transform,
    estimate_robust_transform_from_matches,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_constellation_source():
    """Asymmetric 8-crater constellation in Image A."""
    return np.array([
        [100.0, 150.0],
        [180.0, 120.0],
        [220.0, 260.0],
        [130.0, 310.0],
        [300.0, 180.0],
        [260.0, 340.0],
        [350.0, 280.0],
        [170.0, 210.0],
    ], dtype=np.float64)


@pytest.fixture
def ground_truth_transform():
    """Known ground-truth similarity transform (scale=1.4, rot=35 deg, tx=50, ty=-25)."""
    scale = 1.4
    theta = math.radians(35.0)
    tx = 50.0
    ty = -25.0
    r_mat = np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)],
    ], dtype=np.float64)
    return SimilarityTransform2D(
        scale=scale,
        rotation_rad=theta,
        translation_x=tx,
        translation_y=ty,
        rotation_matrix=r_mat,
    )


# ---------------------------------------------------------------------------
# 1. Clean Data & Outlier Rejection Tests
# ---------------------------------------------------------------------------

def test_clean_data_recovery(clean_constellation_source, ground_truth_transform):
    """Clean data with 0 outliers must recover ground truth to numerical precision."""
    pts_a = clean_constellation_source
    pts_b = ground_truth_transform.apply(pts_a)

    res = estimate_robust_similarity_transform(
        points_a=pts_a,
        points_b=pts_b,
        random_seed=42,
    )

    assert res.num_inliers == len(pts_a)
    assert len(res.outlier_indices) == 0
    assert res.inlier_ratio == 1.0
    assert res.transform.scale == pytest.approx(ground_truth_transform.scale, abs=1e-4)
    assert res.transform.rotation_rad == pytest.approx(ground_truth_transform.rotation_rad, abs=1e-4)
    assert res.transform.translation_x == pytest.approx(ground_truth_transform.translation_x, abs=1e-4)
    assert res.transform.translation_y == pytest.approx(ground_truth_transform.translation_y, abs=1e-4)
    assert res.rmse == pytest.approx(0.0, abs=1e-4)


def test_single_outlier_rejection(clean_constellation_source, ground_truth_transform):
    """
    Corrupt a single correspondence (same magnitude as Phase 9 outlier demo).
    Verify plain least-squares is severely corrupted, while RANSAC isolates the outlier.
    """
    pts_a = clean_constellation_source
    pts_b = ground_truth_transform.apply(pts_a)

    # Corrupt index 0 with large translation offset
    outlier_idx = 0
    pts_b[outlier_idx] += np.array([120.0, -100.0])

    # 1. Plain Phase 9 least-squares (severely distorted)
    ls_result = estimate_similarity_transform(pts_a, pts_b)
    assert ls_result.rmse > 20.0  # Confirms Phase 9 sensitivity

    # 2. Phase 10 RANSAC
    ransac_result = estimate_robust_similarity_transform(
        points_a=pts_a,
        points_b=pts_b,
        reprojection_threshold=3.0,
        random_seed=42,
    )

    assert outlier_idx in ransac_result.outlier_indices
    assert outlier_idx not in ransac_result.inlier_indices
    assert ransac_result.num_inliers == len(pts_a) - 1
    assert ransac_result.rmse < 1e-4
    assert ransac_result.transform.scale == pytest.approx(ground_truth_transform.scale, abs=1e-4)
    assert ransac_result.transform.rotation_rad == pytest.approx(ground_truth_transform.rotation_rad, abs=1e-4)


def test_multiple_outliers_rejection(clean_constellation_source, ground_truth_transform):
    """Test 8 inliers and 3 severe outliers: dominant consensus must be correctly isolated."""
    pts_a = clean_constellation_source.copy()
    pts_b = ground_truth_transform.apply(pts_a)

    # Append 3 outlier correspondences
    outlier_src = np.array([
        [400.0, 100.0],
        [150.0, 450.0],
        [320.0, 400.0],
    ])
    outlier_tgt = np.array([
        [100.0, 300.0],  # Arbitrary scrambled target coordinates
        [500.0, 150.0],
        [200.0, 100.0],
    ])

    all_pts_a = np.vstack([pts_a, outlier_src])
    all_pts_b = np.vstack([pts_b, outlier_tgt])

    res = estimate_robust_similarity_transform(
        points_a=all_pts_a,
        points_b=all_pts_b,
        reprojection_threshold=3.0,
        random_seed=42,
    )

    expected_inliers = set(range(8))
    expected_outliers = {8, 9, 10}

    assert set(res.inlier_indices) == expected_inliers
    assert set(res.outlier_indices) == expected_outliers
    assert res.num_inliers == 8
    assert res.rmse < 1e-4
    assert res.transform.scale == pytest.approx(ground_truth_transform.scale, abs=1e-3)


def test_distractor_matches_rejected(clean_constellation_source, ground_truth_transform):
    """False crater matches with plausible but inconsistent coordinates are rejected."""
    pts_a = clean_constellation_source.copy()
    pts_b = ground_truth_transform.apply(pts_a)

    # Swap target coordinates between points 2 and 5 (creating 2 inconsistent distractor pairs)
    pts_b[[2, 5]] = pts_b[[5, 2]]

    res = estimate_robust_similarity_transform(
        points_a=pts_a,
        points_b=pts_b,
        reprojection_threshold=3.0,
        random_seed=42,
    )

    assert 2 in res.outlier_indices
    assert 5 in res.outlier_indices
    assert res.num_inliers == 6
    assert res.rmse < 1e-4


def test_small_coordinate_noise(clean_constellation_source, ground_truth_transform):
    """Small sub-threshold coordinate noise (sigma=0.3 px) retains all inliers."""
    pts_a = clean_constellation_source.copy()
    pts_b = ground_truth_transform.apply(pts_a)

    rng = np.random.default_rng(101)
    noise = rng.normal(0.0, 0.3, size=pts_b.shape)
    pts_b_noisy = pts_b + noise

    res = estimate_robust_similarity_transform(
        points_a=pts_a,
        points_b=pts_b_noisy,
        reprojection_threshold=3.0,
        random_seed=42,
    )

    assert res.num_inliers == len(pts_a)
    assert res.rmse < 1.0
    assert res.transform.scale == pytest.approx(ground_truth_transform.scale, abs=0.05)


# ---------------------------------------------------------------------------
# 2. Minimal Sample & Degeneracy Gatekeeper Tests
# ---------------------------------------------------------------------------

def test_minimal_sample_size_2():
    """Minimal sample size = 2 must produce a valid exact hypothesis on non-coincident points."""
    pts_a = np.array([[10.0, 20.0], [40.0, 60.0]])
    tf = SimilarityTransform2D(scale=1.5, rotation_rad=0.5, translation_x=12.0, translation_y=-8.0)
    pts_b = tf.apply(pts_a)

    res = estimate_robust_similarity_transform(
        points_a=pts_a,
        points_b=pts_b,
        min_inliers=2,
        random_seed=42,
    )

    assert res.num_inliers == 2
    assert res.rmse == pytest.approx(0.0, abs=1e-5)
    assert res.transform.scale == pytest.approx(1.5, abs=1e-4)


def test_degenerate_coincident_sample_skipped():
    """
    If dataset contains coincident source points, RANSAC sampling must discard
    degenerate pairs without failing, finding consensus among distinct points.
    """
    # 4 distinct points + 1 point coincident with point 0
    pts_a = np.array([
        [50.0, 50.0],
        [50.0, 50.0],  # Duplicate (coincident)
        [150.0, 50.0],
        [150.0, 150.0],
        [50.0, 150.0],
    ])
    tf = SimilarityTransform2D(scale=1.2, rotation_rad=0.3, translation_x=10.0, translation_y=20.0)
    pts_b = tf.apply(pts_a)

    res = estimate_robust_similarity_transform(
        points_a=pts_a,
        points_b=pts_b,
        min_inliers=4,
        random_seed=42,
    )

    assert res.num_inliers >= 4
    assert res.rmse < 1e-4


def test_all_degenerate_raises_value_error():
    """If all source points are identical (spatial variance = 0), must raise ValueError."""
    pts_a = np.array([[10.0, 10.0], [10.0, 10.0], [10.0, 10.0]])
    pts_b = np.array([[20.0, 20.0], [30.0, 30.0], [40.0, 40.0]])

    with pytest.raises(ValueError, match="RANSAC failed to find consensus"):
        estimate_robust_similarity_transform(
            points_a=pts_a,
            points_b=pts_b,
            random_seed=42,
        )


# ---------------------------------------------------------------------------
# 3. Determinism & Seed Stability
# ---------------------------------------------------------------------------

def test_deterministic_reproducibility(clean_constellation_source, ground_truth_transform):
    """Same seed and same input must produce bit-identical results."""
    pts_a = clean_constellation_source.copy()
    pts_b = ground_truth_transform.apply(pts_a)
    pts_b[1] += [80.0, -70.0]  # Introduce 1 outlier

    res1 = estimate_robust_similarity_transform(pts_a, pts_b, random_seed=77)
    res2 = estimate_robust_similarity_transform(pts_a, pts_b, random_seed=77)

    assert res1.inlier_indices == res2.inlier_indices
    assert res1.outlier_indices == res2.outlier_indices
    assert res1.num_iterations == res2.num_iterations
    assert res1.transform.scale == res2.transform.scale
    assert res1.transform.rotation_rad == res2.transform.rotation_rad
    assert res1.rmse == res2.rmse
    np.testing.assert_array_equal(res1.residuals, res2.residuals)


def test_different_seeds_stability(clean_constellation_source, ground_truth_transform):
    """Different seeds must converge to the same inlier set on moderate outlier data."""
    pts_a = clean_constellation_source.copy()
    pts_b = ground_truth_transform.apply(pts_a)
    pts_b[0] += [100.0, -90.0]

    res_seed1 = estimate_robust_similarity_transform(pts_a, pts_b, random_seed=42)
    res_seed2 = estimate_robust_similarity_transform(pts_a, pts_b, random_seed=999)

    assert set(res_seed1.inlier_indices) == set(res_seed2.inlier_indices)
    assert res_seed1.transform.scale == pytest.approx(res_seed2.transform.scale, abs=1e-5)


# ---------------------------------------------------------------------------
# 4. Parameters, Adaptive Stopping & Inlier Policies
# ---------------------------------------------------------------------------

def test_reprojection_threshold_behavior(clean_constellation_source, ground_truth_transform):
    """A tight reprojection threshold classifies marginal noise as outliers."""
    pts_a = clean_constellation_source.copy()
    pts_b = ground_truth_transform.apply(pts_a)

    # Offset point 3 by 2.5 pixels
    pts_b[3] += np.array([2.5, 0.0])

    # Threshold 3.0: point 3 is an inlier
    res_loose = estimate_robust_similarity_transform(pts_a, pts_b, reprojection_threshold=3.0, random_seed=42)
    assert 3 in res_loose.inlier_indices

    # Threshold 1.0: point 3 must be an outlier
    res_tight = estimate_robust_similarity_transform(pts_a, pts_b, reprojection_threshold=1.0, random_seed=42)
    assert 3 in res_tight.outlier_indices


def test_minimum_inlier_failure(clean_constellation_source, ground_truth_transform):
    """If consensus inliers < min_inliers, must raise ValueError."""
    pts_a = clean_constellation_source[:4].copy()
    # Scramble all target points so no 3 points agree
    pts_b = np.array([
        [10.0, 500.0],
        [800.0, 20.0],
        [300.0, 100.0],
        [900.0, 700.0],
    ])

    with pytest.raises(ValueError, match="RANSAC failed to find consensus"):
        estimate_robust_similarity_transform(
            points_a=pts_a,
            points_b=pts_b,
            min_inliers=3,
            max_iterations=50,
            random_seed=42,
        )


def test_adaptive_stopping_rule(clean_constellation_source, ground_truth_transform):
    """Adaptive stopping on 100% inlier data terminates in very few iterations."""
    pts_a = clean_constellation_source
    pts_b = ground_truth_transform.apply(pts_a)

    res = estimate_robust_similarity_transform(
        points_a=pts_a,
        points_b=pts_b,
        max_iterations=1000,
        adaptive_iterations=True,
        confidence=0.99,
        random_seed=42,
    )

    # For 100% inliers and sample_size=2, ceil(log(0.01)/log(0)) -> 1-3 iterations
    assert res.num_iterations < 10
    assert res.num_iterations < res.metadata["max_iterations_configured"]


def test_confidence_guided_sampling(clean_constellation_source, ground_truth_transform):
    """Verify confidence-guided sampling functions properly and respects geometry."""
    pts_a = clean_constellation_source.copy()
    pts_b = ground_truth_transform.apply(pts_a)
    pts_b[0] += [120.0, -80.0]  # Outlier

    # High confidence for inliers, low for outlier
    weights = np.ones(len(pts_a))
    weights[0] = 0.01
    weights[1:] = 1.0

    res = estimate_robust_similarity_transform(
        points_a=pts_a,
        points_b=pts_b,
        weights=weights,
        use_match_confidence=True,
        random_seed=42,
    )

    assert 0 in res.outlier_indices
    assert res.num_inliers == len(pts_a) - 1
    assert res.rmse < 1e-4


# ---------------------------------------------------------------------------
# 5. Serialization & Match Adapter Tests
# ---------------------------------------------------------------------------

def test_ransac_result_serialization(clean_constellation_source, ground_truth_transform):
    """RANSACResult must round-trip through to_dict() and from_dict() faithfully."""
    pts_a = clean_constellation_source
    pts_b = ground_truth_transform.apply(pts_a)

    res = estimate_robust_similarity_transform(pts_a, pts_b, random_seed=42)
    d = res.to_dict()
    res_restored = RANSACResult.from_dict(d)

    assert res_restored.num_inliers == res.num_inliers
    assert res_restored.inlier_indices == res.inlier_indices
    assert res_restored.outlier_indices == res.outlier_indices
    assert res_restored.rmse == pytest.approx(res.rmse, abs=1e-4)
    assert res_restored.transform.scale == pytest.approx(res.transform.scale, abs=1e-4)


def test_estimate_robust_transform_from_matches(ground_truth_transform):
    """End-to-end integration test with CraterGraph and ConstellationMatch objects."""
    craters_a = [
        CraterCandidate(x=100.0, y=100.0, radius=10.0, confidence=0.9),
        CraterCandidate(x=180.0, y=110.0, radius=12.0, confidence=0.85),
        CraterCandidate(x=140.0, y=190.0, radius=11.0, confidence=0.88),
        CraterCandidate(x=220.0, y=170.0, radius=14.0, confidence=0.92),
    ]
    graph_a = build_crater_graph(craters_a)

    # Transform craters to Image B
    pts_a = np.array([[c.x, c.y] for c in craters_a])
    pts_b = ground_truth_transform.apply(pts_a)

    craters_b = [
        CraterCandidate(x=float(p[0]), y=float(p[1]), radius=float(c.radius * ground_truth_transform.scale), confidence=0.9)
        for c, p in zip(craters_a, pts_b)
    ]
    # Add a distractor crater in B
    craters_b.append(CraterCandidate(x=500.0, y=500.0, radius=15.0, confidence=0.5))
    graph_b = build_crater_graph(craters_b)

    # 3 correct matches + 1 bad match to distractor
    matches = [
        ConstellationMatch(source_node_id=0, target_node_id=0, descriptor_distance=0.05, confidence=0.95, matched_neighbor_count=3),
        ConstellationMatch(source_node_id=1, target_node_id=1, descriptor_distance=0.08, confidence=0.92, matched_neighbor_count=3),
        ConstellationMatch(source_node_id=2, target_node_id=2, descriptor_distance=0.06, confidence=0.94, matched_neighbor_count=3),
        ConstellationMatch(source_node_id=3, target_node_id=4, descriptor_distance=0.35, confidence=0.40, matched_neighbor_count=2),  # Outlier
    ]

    res = estimate_robust_transform_from_matches(graph_a, graph_b, matches, random_seed=42)

    assert 3 in res.outlier_indices  # 4th match is outlier
    assert res.num_inliers == 3
    assert res.transform.scale == pytest.approx(ground_truth_transform.scale, abs=1e-3)


# ---------------------------------------------------------------------------
# 6. Input Validation & Error Handling
# ---------------------------------------------------------------------------

def test_input_validation_errors():
    """Verify appropriate ValueError is raised on malformed input data."""
    # Fewer than sample size
    with pytest.raises(ValueError, match="at least 2 correspondences"):
        estimate_robust_similarity_transform(np.zeros((1, 2)), np.zeros((1, 2)))

    # Length mismatch
    with pytest.raises(ValueError, match="identical length"):
        estimate_robust_similarity_transform(np.zeros((4, 2)), np.zeros((3, 2)))

    # Dimension mismatch
    with pytest.raises(ValueError, match="shape \\(N, 2\\)"):
        estimate_robust_similarity_transform(np.zeros((4, 3)), np.zeros((4, 3)))

    # NaN coordinates
    bad_pts = np.zeros((4, 2))
    bad_pts[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        estimate_robust_similarity_transform(bad_pts, np.zeros((4, 2)))

    # Inf coordinates
    bad_pts2 = np.zeros((4, 2))
    bad_pts2[1, 1] = np.inf
    with pytest.raises(ValueError, match="NaN or Inf"):
        estimate_robust_similarity_transform(np.zeros((4, 2)), bad_pts2)
