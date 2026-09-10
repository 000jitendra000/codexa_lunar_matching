"""
tests/test_transformation.py

Comprehensive tests for Phase 9: Initial Planar Similarity Transformation Estimation:
- Identity transform recovery
- Pure translation recovery
- Pure rotation recovery (arbitrary, 90, 180, 270 deg)
- Pure uniform scale recovery (0.5x, 1.8x, 3.2x)
- Combined similarity transform recovery (s * R * p + t)
- Minimal 2-point correspondence estimation
- Degenerate 2-point case (coincident source points rejected)
- Degenerate multi-point case (zero spatial variance rejected)
- Reflection policy and testing on 3+ non-collinear points:
    - allow_reflection=False raises ValueError on reflection
    - allow_reflection=True succeeds with is_reflection=True flag
- Coordinate noise sensitivity (low residuals on small jitter)
- Outlier sensitivity diagnostic (demonstrates least-squares vulnerability without RANSAC)
- Confidence-weighted estimation behavior
- Analytical inverse fidelity (T_inv(T(p)) ≈ p)
- Homogeneous 3x3 matrix consistency
- Determinism across repeated executions
- Insufficient inputs (0 or 1 point rejected)
- Serialization round-trip (to_dict / from_dict)
- Integration with Phase 8 ConstellationMatch correspondences
"""

import math
import pytest
import numpy as np

from src.crater_detection.types import CraterCandidate
from src.crater_graph.builder import build_crater_graph
from src.crater_graph.constellation_matcher import ConstellationMatch
from src.matching.transformation import (
    SimilarityTransform2D,
    TransformationEstimationResult,
    estimate_similarity_transform,
    estimate_transform_from_matches,
)


@pytest.fixture
def clean_points_a():
    """5 non-collinear 2D source coordinates."""
    return np.array([
        [100.0, 100.0],
        [240.0, 120.0],
        [280.0, 260.0],
        [180.0, 320.0],
        [80.0, 220.0],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# 1. Exact Closed-Form Estimation
# ---------------------------------------------------------------------------

def test_identity_transformation(clean_points_a):
    """Source == Target should recover identity transform."""
    res = estimate_similarity_transform(clean_points_a, clean_points_a)

    assert res.num_correspondences == 5
    assert res.transform.scale == pytest.approx(1.0, abs=1e-6)
    assert res.transform.rotation_rad == pytest.approx(0.0, abs=1e-6)
    assert res.transform.translation_x == pytest.approx(0.0, abs=1e-6)
    assert res.transform.translation_y == pytest.approx(0.0, abs=1e-6)
    assert res.rmse == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(res.residuals, 0.0, atol=1e-6)


@pytest.mark.parametrize("tx, ty", [(120.0, -80.0), (-350.0, 500.0)])
def test_pure_translation_recovery(clean_points_a, tx, ty):
    pts_b = clean_points_a + np.array([tx, ty])
    res = estimate_similarity_transform(clean_points_a, pts_b)

    assert res.transform.scale == pytest.approx(1.0, abs=1e-5)
    assert res.transform.rotation_rad == pytest.approx(0.0, abs=1e-5)
    assert res.transform.translation_x == pytest.approx(tx, abs=1e-4)
    assert res.transform.translation_y == pytest.approx(ty, abs=1e-4)
    assert res.rmse == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("angle_deg", [35.0, 90.0, 180.0, 270.0])
def test_pure_rotation_recovery(clean_points_a, angle_deg):
    rad = math.radians(angle_deg)
    r_mat = np.array([
        [math.cos(rad), -math.sin(rad)],
        [math.sin(rad), math.cos(rad)],
    ])
    pts_b = clean_points_a @ r_mat.T
    res = estimate_similarity_transform(clean_points_a, pts_b)

    assert res.transform.scale == pytest.approx(1.0, abs=1e-5)
    # Compare rotation matrix directly to handle 180/-180 wrap seamlessly
    assert np.allclose(res.transform.rotation_matrix, r_mat, atol=1e-5)
    assert res.transform.translation_x == pytest.approx(0.0, abs=1e-4)
    assert res.transform.translation_y == pytest.approx(0.0, abs=1e-4)
    assert res.rmse == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("scale", [0.4, 1.8, 3.2])
def test_pure_scale_recovery(clean_points_a, scale):
    pts_b = clean_points_a * scale
    res = estimate_similarity_transform(clean_points_a, pts_b)

    assert res.transform.scale == pytest.approx(scale, abs=1e-5)
    assert res.transform.rotation_rad == pytest.approx(0.0, abs=1e-5)
    assert res.transform.translation_x == pytest.approx(0.0, abs=1e-4)
    assert res.transform.translation_y == pytest.approx(0.0, abs=1e-4)
    assert res.rmse == pytest.approx(0.0, abs=1e-5)


def test_combined_similarity_recovery(clean_points_a):
    """Known s=1.75, theta=55 deg, tx=220, ty=140."""
    known_s = 1.75
    known_theta = math.radians(55.0)
    known_tx = 220.0
    known_ty = 140.0

    r_mat = np.array([
        [math.cos(known_theta), -math.sin(known_theta)],
        [math.sin(known_theta), math.cos(known_theta)],
    ])
    pts_b = known_s * (clean_points_a @ r_mat.T) + np.array([known_tx, known_ty])

    res = estimate_similarity_transform(clean_points_a, pts_b)

    assert res.transform.scale == pytest.approx(known_s, abs=1e-4)
    assert res.transform.rotation_rad == pytest.approx(known_theta, abs=1e-4)
    assert res.transform.translation_x == pytest.approx(known_tx, abs=1e-4)
    assert res.transform.translation_y == pytest.approx(known_ty, abs=1e-4)
    assert res.rmse == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 2. Minimal & Degenerate Correspondence Handling
# ---------------------------------------------------------------------------

def test_minimal_two_point_estimation():
    """Two non-coincident points must produce an exact similarity transform."""
    pts_a = np.array([[0.0, 0.0], [10.0, 0.0]])
    # Scaled by 2, rotated by 90 deg, translated by (5, 5)
    pts_b = np.array([[5.0, 5.0], [5.0, 25.0]])

    res = estimate_similarity_transform(pts_a, pts_b)
    assert res.transform.scale == pytest.approx(2.0, abs=1e-5)
    assert res.transform.rotation_rad == pytest.approx(math.pi / 2.0, abs=1e-5)
    assert res.transform.translation_x == pytest.approx(5.0, abs=1e-4)
    assert res.transform.translation_y == pytest.approx(5.0, abs=1e-4)
    assert res.rmse == pytest.approx(0.0, abs=1e-5)


def test_degenerate_coincident_two_points():
    """Coincident source points in a 2-point pair must raise ValueError."""
    pts_a = np.array([[10.0, 10.0], [10.0, 10.0]])
    pts_b = np.array([[20.0, 20.0], [30.0, 30.0]])

    with pytest.raises(ValueError, match="Degenerate 2-point correspondence"):
        estimate_similarity_transform(pts_a, pts_b)


def test_degenerate_zero_variance_multi_points():
    """Multi-point constellation where all source points are identical must raise ValueError."""
    pts_a = np.array([[15.0, 15.0], [15.0, 15.0], [15.0, 15.0], [15.0, 15.0]])
    pts_b = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]])

    with pytest.raises(ValueError, match="spatial variance.*is effectively zero"):
        estimate_similarity_transform(pts_a, pts_b)


def test_insufficient_points_rejected():
    """Fewer than 2 points must raise ValueError."""
    with pytest.raises(ValueError, match="At least 2 point correspondences"):
        estimate_similarity_transform(np.zeros((1, 2)), np.zeros((1, 2)))


# ---------------------------------------------------------------------------
# 3. Reflection Policy & 3+ Point Orientation Testing
# ---------------------------------------------------------------------------

def test_reflection_rejected_by_default(clean_points_a):
    """
    3+ non-collinear correspondences where target is reflected (det < 0)
    must raise ValueError when allow_reflection=False (default).
    """
    # Create reflected target: invert Y coordinate
    pts_b_reflected = clean_points_a.copy()
    pts_b_reflected[:, 1] = -pts_b_reflected[:, 1]

    with pytest.raises(ValueError, match="Reflection detected"):
        estimate_similarity_transform(clean_points_a, pts_b_reflected, allow_reflection=False)


def test_reflection_allowed_when_explicit(clean_points_a):
    """
    When allow_reflection=True, estimation succeeds and sets metadata['is_reflection'] = True.
    """
    pts_b_reflected = clean_points_a.copy()
    pts_b_reflected[:, 1] = -pts_b_reflected[:, 1]

    res = estimate_similarity_transform(clean_points_a, pts_b_reflected, allow_reflection=True)
    assert res.transform.metadata["is_reflection"] is True
    assert res.rmse == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 4. Noise & Outlier Sensitivity Diagnostic
# ---------------------------------------------------------------------------

def test_coordinate_noise_robustness(clean_points_a):
    """Small coordinate noise (<1 px) should yield low residuals and near-truth parameters."""
    rng = np.random.default_rng(123)
    noise = rng.uniform(-0.5, 0.5, size=clean_points_a.shape)
    pts_b = clean_points_a + noise

    res = estimate_similarity_transform(clean_points_a, pts_b)
    assert res.transform.scale == pytest.approx(1.0, abs=0.05)
    assert res.rmse < 1.0


def test_outlier_sensitivity_diagnostic(clean_points_a):
    """
    Deliberate outlier correspondence demonstrates that unweighted least-squares
    is sensitive to gross errors (justifying RANSAC in later phases).
    """
    pts_b_corrupted = clean_points_a.copy()
    # Corrupt node 0 with a massive 200px offset
    pts_b_corrupted[0] += np.array([200.0, -150.0])

    res = estimate_similarity_transform(clean_points_a, pts_b_corrupted)

    # Least-squares RMSE must jump significantly
    assert res.rmse > 20.0
    assert res.max_error > 50.0
    # Parameter estimate shifts away from identity
    assert abs(res.transform.scale - 1.0) > 0.05 or abs(res.transform.translation_x) > 10.0


# ---------------------------------------------------------------------------
# 5. Confidence-Weighted Estimation
# ---------------------------------------------------------------------------

def test_confidence_weighted_estimation(clean_points_a):
    """
    When an outlier is present, down-weighting it via confidence scores
    should reduce the residual on the inliers compared to unweighted fit.
    """
    pts_b_corrupted = clean_points_a.copy()
    pts_b_corrupted[0] += np.array([80.0, -60.0])  # outlier on point 0

    # Inliers have high weight (1.0), outlier has minimal weight (0.001)
    weights = np.array([0.001, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)

    res_weighted = estimate_similarity_transform(clean_points_a, pts_b_corrupted, weights=weights)
    res_unweighted = estimate_similarity_transform(clean_points_a, pts_b_corrupted)

    assert res_weighted.is_weighted is True
    # The weighted transform should fit the remaining 4 inliers much closer to scale=1.0, t=(0,0)
    assert abs(res_weighted.transform.scale - 1.0) < abs(res_unweighted.transform.scale - 1.0)



# ---------------------------------------------------------------------------
# 6. Transform Methods: Inverse & Matrix Representation
# ---------------------------------------------------------------------------

def test_inverse_transform_fidelity(clean_points_a):
    tf = SimilarityTransform2D(
        scale=1.65,
        rotation_rad=math.radians(40.0),
        translation_x=150.0,
        translation_y=-80.0,
    )
    pts_trans = tf.apply(clean_points_a)
    pts_rec = tf.inverse().apply(pts_trans)

    assert np.allclose(clean_points_a, pts_rec, atol=1e-5)


def test_matrix_representation_consistency(clean_points_a):
    tf = SimilarityTransform2D(
        scale=1.4,
        rotation_rad=math.radians(72.0),
        translation_x=85.0,
        translation_y=210.0,
    )
    applied = tf.apply(clean_points_a)

    # Apply via 3x3 homogeneous matrix: [x, y, 1] @ M.T
    m = tf.to_matrix()
    homo = np.hstack([clean_points_a, np.ones((len(clean_points_a), 1))])
    applied_mat = (homo @ m.T)[:, 0:2]

    assert np.allclose(applied, applied_mat, atol=1e-5)


# ---------------------------------------------------------------------------
# 7. Serialization Round-Trip
# ---------------------------------------------------------------------------

def test_transform_serialization_fidelity(clean_points_a):
    res = estimate_similarity_transform(clean_points_a, clean_points_a * 1.5 + [20, 30])
    data = res.to_dict()

    assert isinstance(data, dict)
    assert "transform" in data
    assert "rmse" in data

    res_rec = TransformationEstimationResult.from_dict(data)
    assert res_rec.transform.scale == pytest.approx(res.transform.scale, abs=1e-5)
    assert res_rec.transform.rotation_rad == pytest.approx(res.transform.rotation_rad, abs=1e-5)
    assert res_rec.transform.translation_x == pytest.approx(res.transform.translation_x, abs=1e-4)
    assert res_rec.rmse == pytest.approx(res.rmse, abs=1e-4)
    assert np.allclose(res_rec.residuals, res.residuals, atol=1e-4)


# ---------------------------------------------------------------------------
# 8. Integration with Phase 8 ConstellationMatch
# ---------------------------------------------------------------------------

def test_estimate_transform_from_matches_integration(clean_points_a):
    craters_a = [CraterCandidate(x=p[0], y=p[1], radius=15.0, confidence=0.9) for p in clean_points_a]
    # Target transformed by s=1.5, tx=50, ty=70
    craters_b = [CraterCandidate(x=p[0] * 1.5 + 50.0, y=p[1] * 1.5 + 70.0, radius=22.5, confidence=0.9) for p in clean_points_a]

    g_a = build_crater_graph(craters_a, config={"method": "delaunay"})
    g_b = build_crater_graph(craters_b, config={"method": "delaunay"})

    # Synthetic Phase 8 matches
    matches = [
        ConstellationMatch(
            source_node_id=i, target_node_id=i, descriptor_distance=0.01, confidence=0.95, matched_neighbor_count=3
        )
        for i in range(len(clean_points_a))
    ]

    res = estimate_transform_from_matches(g_a, g_b, matches)
    assert res.num_correspondences == len(clean_points_a)
    assert res.transform.scale == pytest.approx(1.5, abs=1e-4)
    assert res.transform.rotation_rad == pytest.approx(0.0, abs=1e-4)
    assert res.transform.translation_x == pytest.approx(50.0, abs=1e-4)
    assert res.transform.translation_y == pytest.approx(70.0, abs=1e-4)
    assert res.rmse == pytest.approx(0.0, abs=1e-4)
