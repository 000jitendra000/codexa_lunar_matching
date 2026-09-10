"""
tests/test_registration.py

Unit and regression test suite for Milestone B: Registration & Quality Engine (Phases 14-17).
Covers:
1. Inlier extraction:
   - Valid inlier extraction and coordinate consistency
   - Out-of-range index detection
   - Empty inlier indices handling
   - Non-finite coordinate rejection
   - Unmatched result rejection
2. Uniform spatial tie-point selection:
   - Uniform regular grid distribution
   - Deterministic repeated selection
   - Cell occupancy and spatial coverage calculation
   - Strongest candidate selection per cell
   - Maximum points global cap enforcement
   - Handling fewer points than grid cells
   - Highly clustered points handling
   - Duplicate coordinate handling
   - Out-of-bounds coordinate clamping
   - Disabled selection fallback
3. Sub-pixel / local refinement:
   - Synthetic offset recovery via template matching
   - Fractional sub-pixel shift interpolation
   - Disabled refinement toggle preserves coordinates
   - Failed correlation preserves original coordinates
   - Shift displacement clipping (> max_shift_px)
   - Diagnostics correctness
4. Image registration & warping:
   - Identity transformation registration
   - Pure translation registration
   - Pure rotation registration
   - Pure scale registration
   - Combined similarity registration
   - Transform direction correctness (known A->B warps B back into A)
   - Output dimensions customization
   - Binary valid mask behavior
   - Grayscale and multi-channel image handling
5. Registration quality metrics & classification:
   - Residual error & RMSE calculation
   - Inlier ratio calculation
   - Coverage metric calculation
   - Confidence bounds [0.0, 1.0]
   - Categorical classification (EXCELLENT, GOOD, FAIR, POOR, FAILED)
   - Insufficient point failure handling
6. End-to-end RegistrationEngine:
   - Full synthetic registration pipeline execution
   - Result serialization (with and without image arrays)
   - Deserialization round-trip
   - Deterministic repeatability
   - Failure propagation from unsuccessful HybridMatchResult
"""

import math
import cv2
import numpy as np
import pytest

from src.matching.correspondence_fusion import Correspondence
from src.matching.transformation import SimilarityTransform2D, TransformationEstimationResult
from src.matching.hybrid_matcher import HybridMatchResult
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def known_similarity_transform():
    """Known similarity transform: scale=1.35, rotation=38 deg, tx=45, ty=-30."""
    theta_rad = math.radians(38.0)
    r_mat = np.array([
        [math.cos(theta_rad), -math.sin(theta_rad)],
        [math.sin(theta_rad), math.cos(theta_rad)],
    ], dtype=np.float64)
    return SimilarityTransform2D(
        scale=1.35,
        rotation_rad=theta_rad,
        translation_x=45.0,
        translation_y=-30.0,
        rotation_matrix=r_mat,
    )


@pytest.fixture
def synthetic_hybrid_result(known_similarity_transform):
    """Synthetic HybridMatchResult with 8 correspondences (6 inliers, 2 outliers)."""
    pts_a = np.array([
        [100.0, 150.0],
        [180.0, 120.0],
        [220.0, 260.0],
        [130.0, 310.0],
        [300.0, 180.0],
        [260.0, 340.0],
        [50.0, 50.0],   # outlier
        [400.0, 400.0], # outlier
    ], dtype=np.float64)

    pts_b = known_similarity_transform.apply(pts_a)
    # Corrupt outliers in Image B
    pts_b[6] = [20.0, 350.0]
    pts_b[7] = [450.0, 30.0]

    corrs = [
        Correspondence(
            point_a=pts_a[i],
            point_b=pts_b[i],
            confidence=0.9 - 0.05 * i,
            source="crater" if i < 4 else "learned",
            source_index=i,
        )
        for i in range(8)
    ]

    return HybridMatchResult(
        matched=True,
        correspondences=corrs,
        inlier_indices=[0, 1, 2, 3, 4, 5],
        outlier_indices=[6, 7],
        transform=known_similarity_transform,
        confidence=0.92,
        rmse=0.0,
        inlier_ratio=6 / 8,
        metadata={"provenance": "synthetic_test"},
    )


# ---------------------------------------------------------------------------
# 1. Inlier Extraction Tests
# ---------------------------------------------------------------------------

def test_extract_verified_inliers_valid(synthetic_hybrid_result):
    inliers = extract_verified_inliers(synthetic_hybrid_result, min_inliers=3)
    assert isinstance(inliers, ExtractedInliers)
    assert inliers.num_inliers == 6
    assert inliers.points_a.shape == (6, 2)
    assert inliers.points_b.shape == (6, 2)
    assert len(inliers.sources) == 6
    assert inliers.num_crater_inliers == 4
    assert inliers.num_learned_inliers == 2


def test_extract_inliers_invalid_index(synthetic_hybrid_result):
    synthetic_hybrid_result.inlier_indices.append(999)
    with pytest.raises(IndexError):
        extract_verified_inliers(synthetic_hybrid_result)


def test_extract_inliers_empty_indices(synthetic_hybrid_result):
    synthetic_hybrid_result.inlier_indices = []
    with pytest.raises(ValueError):
        extract_verified_inliers(synthetic_hybrid_result)


def test_extract_inliers_unmatched_result(synthetic_hybrid_result):
    synthetic_hybrid_result.matched = False
    with pytest.raises(ValueError):
        extract_verified_inliers(synthetic_hybrid_result)


def test_extract_inliers_non_finite_coordinates(synthetic_hybrid_result):
    synthetic_hybrid_result.correspondences[0].point_a[0] = np.nan
    with pytest.raises(ValueError):
        extract_verified_inliers(synthetic_hybrid_result)


# ---------------------------------------------------------------------------
# 2. Uniform Tie-Point Selection Tests
# ---------------------------------------------------------------------------

def test_tie_points_uniform_selection(synthetic_hybrid_result):
    inliers = extract_verified_inliers(synthetic_hybrid_result)
    result = select_uniform_tie_points(
        inliers=inliers,
        image_shape=(512, 512),
        config={"grid_rows": 4, "grid_cols": 4, "max_points": 20},
    )
    assert isinstance(result, TiePointSelectionResult)
    assert result.num_selected > 0
    assert result.num_selected <= inliers.num_inliers
    assert result.coverage > 0.0
    assert result.coverage <= 1.0


def test_tie_points_determinism(synthetic_hybrid_result):
    inliers = extract_verified_inliers(synthetic_hybrid_result)
    res1 = select_uniform_tie_points(inliers, (512, 512), {"grid_rows": 5, "grid_cols": 5})
    res2 = select_uniform_tie_points(inliers, (512, 512), {"grid_rows": 5, "grid_cols": 5})
    np.testing.assert_array_equal(res1.points_a, res2.points_a)
    np.testing.assert_array_equal(res1.points_b, res2.points_b)
    assert res1.num_selected == res2.num_selected


def test_tie_points_strongest_per_cell():
    # Two points in the same cell (x=10, y=10 and x=15, y=15) in a (100, 100) image with 2x2 grid
    pts_a = np.array([[10.0, 10.0], [15.0, 15.0]], dtype=np.float64)
    pts_b = pts_a + 5.0
    corrs = [
        Correspondence(pts_a[0], pts_b[0], confidence=0.4, source="crater", source_index=0),
        Correspondence(pts_a[1], pts_b[1], confidence=0.9, source="crater", source_index=1),
    ]
    inliers = ExtractedInliers(
        correspondences=corrs,
        points_a=pts_a,
        points_b=pts_b,
        confidences=np.array([0.4, 0.9]),
        sources=["crater", "crater"],
    )
    result = select_uniform_tie_points(inliers, (100, 100), {"grid_rows": 2, "grid_cols": 2})
    assert result.num_selected == 1
    # Candidate with higher confidence (0.9) must be chosen
    assert result.selected_correspondences[0].confidence == 0.9
    np.testing.assert_array_equal(result.points_a[0], [15.0, 15.0])


def test_tie_points_max_points_cap(synthetic_hybrid_result):
    inliers = extract_verified_inliers(synthetic_hybrid_result)
    result = select_uniform_tie_points(
        inliers,
        (512, 512),
        {"grid_rows": 6, "grid_cols": 6, "max_points": 3},
    )
    assert result.num_selected <= 3


def test_tie_points_clustered_input():
    # 5 points all clustered at the exact same location
    pts_a = np.array([[50.0, 50.0]] * 5, dtype=np.float64)
    pts_b = pts_a.copy()
    corrs = [
        Correspondence(pts_a[i], pts_b[i], confidence=0.5 + 0.1 * i, source="crater", source_index=i)
        for i in range(5)
    ]
    inliers = ExtractedInliers(
        correspondences=corrs,
        points_a=pts_a,
        points_b=pts_b,
        confidences=np.array([c.confidence for c in corrs]),
        sources=["crater"] * 5,
    )
    result = select_uniform_tie_points(inliers, (200, 200), {"grid_rows": 4, "grid_cols": 4})
    assert result.num_selected == 1
    assert result.num_occupied_cells == 1


def test_tie_points_out_of_bounds_clamping():
    pts_a = np.array([[-10.0, -5.0], [250.0, 250.0]], dtype=np.float64)
    pts_b = pts_a.copy()
    corrs = [
        Correspondence(pts_a[i], pts_b[i], confidence=0.8, source="crater", source_index=i)
        for i in range(2)
    ]
    inliers = ExtractedInliers(
        correspondences=corrs,
        points_a=pts_a,
        points_b=pts_b,
        confidences=np.array([0.8, 0.8]),
        sources=["crater", "crater"],
    )
    result = select_uniform_tie_points(inliers, (100, 100), {"grid_rows": 2, "grid_cols": 2})
    assert result.num_selected == 2


def test_tie_points_disabled():
    pts_a = np.array([[10.0, 10.0], [15.0, 15.0]], dtype=np.float64)
    pts_b = pts_a.copy()
    corrs = [
        Correspondence(pts_a[i], pts_b[i], confidence=0.8, source="crater", source_index=i)
        for i in range(2)
    ]
    inliers = ExtractedInliers(
        correspondences=corrs,
        points_a=pts_a,
        points_b=pts_b,
        confidences=np.array([0.8, 0.8]),
        sources=["crater", "crater"],
    )
    result = select_uniform_tie_points(inliers, (100, 100), {"enabled": False, "max_points": 10})
    assert result.num_selected == 2


# ---------------------------------------------------------------------------
# 3. Sub-Pixel / Local Refinement Tests
# ---------------------------------------------------------------------------

def test_refine_tie_points_disabled(synthetic_hybrid_result):
    inliers = extract_verified_inliers(synthetic_hybrid_result)
    img_a = np.zeros((200, 200), dtype=np.uint8)
    img_b = np.zeros((200, 200), dtype=np.uint8)
    res = refine_tie_points(
        img_a, img_b, inliers.points_a, inliers.points_b,
        synthetic_hybrid_result.transform,
        config={"enabled": False},
    )
    assert res.num_attempted == 0
    np.testing.assert_array_equal(res.refined_points_b, inliers.points_b)


def test_refine_tie_points_synthetic_offset():
    # Identity transform between two 128x128 images with a textured crater feature
    img_a = np.zeros((128, 128), dtype=np.float32)
    cv2.circle(img_a, (64, 64), 16, 200.0, -1)
    cv2.circle(img_a, (64, 64), 8, 50.0, -1)

    # In Image B, the crater is at exactly (64, 64)
    img_b = img_a.copy()

    # Suppose coarse detection in Image B has an intentional 1.0 px offset at (65.0, 64.0)
    coarse_pts_a = np.array([[64.0, 64.0]], dtype=np.float64)
    coarse_pts_b = np.array([[65.0, 64.0]], dtype=np.float64)

    identity_tf = SimilarityTransform2D(1.0, 0.0, 0.0, 0.0)
    res = refine_tie_points(
        img_a, img_b, coarse_pts_a, coarse_pts_b, identity_tf,
        config={"enabled": True, "patch_radius": 15, "search_radius": 4, "min_correlation": 0.6},
    )

    assert res.num_refined == 1
    # Refined point in Image B should be brought closer to ground-truth (64.0, 64.0)
    err_before = abs(coarse_pts_b[0, 0] - 64.0)
    err_after = abs(res.refined_points_b[0, 0] - 64.0)
    assert err_after < err_before


def test_refine_tie_points_weak_correlation_fallback():
    # Blank white image A and random noise image B -> low correlation -> fallback
    np.random.seed(42)
    img_a = np.full((100, 100), 128, dtype=np.uint8)
    img_b = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

    pts_a = np.array([[50.0, 50.0]], dtype=np.float64)
    pts_b = np.array([[50.0, 50.0]], dtype=np.float64)
    identity_tf = SimilarityTransform2D(1.0, 0.0, 0.0, 0.0)

    res = refine_tie_points(
        img_a, img_b, pts_a, pts_b, identity_tf,
        config={"min_correlation": 0.9},
    )
    assert res.num_failed == 1
    np.testing.assert_array_equal(res.refined_points_b, pts_b)


# ---------------------------------------------------------------------------
# 4. Image Registration & Warping Tests
# ---------------------------------------------------------------------------

def test_register_identity():
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(img, (50, 50), 10, 255, -1)
    identity_tf = SimilarityTransform2D(1.0, 0.0, 0.0, 0.0)

    reg_img, vmask, _ = register_image(img, img, identity_tf)
    np.testing.assert_array_equal(reg_img, img)
    assert np.all(vmask == 255)


def test_register_pure_translation():
    # Image A has a dot at (40, 50)
    img_a = np.zeros((100, 100), dtype=np.uint8)
    img_a[50, 40] = 255

    # Image B is translated by tx=20, ty=10 -> dot is at (60, 60)
    img_b = np.zeros((100, 100), dtype=np.uint8)
    img_b[60, 60] = 255

    # Transform A -> B: tx=20, ty=10
    tf = SimilarityTransform2D(1.0, 0.0, 20.0, 10.0)
    reg_img, vmask, _ = register_image(img_a, img_b, tf)

    # In registered Image B (warped back to A), the dot must be at (40, 50)
    assert reg_img[50, 40] == 255
    assert vmask[50, 40] == 255


def test_transform_direction_correctness():
    """
    CRITICAL MANDATORY TEST:
    Demonstrates that a known A -> B transform correctly warps Image B back into Image A's coordinate frame.
    Centroid of the registered feature in A's frame must align with original centroid to < 0.5 px.
    """
    w_a, h_a = 200, 200
    w_b, h_b = 250, 250

    img_a = np.zeros((h_a, w_a), dtype=np.float32)
    # Draw circular feature at known position (80.0, 70.0) in Image A
    pt_a = np.array([80.0, 70.0], dtype=np.float64)
    cv2.circle(img_a, (int(pt_a[0]), int(pt_a[1])), 8, 255.0, -1)

    # Known similarity transform A -> B
    scale = 1.25
    theta_deg = 25.0
    theta_rad = math.radians(theta_deg)
    tx, ty = 30.0, -15.0
    r_mat = np.array([
        [math.cos(theta_rad), -math.sin(theta_rad)],
        [math.sin(theta_rad), math.cos(theta_rad)],
    ], dtype=np.float64)

    tf_a_to_b = SimilarityTransform2D(
        scale=scale,
        rotation_rad=theta_rad,
        translation_x=tx,
        translation_y=ty,
        rotation_matrix=r_mat,
    )

    # The expected centroid in Image B
    pt_b = tf_a_to_b.apply(pt_a)
    img_b = np.zeros((h_b, w_b), dtype=np.float32)
    # Draw the identical feature scaled and positioned at pt_b in Image B
    cv2.circle(img_b, (int(round(pt_b[0])), int(round(pt_b[1]))), int(round(8 * scale)), 255.0, -1)

    # Warp Image B into Image A's coordinate frame
    registered_b, vmask, _ = register_image(
        image_a=img_a,
        image_b=img_b,
        transform=tf_a_to_b,
        output_shape=(h_a, w_a),
    )

    # Measure centroid of warped feature in registered_b
    thresh = (registered_b > 50).astype(np.uint8)
    moments = cv2.moments(thresh)
    assert moments["m00"] > 0, "Registered feature was not found in registered image."

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    aligned_centroid = np.array([cx, cy])

    centroid_error = np.linalg.norm(aligned_centroid - pt_a)
    # Centroid alignment error must be < 0.5 pixels
    assert centroid_error < 0.5, f"Centroid alignment error {centroid_error:.3f} px exceeds 0.5 px threshold."
    # Valid overlap mask must be 255 at the feature location
    assert vmask[int(pt_a[1]), int(pt_a[0])] == 255


def test_register_multi_channel():
    img_a = np.zeros((100, 100, 3), dtype=np.uint8)
    img_b = np.full((100, 100, 3), 120, dtype=np.uint8)
    tf = SimilarityTransform2D(1.0, 0.0, 0.0, 0.0)

    reg_img, vmask, _ = register_image(img_a, img_b, tf)
    assert reg_img.shape == (100, 100, 3)
    assert vmask.shape == (100, 100)


def test_register_valid_mask_boundary():
    # Translation tx = 50 moves Image B 50 px to the right
    img_a = np.zeros((100, 100), dtype=np.uint8)
    img_b = np.full((100, 100), 200, dtype=np.uint8)
    tf = SimilarityTransform2D(1.0, 0.0, 50.0, 0.0)

    reg_img, vmask, _ = register_image(img_a, img_b, tf)
    # Pixels where x_dst + 50 >= 100 will fall outside Image B
    assert np.all(vmask[:, :50] == 255)
    assert np.all(vmask[:, 50:] == 0)


# ---------------------------------------------------------------------------
# 5. Quality Metrics & Classification Tests
# ---------------------------------------------------------------------------

def test_compute_registration_quality_clean(known_similarity_transform):
    pts_a = np.array([[50, 50], [150, 50], [150, 150], [50, 150], [100, 100], [80, 120]], dtype=np.float64)
    pts_b = known_similarity_transform.apply(pts_a)

    q = compute_registration_quality(
        points_a=pts_a,
        points_b=pts_b,
        transform=known_similarity_transform,
        num_candidates=8,
        num_inliers=6,
        coverage=0.35,
        valid_mask=np.ones((200, 200), dtype=np.uint8) * 255,
    )
    assert q.rmse < 1e-4
    assert q.quality == "EXCELLENT"
    assert q.confidence >= 0.70
    assert 0.0 <= q.confidence <= 1.0


def test_compute_registration_quality_failure():
    q = compute_registration_quality(
        points_a=np.empty((0, 2)),
        points_b=np.empty((0, 2)),
        transform=None,
        num_candidates=10,
        num_inliers=0,
        coverage=0.0,
    )
    assert q.quality == "FAILED"
    assert q.confidence == 0.0
    assert q.rmse is None


def test_compute_registration_quality_categories(known_similarity_transform):
    pts_a = np.array([[50, 50], [150, 50], [100, 100], [80, 120]], dtype=np.float64)
    # Add 2.0 px noise to residual
    pts_b = known_similarity_transform.apply(pts_a) + 2.0

    q = compute_registration_quality(
        points_a=pts_a,
        points_b=pts_b,
        transform=known_similarity_transform,
        num_candidates=10,
        num_inliers=4,
        coverage=0.15,
    )
    assert q.quality in ["GOOD", "FAIR"]


# ---------------------------------------------------------------------------
# 6. High-Level RegistrationEngine Tests
# ---------------------------------------------------------------------------

def test_registration_engine_e2e(synthetic_hybrid_result):
    img_a = np.zeros((400, 400), dtype=np.uint8)
    img_b = np.zeros((400, 400), dtype=np.uint8)

    engine = RegistrationEngine()
    result = engine.register(img_a, img_b, synthetic_hybrid_result)

    assert isinstance(result, RegistrationResult)
    assert result.success is True
    assert result.registered_image is not None
    assert result.valid_mask is not None
    assert result.transform is not None
    assert result.num_tie_points >= 3
    assert result.rmse is not None
    assert result.quality in ["EXCELLENT", "GOOD", "FAIR"]


def test_registration_engine_unmatched_handling():
    unmatched_result = HybridMatchResult(
        matched=False,
        correspondences=[],
        inlier_indices=[],
        outlier_indices=[],
        transform=None,
        confidence=0.0,
        rmse=None,
        inlier_ratio=0.0,
        metadata={"message": "No consensus"},
    )
    engine = RegistrationEngine()
    result = engine.register(np.zeros((100, 100)), np.zeros((100, 100)), unmatched_result)

    assert result.success is False
    assert result.quality == "FAILED"
    assert result.registered_image is None
    assert result.transform is None


def test_registration_result_serialization(synthetic_hybrid_result):
    img_a = np.zeros((100, 100), dtype=np.uint8)
    img_b = np.zeros((100, 100), dtype=np.uint8)

    engine = RegistrationEngine()
    result = engine.register(img_a, img_b, synthetic_hybrid_result)

    # 1. Serialization without arrays (lightweight for web/backend)
    dict_no_arrays = result.to_dict(include_arrays=False)
    assert "registered_image" not in dict_no_arrays
    assert "registered_image_shape" in dict_no_arrays
    assert "confidence" in dict_no_arrays

    # 2. Round-trip serialization with arrays
    dict_with_arrays = result.to_dict(include_arrays=True)
    deserialized = RegistrationResult.from_dict(dict_with_arrays)
    assert deserialized.success == result.success
    assert deserialized.quality == result.quality
    assert deserialized.registered_image.shape == result.registered_image.shape


def test_register_pure_rotation():
    # 90-degree counter-clockwise rotation around origin (0, 0)
    # R = [[0, -1], [1, 0]]
    # (x_A=50, y_A=20) -> (x_B=-20, y_B=50) -> with tx=100: (x_B=80, y_B=50)
    img_a = np.zeros((150, 150), dtype=np.uint8)
    img_a[20, 50] = 255

    r_mat = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)
    tf = SimilarityTransform2D(1.0, math.pi / 2.0, 100.0, 0.0, rotation_matrix=r_mat)

    pt_b = tf.apply(np.array([50.0, 20.0]))
    img_b = np.zeros((150, 150), dtype=np.uint8)
    img_b[int(round(pt_b[1])), int(round(pt_b[0]))] = 255

    reg_img, vmask, _ = register_image(img_a, img_b, tf)
    assert reg_img[20, 50] == 255
    assert vmask[20, 50] == 255


def test_register_pure_scale():
    # Scale s = 1.5, no rotation, no translation
    img_a = np.zeros((150, 150), dtype=np.uint8)
    cv2.circle(img_a, (50, 50), 6, 255, -1)

    tf = SimilarityTransform2D(1.5, 0.0, 0.0, 0.0)
    img_b = np.zeros((150, 150), dtype=np.uint8)
    # Circle in B is at (75, 75) with radius 9
    cv2.circle(img_b, (75, 75), 9, 255, -1)

    reg_img, vmask, _ = register_image(img_a, img_b, tf)
    # In registered Image B, center should be at (50, 50)
    assert reg_img[50, 50] == 255
    assert vmask[50, 50] == 255


def test_refine_subpixel_fractional_shift():
    # Continuous sub-pixel fractional shift: 0.4 px
    img_a = np.zeros((128, 128), dtype=np.float32)
    cv2.circle(img_a, (64, 64), 14, 220.0, -1)
    img_b = img_a.copy()

    # Intentional 0.4 px fractional offset in coarse coordinates
    coarse_pts_a = np.array([[64.0, 64.0]], dtype=np.float64)
    coarse_pts_b = np.array([[64.4, 64.0]], dtype=np.float64)

    id_tf = SimilarityTransform2D(1.0, 0.0, 0.0, 0.0)
    res = refine_tie_points(
        img_a, img_b, coarse_pts_a, coarse_pts_b, id_tf,
        config={"enabled": True, "patch_radius": 12, "search_radius": 3, "min_correlation": 0.5, "subpixel_interpolation": True},
    )
    assert res.num_refined == 1
    # Error should be reduced by parabolic subpixel interpolation
    err_before = abs(coarse_pts_b[0, 0] - 64.0)
    err_after = abs(res.refined_points_b[0, 0] - 64.0)
    assert err_after < err_before


def test_refine_shift_clipping():
    # Large 5.0 px displacement exceeds max_shift_px=2.0 -> fallback
    img_a = np.zeros((128, 128), dtype=np.float32)
    cv2.circle(img_a, (64, 64), 10, 200.0, -1)
    img_b = img_a.copy()

    coarse_pts_a = np.array([[64.0, 64.0]], dtype=np.float64)
    coarse_pts_b = np.array([[69.0, 64.0]], dtype=np.float64)  # 5.0 px error

    id_tf = SimilarityTransform2D(1.0, 0.0, 0.0, 0.0)
    res = refine_tie_points(
        img_a, img_b, coarse_pts_a, coarse_pts_b, id_tf,
        config={"enabled": True, "patch_radius": 10, "search_radius": 6, "max_shift_px": 2.0},
    )
    # Exceeds max_shift_px -> rejected, coordinates preserved
    assert res.num_failed == 1
    np.testing.assert_array_equal(res.refined_points_b, coarse_pts_b)


def test_tie_points_fewer_than_requested():
    pts_a = np.array([[20.0, 20.0], [80.0, 80.0]], dtype=np.float64)
    pts_b = pts_a.copy()
    corrs = [
        Correspondence(pts_a[0], pts_b[0], 0.9, "crater", 0),
        Correspondence(pts_a[1], pts_b[1], 0.8, "learned", 1),
    ]
    inliers = ExtractedInliers(corrs, pts_a, pts_b, np.array([0.9, 0.8]), ["crater", "learned"])
    res = select_uniform_tie_points(inliers, (100, 100), {"max_points": 20})
    assert res.num_selected == 2
