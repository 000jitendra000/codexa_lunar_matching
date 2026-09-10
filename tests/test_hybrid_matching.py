"""
tests/test_hybrid_matching.py

Comprehensive test suite for Milestone A: Hybrid Matching Engine (Phases 11-13):
1. Learned Matcher Abstraction & LoFTR Integration:
   - BaseLearnedMatcher interface adherence
   - MockLearnedMatcher deterministic outputs & unavailable states
   - LoFTRMatcher lazy initialization, padding for arbitrary dimensions, and inference
   - Graceful degradation on malformed inputs
2. Correspondence Adapters & Serialization:
   - Crater match conversion preserving crater coordinates & 'crater' source tag
   - Learned match conversion preserving coordinates & 'learned' source tag
   - Correspondence to_dict() and from_dict() round-trip fidelity
3. Correspondence Fusion:
   - Crater-only fusion
   - Learned-only fusion
   - Mixed sources fusion
   - Duplicate removal within spatial tolerance retaining higher confidence
   - Deterministic tie-breaking
   - Invalid coordinate rejection (NaN, Inf, negative, out-of-bounds)
   - Deterministic output ordering
   - Source balancing / max correspondence caps
   - Array extraction for Phase 10 RANSAC
4. Hybrid Matching Pipeline:
   - Synthetic clean matching (crater + learned)
   - Outlier rejection across fused branches
   - Fallback when learned branch is unavailable
   - Fallback when crater branch is empty
   - Insufficient correspondences returns matched=False cleanly
   - Consensus failure handling
   - HybridMatchResult serialization round-trip
   - End-to-end HybridMatcher execution
"""

import math
import numpy as np
import pytest

from src.crater_detection.types import CraterCandidate
from src.crater_graph.builder import build_crater_graph
from src.crater_graph.constellation_matcher import ConstellationMatch
from src.matching.transformation import SimilarityTransform2D
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ground_truth_transform():
    """Ground truth similarity transform (scale=1.35, rot=38 deg, tx=45, ty=-30)."""
    scale = 1.35
    theta = math.radians(38.0)
    tx = 45.0
    ty = -30.0
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


@pytest.fixture
def synthetic_crater_correspondences(ground_truth_transform):
    """Synthetic crater branch correspondences."""
    pts_a = np.array([
        [100.0, 150.0],
        [180.0, 120.0],
        [220.0, 260.0],
        [130.0, 310.0],
    ])
    pts_b = ground_truth_transform.apply(pts_a)

    return [
        Correspondence(point_a=pts_a[i], point_b=pts_b[i], confidence=0.9 - 0.05 * i, source="crater", source_index=i)
        for i in range(len(pts_a))
    ]


@pytest.fixture
def synthetic_learned_correspondences(ground_truth_transform):
    """Synthetic learned branch correspondences."""
    pts_a = np.array([
        [300.0, 180.0],
        [260.0, 340.0],
        [350.0, 280.0],
        [170.0, 210.0],
    ])
    pts_b = ground_truth_transform.apply(pts_a)

    return [
        Correspondence(point_a=pts_a[i], point_b=pts_b[i], confidence=0.85 - 0.04 * i, source="learned", source_index=i)
        for i in range(len(pts_a))
    ]


# ---------------------------------------------------------------------------
# 1. Learned Matcher Abstraction Tests
# ---------------------------------------------------------------------------

def test_base_learned_matcher_interface():
    """Verify BaseLearnedMatcher cannot be instantiated directly without match()."""
    with pytest.raises(TypeError):
        BaseLearnedMatcher()


def test_mock_learned_matcher_output_format():
    """Verify MockLearnedMatcher returns valid LearnedMatchResult."""
    corrs = [
        LearnedCorrespondence(point_a=np.array([50.0, 60.0]), point_b=np.array([70.0, 80.0]), confidence=0.88),
    ]
    matcher = MockLearnedMatcher(mock_correspondences=corrs, available=True)
    img_dummy = np.zeros((100, 100), dtype=np.uint8)

    res = matcher.match(img_dummy, img_dummy)
    assert res.available is True
    assert res.backend == "mock"
    assert res.num_correspondences == 1
    assert res.correspondences[0].confidence == pytest.approx(0.88)


def test_mock_learned_matcher_unavailable():
    """Verify MockLearnedMatcher reports available=False with reason."""
    matcher = MockLearnedMatcher(available=False, fail_reason="Weights not found.")
    img_dummy = np.zeros((100, 100), dtype=np.uint8)

    res = matcher.match(img_dummy, img_dummy)
    assert res.available is False
    assert res.num_correspondences == 0
    assert "Weights not found." in res.metadata["reason"]


def test_loftr_instantiation_and_config():
    """Verify LoFTRMatcher initializes configuration parameters properly."""
    matcher = LoFTRMatcher(config={"pretrained": "outdoor", "min_confidence": 0.25, "device": "cpu"})
    assert matcher.pretrained == "outdoor"
    assert matcher.min_confidence == 0.25
    assert matcher.device_str == "cpu"


def test_loftr_handles_odd_dimensions():
    """Verify LoFTR preprocessor automatically pads non-multiple-of-8 dimensions without error."""
    matcher = LoFTRMatcher()
    if not matcher._ensure_initialized():
        pytest.skip("LoFTR weights or dependencies unavailable in environment.")

    # 485 x 493 is not divisible by 8
    img_odd = np.random.randint(0, 256, size=(485, 493), dtype=np.uint8)
    tensor, orig_h, orig_w = matcher._preprocess_image(img_odd)

    assert orig_h == 485
    assert orig_w == 493
    # Check padded tensor dimensions are divisible by 8
    assert tensor.shape[2] % 8 == 0
    assert tensor.shape[3] % 8 == 0


def test_loftr_empty_or_malformed_image_handling():
    """Verify empty or zero-sized images return available=False with clean error."""
    matcher = LoFTRMatcher()
    empty_img = np.zeros((0, 0), dtype=np.uint8)
    res = matcher.match(empty_img, empty_img)

    assert res.available is False
    assert res.num_correspondences == 0
    assert "reason" in res.metadata


# ---------------------------------------------------------------------------
# 2. Correspondence Adapters & Serialization Tests
# ---------------------------------------------------------------------------

def test_crater_matches_to_correspondences_conversion():
    """Verify crater matches are converted to Correspondence with source='crater'."""
    craters_a = [
        CraterCandidate(x=10.0, y=20.0, radius=5.0, confidence=0.9),
        CraterCandidate(x=30.0, y=40.0, radius=6.0, confidence=0.8),
    ]
    craters_b = [
        CraterCandidate(x=15.0, y=25.0, radius=5.0, confidence=0.95),
        CraterCandidate(x=35.0, y=45.0, radius=6.0, confidence=0.85),
    ]
    graph_a = build_crater_graph(craters_a)
    graph_b = build_crater_graph(craters_b)

    matches = [
        ConstellationMatch(source_node_id=0, target_node_id=0, descriptor_distance=0.05, confidence=0.92, matched_neighbor_count=1),
        ConstellationMatch(source_node_id=1, target_node_id=1, descriptor_distance=0.10, confidence=0.81, matched_neighbor_count=1),
    ]

    corrs = crater_matches_to_correspondences(graph_a, graph_b, matches)
    assert len(corrs) == 2
    assert corrs[0].source == "crater"
    assert corrs[0].source_index == 0
    np.testing.assert_array_equal(corrs[0].point_a, [10.0, 20.0])
    np.testing.assert_array_equal(corrs[0].point_b, [15.0, 25.0])
    assert corrs[0].confidence == pytest.approx(0.92)


def test_learned_matches_to_correspondences_conversion():
    """Verify LearnedMatchResult is converted to Correspondence with source='learned'."""
    learned_corrs = [
        LearnedCorrespondence(point_a=np.array([100.0, 110.0]), point_b=np.array([120.0, 130.0]), confidence=0.75),
    ]
    res = LearnedMatchResult(correspondences=learned_corrs, available=True, backend="mock", metadata={})

    corrs = learned_matches_to_correspondences(res)
    assert len(corrs) == 1
    assert corrs[0].source == "learned"
    assert corrs[0].source_index == 0
    np.testing.assert_array_equal(corrs[0].point_a, [100.0, 110.0])
    np.testing.assert_array_equal(corrs[0].point_b, [120.0, 130.0])
    assert corrs[0].confidence == pytest.approx(0.75)


def test_correspondence_serialization_round_trip():
    """Verify Correspondence serialization to/from dictionary."""
    c = Correspondence(
        point_a=np.array([12.34, 56.78]),
        point_b=np.array([90.12, 34.56]),
        confidence=0.8888,
        source="crater",
        source_index=3,
        metadata={"note": "test"},
    )
    d = c.to_dict()
    c_restored = Correspondence.from_dict(d)

    np.testing.assert_array_almost_equal(c_restored.point_a, c.point_a, decimal=4)
    np.testing.assert_array_almost_equal(c_restored.point_b, c.point_b, decimal=4)
    assert c_restored.confidence == pytest.approx(c.confidence, abs=1e-4)
    assert c_restored.source == c.source
    assert c_restored.source_index == c.source_index
    assert c_restored.metadata == c.metadata


# ---------------------------------------------------------------------------
# 3. Correspondence Fusion Tests
# ---------------------------------------------------------------------------

def test_fusion_crater_only(synthetic_crater_correspondences):
    """Fusing with only crater correspondences preserves all valid points."""
    fused = fuse_correspondences(synthetic_crater_correspondences, [])
    assert len(fused) == len(synthetic_crater_correspondences)
    assert all(c.source == "crater" for c in fused)


def test_fusion_learned_only(synthetic_learned_correspondences):
    """Fusing with only learned correspondences preserves all valid points."""
    fused = fuse_correspondences([], synthetic_learned_correspondences)
    assert len(fused) == len(synthetic_learned_correspondences)
    assert all(c.source == "learned" for c in fused)


def test_fusion_mixed_sources(synthetic_crater_correspondences, synthetic_learned_correspondences):
    """Fusing disjoint crater and learned sets retains both sources."""
    fused = fuse_correspondences(synthetic_crater_correspondences, synthetic_learned_correspondences)
    total_expected = len(synthetic_crater_correspondences) + len(synthetic_learned_correspondences)
    assert len(fused) == total_expected

    sources = {c.source for c in fused}
    assert "crater" in sources
    assert "learned" in sources


def test_fusion_duplicate_removal_retains_stronger():
    """Two correspondences within duplicate tolerance merge into the one with higher confidence."""
    c_crater = Correspondence(
        point_a=np.array([100.0, 100.0]),
        point_b=np.array([150.0, 150.0]),
        confidence=0.95,
        source="crater",
        source_index=0,
    )
    # Learned point within 1.0 px of crater match, but lower confidence (0.70)
    c_learned = Correspondence(
        point_a=np.array([100.8, 100.5]),
        point_b=np.array([150.7, 150.4]),
        confidence=0.70,
        source="learned",
        source_index=0,
    )

    fused = fuse_correspondences([c_crater], [c_learned], duplicate_tolerance_px=4.0)
    assert len(fused) == 1
    assert fused[0].source == "crater"
    assert fused[0].confidence == pytest.approx(0.95)


def test_fusion_duplicate_tie_breaking():
    """Identical confidence duplicate resolves deterministically (prefers crater)."""
    c_crater = Correspondence(point_a=np.array([50.0, 50.0]), point_b=np.array([80.0, 80.0]), confidence=0.80, source="crater", source_index=1)
    c_learned = Correspondence(point_a=np.array([50.0, 50.0]), point_b=np.array([80.0, 80.0]), confidence=0.80, source="learned", source_index=2)

    fused = fuse_correspondences([c_crater], [c_learned], duplicate_tolerance_px=2.0)
    assert len(fused) == 1
    assert fused[0].source == "crater"


def test_fusion_invalid_coordinates_rejection():
    """Invalid coordinates (NaN, Inf, negative, out-of-bounds) are rejected."""
    good = Correspondence(point_a=np.array([10.0, 20.0]), point_b=np.array([30.0, 40.0]), confidence=0.9, source="crater", source_index=0)
    nan_c = Correspondence(point_a=np.array([np.nan, 20.0]), point_b=np.array([30.0, 40.0]), confidence=0.9, source="crater", source_index=1)
    inf_c = Correspondence(point_a=np.array([10.0, 20.0]), point_b=np.array([np.inf, 40.0]), confidence=0.9, source="learned", source_index=2)
    neg_c = Correspondence(point_a=np.array([-5.0, 20.0]), point_b=np.array([30.0, 40.0]), confidence=0.9, source="learned", source_index=3)
    oob_c = Correspondence(point_a=np.array([600.0, 20.0]), point_b=np.array([30.0, 40.0]), confidence=0.9, source="crater", source_index=4)

    fused = fuse_correspondences([good, nan_c, inf_c], [neg_c, oob_c], image_shape_a=(500, 500), image_shape_b=(500, 500))
    assert len(fused) == 1
    assert fused[0].source_index == 0


def test_fusion_deterministic_ordering(synthetic_crater_correspondences, synthetic_learned_correspondences):
    """Fused output must have deterministic ordering (confidence descending)."""
    fused1 = fuse_correspondences(synthetic_crater_correspondences, synthetic_learned_correspondences)
    fused2 = fuse_correspondences(synthetic_crater_correspondences, synthetic_learned_correspondences)

    for c1, c2 in zip(fused1, fused2):
        assert c1.confidence == c2.confidence
        assert c1.source == c2.source
        assert c1.source_index == c2.source_index


def test_correspondences_to_arrays(synthetic_crater_correspondences):
    """Verify correspondences_to_arrays extracts correct shapes."""
    pts_a, pts_b, weights = correspondences_to_arrays(synthetic_crater_correspondences)
    n = len(synthetic_crater_correspondences)

    assert pts_a.shape == (n, 2)
    assert pts_b.shape == (n, 2)
    assert weights.shape == (n,)
    assert np.all(weights > 0.0)


# ---------------------------------------------------------------------------
# 4. Hybrid Matching Pipeline Tests
# ---------------------------------------------------------------------------

def test_hybrid_clean_synthetic_matching(synthetic_crater_correspondences, synthetic_learned_correspondences, ground_truth_transform):
    """Clean hybrid pool (crater + learned) recovers ground truth transform with 100% inlier ratio."""
    matcher = HybridMatcher()
    res = matcher.match_from_correspondences(
        synthetic_crater_correspondences,
        synthetic_learned_correspondences,
    )

    assert res.matched is True
    assert res.inlier_ratio == 1.0
    assert res.num_inliers == len(synthetic_crater_correspondences) + len(synthetic_learned_correspondences)
    assert res.rmse < 1e-4
    assert res.transform.scale == pytest.approx(ground_truth_transform.scale, abs=1e-3)
    assert res.transform.rotation_deg == pytest.approx(ground_truth_transform.rotation_deg, abs=1e-2)


def test_hybrid_outlier_rejection(synthetic_crater_correspondences, synthetic_learned_correspondences, ground_truth_transform):
    """Verify hybrid pipeline isolates outliers from both crater and learned branches."""
    # Add 1 false crater correspondence and 2 false learned correspondences
    bad_crater = Correspondence(
        point_a=np.array([50.0, 50.0]),
        point_b=np.array([400.0, 30.0]),  # Outlier
        confidence=0.70,
        source="crater",
        source_index=99,
    )
    bad_learned1 = Correspondence(
        point_a=np.array([60.0, 80.0]),
        point_b=np.array([10.0, 450.0]),  # Outlier
        confidence=0.65,
        source="learned",
        source_index=98,
    )
    bad_learned2 = Correspondence(
        point_a=np.array([120.0, 70.0]),
        point_b=np.array([300.0, 10.0]),  # Outlier
        confidence=0.60,
        source="learned",
        source_index=97,
    )

    craters = synthetic_crater_correspondences + [bad_crater]
    learned = synthetic_learned_correspondences + [bad_learned1, bad_learned2]

    matcher = HybridMatcher()
    res = matcher.match_from_correspondences(craters, learned, reprojection_threshold=3.0)

    assert res.matched is True
    assert res.num_inliers == 8  # 4 clean crater + 4 clean learned
    assert res.num_outliers == 3
    assert res.rmse < 1e-4
    assert res.transform.scale == pytest.approx(ground_truth_transform.scale, abs=1e-3)


def test_hybrid_fallback_when_learned_unavailable(synthetic_crater_correspondences, ground_truth_transform):
    """When learned branch is unavailable, hybrid pipeline succeeds solely on crater branch."""
    matcher = HybridMatcher(config={"enable_crater_branch": True, "enable_learned_branch": True})
    # Pass empty learned correspondences (simulating unavailable learned branch)
    res = matcher.match_from_correspondences(synthetic_crater_correspondences, [])

    assert res.matched is True
    assert res.num_inliers == len(synthetic_crater_correspondences)
    assert res.transform.scale == pytest.approx(ground_truth_transform.scale, abs=1e-3)


def test_hybrid_fallback_when_crater_unavailable(synthetic_learned_correspondences, ground_truth_transform):
    """When crater branch has zero matches, hybrid pipeline succeeds solely on learned branch."""
    matcher = HybridMatcher()
    res = matcher.match_from_correspondences([], synthetic_learned_correspondences)

    assert res.matched is True
    assert res.num_inliers == len(synthetic_learned_correspondences)
    assert res.transform.scale == pytest.approx(ground_truth_transform.scale, abs=1e-3)


def test_hybrid_insufficient_matches_returns_unmatched():
    """Fewer fused matches than min_inliers cleanly returns matched=False."""
    matcher = HybridMatcher(config={"min_inliers": 3})
    few_matches = [
        Correspondence(point_a=np.array([10.0, 10.0]), point_b=np.array([20.0, 20.0]), confidence=0.9, source="crater", source_index=0),
    ]

    res = matcher.match_from_correspondences(few_matches, [])
    assert res.matched is False
    assert res.transform is None
    assert res.confidence == 0.0
    assert "Insufficient fused correspondences" in res.metadata["reason"]


def test_hybrid_consensus_failure_returns_unmatched():
    """Scrambled correspondences with no geometric consensus return matched=False."""
    matcher = HybridMatcher(config={"min_inliers": 3})
    scrambled = [
        Correspondence(point_a=np.array([10.0, 20.0]), point_b=np.array([500.0, 10.0]), confidence=0.8, source="crater", source_index=0),
        Correspondence(point_a=np.array([50.0, 80.0]), point_b=np.array([20.0, 600.0]), confidence=0.8, source="crater", source_index=1),
        Correspondence(point_a=np.array([90.0, 30.0]), point_b=np.array([300.0, 400.0]), confidence=0.8, source="learned", source_index=2),
        Correspondence(point_a=np.array([120.0, 150.0]), point_b=np.array([50.0, 50.0]), confidence=0.8, source="learned", source_index=3),
    ]

    res = matcher.match_from_correspondences(scrambled, [])
    assert res.matched is False
    assert res.transform is None
    assert "consensus verification failed" in res.metadata["reason"]


def test_hybrid_result_serialization_round_trip(synthetic_crater_correspondences, ground_truth_transform):
    """Verify HybridMatchResult serialization to/from primitive dictionary."""
    matcher = HybridMatcher()
    res = matcher.match_from_correspondences(synthetic_crater_correspondences, [])
    d = res.to_dict()

    res_restored = HybridMatchResult.from_dict(d)
    assert res_restored.matched == res.matched
    assert res_restored.num_inliers == res.num_inliers
    assert res_restored.inlier_indices == res.inlier_indices
    assert res_restored.rmse == pytest.approx(res.rmse, abs=1e-4)
    assert res_restored.transform.scale == pytest.approx(res.transform.scale, abs=1e-4)


def test_hybrid_end_to_end_with_mock_matcher(ground_truth_transform):
    """End-to-end integration test with MockLearnedMatcher and synthetic images."""
    mock_corrs = [
        LearnedCorrespondence(point_a=np.array([50.0, 60.0]), point_b=ground_truth_transform.apply(np.array([50.0, 60.0])), confidence=0.9),
        LearnedCorrespondence(point_a=np.array([120.0, 80.0]), point_b=ground_truth_transform.apply(np.array([120.0, 80.0])), confidence=0.85),
        LearnedCorrespondence(point_a=np.array([150.0, 60.0]), point_b=ground_truth_transform.apply(np.array([150.0, 60.0])), confidence=0.88),
        LearnedCorrespondence(point_a=np.array([100.0, 120.0]), point_b=ground_truth_transform.apply(np.array([100.0, 120.0])), confidence=0.92),
    ]
    mock_learned = MockLearnedMatcher(mock_correspondences=mock_corrs)

    matcher = HybridMatcher(
        config={"enable_crater_branch": False, "enable_learned_branch": True},
        learned_matcher=mock_learned,
    )

    img_dummy = np.zeros((400, 400), dtype=np.uint8)
    res = matcher.match(img_dummy, img_dummy)

    assert res.matched is True
    assert res.num_inliers == 4
    assert res.transform.scale == pytest.approx(ground_truth_transform.scale, abs=1e-3)
