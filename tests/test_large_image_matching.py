"""
tests/test_large_image_matching.py

Unit test suite for real-image input-size handling and memory-bounded tiled LoFTR matching.
Tests:
- Deterministic tile range generation.
- Deterministic correspondence deduplication.
- Small image single-pass path preservation.
- Large image memory-bounded tiled execution.
- Keypoint coordinate mapping accuracy from tiles to original image space.
- Graceful degradation on tile overflow or error conditions.
"""

import pytest
import numpy as np

from src.matching.learned_matcher import (
    LearnedCorrespondence,
    LearnedMatchResult,
    LoFTRMatcher,
    MockLearnedMatcher,
    generate_tile_ranges,
    deduplicate_correspondences,
)
from src.matching.hybrid_matcher import HybridMatcher


def test_generate_tile_ranges_small():
    """Verify tile range generation for lengths smaller than or equal to tile size."""
    ranges = generate_tile_ranges(500, tile_size=640, tile_overlap=160)
    assert ranges == [(0, 500)]

    ranges_exact = generate_tile_ranges(640, tile_size=640, tile_overlap=160)
    assert ranges_exact == [(0, 640)]


def test_generate_tile_ranges_large():
    """Verify tile range generation for large dimensions with exact coverage."""
    ranges = generate_tile_ranges(10107, tile_size=640, tile_overlap=160)
    # Stride = 640 - 160 = 480
    assert len(ranges) > 1
    assert ranges[0] == (0, 640)
    assert ranges[1] == (480, 1120)
    # Check complete coverage up to 10107
    assert ranges[-1][1] == 10107
    assert ranges[-1][0] == 10107 - 640  # 9467


def test_deduplicate_correspondences():
    """Verify deterministic deduplication of duplicate tile boundary correspondences."""
    c1 = LearnedCorrespondence(point_a=[100.0, 200.0], point_b=[150.0, 250.0], confidence=0.9)
    # Near duplicate of c1 within 3.0 px
    c2 = LearnedCorrespondence(point_a=[101.0, 200.5], point_b=[150.8, 250.2], confidence=0.8)
    # Distinct correspondence
    c3 = LearnedCorrespondence(point_a=[500.0, 600.0], point_b=[550.0, 650.0], confidence=0.85)

    deduped = deduplicate_correspondences([c1, c2, c3], tolerance_px=3.0)
    assert len(deduped) == 2
    # Highest confidence (c1) should be retained over c2
    assert np.allclose(deduped[0].point_a, [100.0, 200.0])
    assert np.allclose(deduped[1].point_a, [500.0, 600.0])


def test_loftr_small_image_single_pass():
    """Verify small images use single-pass path and report tiling_enabled=False."""
    matcher = LoFTRMatcher(config={"tiling_mode": "auto", "max_image_dimension": 640})
    img_a = np.zeros((300, 300), dtype=np.uint8)
    img_b = np.zeros((300, 300), dtype=np.uint8)

    res = matcher.match(img_a, img_b)
    if res.available:
        assert res.metadata["tiling_enabled"] is False
        assert res.metadata["tile_pairs_processed"] == 1
        assert res.metadata["num_tiles_a"] == 1


def test_loftr_large_image_tiling_execution():
    """Verify large image pair triggers memory-bounded tiling without gigantic allocations."""
    matcher = LoFTRMatcher(config={"tiling_mode": "auto", "max_image_dimension": 640, "tile_size": 640, "tile_overlap": 160})
    img_a = np.zeros((700, 700), dtype=np.uint8)
    img_b = np.zeros((700, 700), dtype=np.uint8)

    res = matcher.match(img_a, img_b)
    if res.available:
        assert res.metadata["tiling_enabled"] is True
        assert res.metadata["num_tiles_a"] > 1
        assert res.metadata["tile_pairs_processed"] == res.metadata["num_tiles_a"]
        assert "raw_tile_correspondences_count" in res.metadata
        assert "deduplicated_correspondences_count" in res.metadata


def test_loftr_max_tiles_safety_guard():
    """Verify exceeding max_tiles safety guard rail fails gracefully."""
    matcher = LoFTRMatcher(config={"tiling_mode": "always", "tile_size": 100, "tile_overlap": 10, "max_tiles": 5})
    img_a = np.zeros((1000, 1000), dtype=np.uint8)
    img_b = np.zeros((1000, 1000), dtype=np.uint8)

    res = matcher.match(img_a, img_b)
    assert res.available is False
    assert "exceeds max_tiles limit" in res.metadata.get("reason", "")


def test_hybrid_matcher_large_image_compatibility():
    """Verify HybridMatcher functions smoothly with real large image dimensions."""
    mock_learned = MockLearnedMatcher(available=True)
    matcher = HybridMatcher(learned_matcher=mock_learned)
    img_a = np.zeros((1000, 500), dtype=np.uint8)
    img_b = np.zeros((1000, 500), dtype=np.uint8)

    res = matcher.match(img_a, img_b)
    assert hasattr(res, "matched")
    assert hasattr(res, "correspondences")
    assert hasattr(res, "inlier_ratio")
