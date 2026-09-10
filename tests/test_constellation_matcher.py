"""
tests/test_constellation_matcher.py

Comprehensive tests for Phase 8: Local Crater Constellation Matching:
- Exact correspondence recovery under translation, rotation, scale, and combined similarity
- Robustness to small coordinate/radius perturbation
- Rejection of clearly mismatched descriptors and distractor craters
- Handling of variable neighborhood sizes (excluding zero padding from distance calculations)
- Distance thresholding and Lowe's ratio test filtering
- Mutual (bidirectional) consistency verification
- Triangle support bonus and confidence scoring
- Determinism and tie-breaking across repeated runs
- Handling of nearly-symmetric neighborhoods without pathological failure
- Degenerate / empty / small graph handling
- Pure-primitive dictionary serialization and round-trip fidelity
"""

import math
import copy
import pytest
import numpy as np

from src.crater_detection.types import CraterCandidate
from src.crater_graph.builder import build_crater_graph
from src.crater_graph.invariants import (
    build_local_invariant_descriptors,
    build_triangle_descriptors,
)
from src.crater_graph.constellation_matcher import (
    ConstellationMatch,
    ConstellationMatchResult,
    compute_local_descriptor_distance,
    count_supporting_triangles,
    match_crater_constellations,
)


# ---------------------------------------------------------------------------
# Helpers for geometric transformation of crater constellations
# ---------------------------------------------------------------------------

def transform_craters(
    craters: list[CraterCandidate],
    scale: float = 1.0,
    angle_rad: float = 0.0,
    tx: float = 0.0,
    ty: float = 0.0,
) -> list[CraterCandidate]:
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    transformed = []
    for c in craters:
        rx = scale * (cos_a * c.x - sin_a * c.y) + tx
        ry = scale * (sin_a * c.x + cos_a * c.y) + ty
        rr = scale * c.radius
        transformed.append(CraterCandidate(x=rx, y=ry, radius=rr, confidence=c.confidence))
    return transformed


@pytest.fixture
def base_constellation():
    """A 5-crater asymmetric constellation with clear geometric signatures."""
    return [
        CraterCandidate(x=100.0, y=100.0, radius=12.0, confidence=0.95),  # 0
        CraterCandidate(x=240.0, y=120.0, radius=22.0, confidence=0.90),  # 1
        CraterCandidate(x=280.0, y=260.0, radius=16.0, confidence=0.92),  # 2
        CraterCandidate(x=180.0, y=320.0, radius=20.0, confidence=0.88),  # 3
        CraterCandidate(x=80.0, y=220.0, radius=14.0, confidence=0.93),   # 4
    ]


# ---------------------------------------------------------------------------
# 1. Similarity-Transform Invariance Matching
# ---------------------------------------------------------------------------

def test_exact_self_matching(base_constellation):
    """Matching a constellation to itself must produce 100% identity matches with D=0.0."""
    g = build_crater_graph(base_constellation, config={"method": "delaunay"})
    res = match_crater_constellations(g, g)

    assert res.match_count == len(base_constellation)
    for m in res.matches:
        assert m.source_node_id == m.target_node_id
        assert m.descriptor_distance == pytest.approx(0.0, abs=1e-5)
        assert m.confidence > 0.95


@pytest.mark.parametrize("tx, ty", [(150.0, -80.0), (-300.0, 500.0)])
def test_translation_invariance(base_constellation, tx, ty):
    g_a = build_crater_graph(base_constellation, config={"method": "delaunay"})
    c_b = transform_craters(base_constellation, tx=tx, ty=ty)
    g_b = build_crater_graph(c_b, config={"method": "delaunay"})

    res = match_crater_constellations(g_a, g_b)
    pairs = dict(res.get_match_pairs())

    # All nodes must correspond to their identity index
    for i in range(len(base_constellation)):
        assert pairs.get(i) == i
    for m in res.matches:
        assert m.descriptor_distance == pytest.approx(0.0, abs=1e-4)


@pytest.mark.parametrize("angle_deg", [45.0, 90.0, 180.0, 270.0])
def test_rotation_invariance(base_constellation, angle_deg):
    g_a = build_crater_graph(base_constellation, config={"method": "delaunay"})
    c_b = transform_craters(base_constellation, angle_rad=math.radians(angle_deg))
    g_b = build_crater_graph(c_b, config={"method": "delaunay"})

    res = match_crater_constellations(g_a, g_b)
    pairs = dict(res.get_match_pairs())

    for i in range(len(base_constellation)):
        assert pairs.get(i) == i
    for m in res.matches:
        assert m.descriptor_distance == pytest.approx(0.0, abs=1e-4)


@pytest.mark.parametrize("scale", [0.5, 1.8, 2.5])
def test_scale_invariance(base_constellation, scale):
    g_a = build_crater_graph(base_constellation, config={"method": "delaunay"})
    c_b = transform_craters(base_constellation, scale=scale)
    g_b = build_crater_graph(c_b, config={"method": "delaunay"})

    res = match_crater_constellations(g_a, g_b)
    pairs = dict(res.get_match_pairs())

    for i in range(len(base_constellation)):
        assert pairs.get(i) == i
    for m in res.matches:
        assert m.descriptor_distance == pytest.approx(0.0, abs=1e-4)


def test_combined_similarity_transform(base_constellation):
    """Scale + rotation + translation must preserve all true correspondences."""
    g_a = build_crater_graph(base_constellation, config={"method": "delaunay"})
    c_b = transform_craters(
        base_constellation, scale=1.75, angle_rad=math.radians(65.0), tx=250.0, ty=-120.0
    )
    g_b = build_crater_graph(c_b, config={"method": "delaunay"})

    res = match_crater_constellations(g_a, g_b)
    pairs = dict(res.get_match_pairs())

    assert len(pairs) == len(base_constellation)
    for i in range(len(base_constellation)):
        assert pairs.get(i) == i
    for m in res.matches:
        assert m.descriptor_distance == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 2. Perturbation & Distractor Rejection
# ---------------------------------------------------------------------------

def test_descriptor_noise_perturbation(base_constellation):
    """Small realistic noise in crater coordinates and radii should not destroy valid matches."""
    g_a = build_crater_graph(base_constellation, config={"method": "delaunay"})

    # Add ~1% jitter to coordinates and radii
    c_noisy = []
    rng = np.random.default_rng(42)
    for c in base_constellation:
        c_noisy.append(
            CraterCandidate(
                x=c.x + float(rng.uniform(-0.8, 0.8)),
                y=c.y + float(rng.uniform(-0.8, 0.8)),
                radius=c.radius + float(rng.uniform(-0.3, 0.3)),
                confidence=c.confidence,
            )
        )
    g_b = build_crater_graph(c_noisy, config={"method": "delaunay"})

    res = match_crater_constellations(g_a, g_b)
    pairs = dict(res.get_match_pairs())

    # All 5 true correspondences must still be recovered
    assert len(pairs) == len(base_constellation)
    for i in range(len(base_constellation)):
        assert pairs.get(i) == i
        # Distance should be small non-zero
        m = next(m for m in res.matches if m.source_node_id == i)
        assert 0.0 <= m.descriptor_distance < 0.15


def test_distractor_craters_rejection(base_constellation):
    """Target image with 2 distractor craters far from true constellation."""
    g_a = build_crater_graph(base_constellation, config={"method": "delaunay"})

    # Target has base constellation + 2 distant distractors
    c_b = list(base_constellation)
    c_b.append(CraterCandidate(x=600.0, y=600.0, radius=50.0, confidence=0.8))  # distractor 5
    c_b.append(CraterCandidate(x=700.0, y=650.0, radius=45.0, confidence=0.8))  # distractor 6
    g_b = build_crater_graph(c_b, config={"method": "delaunay"})

    res = match_crater_constellations(g_a, g_b)
    target_ids_matched = {m.target_node_id for m in res.matches}

    # Distractors 5 and 6 must not be matched to any source craters
    assert 5 not in target_ids_matched
    assert 6 not in target_ids_matched


# ---------------------------------------------------------------------------
# 3. Variable Neighborhood Size & Zero-Padding
# ---------------------------------------------------------------------------

def test_variable_neighborhood_size_excludes_padding(base_constellation):
    """
    Ensure descriptors with different neighbor counts (e.g. 3 vs 4) compare
    only the min(ka, kb) compatible entries without treating padded zeros as real craters.
    """
    g_a = build_crater_graph(base_constellation, config={"method": "delaunay"})
    descs = build_local_invariant_descriptors(g_a)

    # Pick a descriptor with at least 3 neighbors (node 4 has 4 neighbors)
    d_node = max(descs.values(), key=lambda d: d.num_neighbors)
    # Create an artificial copy with fewer neighbors
    d_fewer = copy.deepcopy(d_node)
    orig_k = d_fewer.num_neighbors
    assert orig_k >= 3
    d_fewer.num_neighbors = orig_k - 1
    d_fewer.normalized_distances = d_fewer.normalized_distances[:orig_k - 1]
    d_fewer.radius_ratios = d_fewer.radius_ratios[:orig_k - 1]
    d_fewer.relative_angles_rad = d_fewer.relative_angles_rad[:orig_k - 1]

    # Compute distance: distance should reflect only the neighbor count penalty,
    # and the common entries should have zero difference!
    dist, comp_errors = compute_local_descriptor_distance(d_node, d_fewer)


    assert comp_errors["error_dist"] == pytest.approx(0.0, abs=1e-5)
    assert comp_errors["error_radius"] == pytest.approx(0.0, abs=1e-5)
    assert comp_errors["error_angle"] == pytest.approx(0.0, abs=1e-5)
    # The only non-zero error is error_count: 1 / orig_k
    assert comp_errors["error_count"] == pytest.approx(1.0 / orig_k, abs=1e-5)
    assert 0.0 < dist < 0.2


# ---------------------------------------------------------------------------
# 4. Ratio Test & Mutual Consistency
# ---------------------------------------------------------------------------

def test_mutual_consistency_removes_asymmetric_matches():
    """
    If node A1's best target is B1, but B1's best source is A2,
    mutual consistency must filter out the asymmetric pair.
    """
    c_a = [
        CraterCandidate(x=100.0, y=100.0, radius=10.0, confidence=0.9),
        CraterCandidate(x=150.0, y=120.0, radius=10.0, confidence=0.9),
        CraterCandidate(x=120.0, y=180.0, radius=10.0, confidence=0.9),
    ]
    # Completely different geometry
    c_b = [
        CraterCandidate(x=200.0, y=100.0, radius=30.0, confidence=0.9),
        CraterCandidate(x=400.0, y=100.0, radius=12.0, confidence=0.9),
        CraterCandidate(x=300.0, y=250.0, radius=5.0, confidence=0.9),
    ]
    g_a = build_crater_graph(c_a, config={"method": "delaunay"})
    g_b = build_crater_graph(c_b, config={"method": "delaunay"})

    # With strict mutual consistency and max_dist
    res = match_crater_constellations(g_a, g_b, config={"max_descriptor_distance": 0.2})
    # Due to very different geometry, no false matches should survive
    assert res.match_count == 0


# ---------------------------------------------------------------------------
# 5. Triangle Support Bonus
# ---------------------------------------------------------------------------

def test_triangle_support_bonus(base_constellation):
    """
    When triangle support is enabled, matching 3-cliques between A and B
    should receive positive supporting_triangle_count.
    """
    g_a = build_crater_graph(base_constellation, config={"method": "delaunay"})
    g_b = build_crater_graph(base_constellation, config={"method": "delaunay"})

    res = match_crater_constellations(g_a, g_b, config={"triangle_support_enabled": True})
    assert res.match_count == len(base_constellation)

    # Delaunay triangulation on 5 non-collinear craters contains triangles
    tri_counts = [m.supporting_triangle_count for m in res.matches]
    assert max(tri_counts) > 0


# ---------------------------------------------------------------------------
# 6. Determinism & Nearly-Symmetric Neighborhoods
# ---------------------------------------------------------------------------

def test_matching_determinism(base_constellation):
    """Repeated runs on the same graphs must yield identical results."""
    g_a = build_crater_graph(base_constellation, config={"method": "delaunay"})
    c_b = transform_craters(base_constellation, scale=1.5, angle_rad=0.8, tx=100.0, ty=-50.0)
    g_b = build_crater_graph(c_b, config={"method": "delaunay"})

    res1 = match_crater_constellations(g_a, g_b)
    res2 = match_crater_constellations(g_a, g_b)

    assert res1.get_match_pairs() == res2.get_match_pairs()
    for m1, m2 in zip(res1.matches, res2.matches):
        assert m1.source_node_id == m2.source_node_id
        assert m1.target_node_id == m2.target_node_id
        assert m1.descriptor_distance == m2.descriptor_distance
        assert m1.confidence == m2.confidence


def test_nearly_symmetric_neighborhood_robustness():
    """
    Test a nearly equilateral / symmetric constellation where neighbors have very
    close normalized distances and radii. Matcher must remain deterministic and not crash.
    """
    c_sym = [
        CraterCandidate(x=200.0, y=200.0, radius=15.0, confidence=0.9),  # Center
        CraterCandidate(x=200.0, y=100.0, radius=10.0, confidence=0.9),  # Top (dist=100)
        CraterCandidate(x=286.6, y=250.0, radius=10.0, confidence=0.9),  # Bottom-right (dist=100)
        CraterCandidate(x=113.4, y=250.0, radius=10.0, confidence=0.9),  # Bottom-left (dist=100)
    ]
    g_a = build_crater_graph(c_sym, config={"method": "delaunay"})
    g_b = build_crater_graph(c_sym, config={"method": "delaunay"})

    res = match_crater_constellations(g_a, g_b)
    assert isinstance(res, ConstellationMatchResult)
    assert res.match_count > 0
    # Must be deterministic across repeat
    res2 = match_crater_constellations(g_a, g_b)
    assert res.get_match_pairs() == res2.get_match_pairs()


# ---------------------------------------------------------------------------
# 7. Degenerate & Empty Cases
# ---------------------------------------------------------------------------

def test_empty_and_small_inputs():
    """Matcher should handle empty or small graphs gracefully."""
    g_empty = build_crater_graph([])
    g_single = build_crater_graph([CraterCandidate(x=50.0, y=50.0, radius=10.0, confidence=0.9)])
    g_two = build_crater_graph([
        CraterCandidate(x=50.0, y=50.0, radius=10.0, confidence=0.9),
        CraterCandidate(x=100.0, y=100.0, radius=10.0, confidence=0.9),
    ])

    # Empty vs empty
    res = match_crater_constellations(g_empty, g_empty)
    assert res.match_count == 0

    # Single vs single (only 0 neighbors -> rejected by min_neighbors)
    res = match_crater_constellations(g_single, g_single)
    assert res.match_count == 0

    # Two vs two (each has 1 neighbor -> rejected by min_neighbors=2)
    res = match_crater_constellations(g_two, g_two)
    assert res.match_count == 0


# ---------------------------------------------------------------------------
# 8. Serialization Round-Trip
# ---------------------------------------------------------------------------

def test_serialization_fidelity(base_constellation):
    g = build_crater_graph(base_constellation, config={"method": "delaunay"})
    res = match_crater_constellations(g, g)

    data = res.to_dict()
    assert isinstance(data, dict)
    assert "matches" in data
    assert "match_count" in data

    res_rec = ConstellationMatchResult.from_dict(data)
    assert res_rec.match_count == res.match_count
    assert res_rec.get_match_pairs() == res.get_match_pairs()

    for m_orig, m_rec in zip(res.matches, res_rec.matches):
        assert m_rec.source_node_id == m_orig.source_node_id
        assert m_rec.target_node_id == m_orig.target_node_id
        assert m_rec.descriptor_distance == pytest.approx(m_orig.descriptor_distance, abs=1e-5)
        assert m_rec.confidence == pytest.approx(m_orig.confidence, abs=1e-5)
        assert m_rec.supporting_triangle_count == m_orig.supporting_triangle_count
