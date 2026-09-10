"""
tests/test_crater_invariants.py

Comprehensive tests for Phase 7: Scale/Rotation-Robust Normalized Relationships:
- Angle wrapping (wrap_angle_pi)
- Triangle permutation invariance (all 6 vertex permutations maintain correspondence)
- Planar similarity transform invariance:
    - Pure translation
    - Pure rotation (90, 180, arbitrary angles)
    - Pure uniform scale (0.5x, 2.0x, 3.5x)
    - Combined similarity transform (scale + rotation + translation)
- Local star-neighborhood descriptor canonicalization and determinism
- Triangle extraction and descriptor construction
- Degenerate geometries (collinear, zero radius, zero area, small/empty graphs)
- Pure-primitive dictionary serialization and round-trip fidelity
- Full integration with Phase 6 CraterGraph
"""

import math
import itertools
import pytest
import numpy as np

from src.crater_detection.types import CraterCandidate
from src.crater_graph.graph import CraterGraph
from src.crater_graph.builder import build_crater_graph
from src.crater_graph.invariants import (
    wrap_angle_pi,
    LocalCraterDescriptor,
    TriangleDescriptor,
    find_graph_triangles,
    build_crater_invariant_descriptor,
    build_local_invariant_descriptors,
    build_triangle_descriptor,
    build_triangle_descriptors,
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
    """
    Apply planar similarity transformation: p' = s * R(theta) * p + t, r' = s * r.
    """
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    transformed = []
    for c in craters:
        # Scale and rotate relative to origin
        rx = scale * (cos_a * c.x - sin_a * c.y) + tx
        ry = scale * (sin_a * c.x + cos_a * c.y) + ty
        rr = scale * c.radius
        transformed.append(
            CraterCandidate(x=rx, y=ry, radius=rr, confidence=c.confidence)
        )
    return transformed


@pytest.fixture
def constellation_triangle():
    """A scalene triangle constellation with distinct radii."""
    return [
        CraterCandidate(x=100.0, y=100.0, radius=12.0, confidence=0.9),
        CraterCandidate(x=220.0, y=140.0, radius=20.0, confidence=0.85),
        CraterCandidate(x=150.0, y=280.0, radius=16.0, confidence=0.95),
    ]


@pytest.fixture
def constellation_quad():
    """A 4-crater asymmetric constellation."""
    return [
        CraterCandidate(x=100.0, y=100.0, radius=10.0, confidence=0.9),
        CraterCandidate(x=250.0, y=120.0, radius=18.0, confidence=0.85),
        CraterCandidate(x=200.0, y=260.0, radius=14.0, confidence=0.95),
        CraterCandidate(x=80.0, y=220.0, radius=22.0, confidence=0.88),
    ]


# ---------------------------------------------------------------------------
# 1. Angle Wrapping
# ---------------------------------------------------------------------------

def test_wrap_angle_pi_boundaries():
    assert wrap_angle_pi(0.0) == pytest.approx(0.0, abs=1e-9)
    assert wrap_angle_pi(math.pi) == pytest.approx(math.pi, abs=1e-9)
    assert wrap_angle_pi(-math.pi) == pytest.approx(math.pi, abs=1e-9) or wrap_angle_pi(-math.pi) == pytest.approx(-math.pi, abs=1e-9)
    assert wrap_angle_pi(3.0 * math.pi) == pytest.approx(math.pi, abs=1e-9) or wrap_angle_pi(3.0 * math.pi) == pytest.approx(-math.pi, abs=1e-9)
    assert wrap_angle_pi(math.pi / 2.0) == pytest.approx(math.pi / 2.0, abs=1e-9)
    assert wrap_angle_pi(-math.pi / 2.0) == pytest.approx(-math.pi / 2.0, abs=1e-9)
    assert wrap_angle_pi(2.0 * math.pi) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. Triangle Permutation Invariance and Radius Correspondence
# ---------------------------------------------------------------------------

def test_triangle_permutation_invariance_all_six_permutations(constellation_triangle):
    """
    Ensure all 6 permutations of triangle vertices produce the exact same canonical
    TriangleDescriptor while preserving vertex-angle-side-radius correspondence.
    """
    graph = build_crater_graph(constellation_triangle, config={"method": "delaunay"})
    perm_descriptors = []

    for perm in itertools.permutations([0, 1, 2]):
        td = build_triangle_descriptor(graph, perm, min_area=1.0)
        assert td is not None
        perm_descriptors.append(td)

    ref = perm_descriptors[0]
    for other in perm_descriptors[1:]:
        # Canonical nodes should be identical
        assert other.nodes == ref.nodes
        # Perimeter, normalized sides, angles, normalized radii should match exactly
        assert other.perimeter_px == pytest.approx(ref.perimeter_px, abs=1e-6)
        assert np.allclose(other.normalized_sides, ref.normalized_sides, atol=1e-6)
        assert np.allclose(other.internal_angles_rad, ref.internal_angles_rad, atol=1e-6)
        assert np.allclose(other.normalized_radii, ref.normalized_radii, atol=1e-6)
        assert np.allclose(other.feature_vector, ref.feature_vector, atol=1e-6)


def test_triangle_radius_correspondence(constellation_triangle):
    """
    Ensure that in the canonical representation, the normalized radius r_i
    corresponds to vertex v_i, internal angle alpha_i, and opposite side s_i.
    """
    graph = build_crater_graph(constellation_triangle, config={"method": "delaunay"})
    td = build_triangle_descriptor(graph, (0, 1, 2))
    assert td is not None

    v1, v2, v3 = td.nodes
    c1, c2, c3 = graph.get_crater(v1), graph.get_crater(v2), graph.get_crater(v3)
    r_max = max(c1.radius, c2.radius, c3.radius)

    # Radius correspondence: normalized_radii[0] must equal c1.radius / r_max
    assert td.normalized_radii[0] == pytest.approx(c1.radius / r_max, abs=1e-6)
    assert td.normalized_radii[1] == pytest.approx(c2.radius / r_max, abs=1e-6)
    assert td.normalized_radii[2] == pytest.approx(c3.radius / r_max, abs=1e-6)

    # Opposite side s1 is distance between v2 and v3
    s1_expected = math.sqrt((c2.x - c3.x) ** 2 + (c2.y - c3.y) ** 2)
    assert td.side_lengths_px[0] == pytest.approx(s1_expected, abs=1e-4)


# ---------------------------------------------------------------------------
# 3. Planar Similarity Transformations on Triangles
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tx, ty", [(100.0, -50.0), (-300.0, 450.0), (1000.0, 2000.0)])
def test_triangle_translation_invariance(constellation_triangle, tx, ty):
    g_orig = build_crater_graph(constellation_triangle, config={"method": "delaunay"})
    td_orig = build_triangle_descriptors(g_orig)[0]

    c_trans = transform_craters(constellation_triangle, tx=tx, ty=ty)
    g_trans = build_crater_graph(c_trans, config={"method": "delaunay"})
    td_trans = build_triangle_descriptors(g_trans)[0]

    assert np.allclose(td_orig.normalized_sides, td_trans.normalized_sides, atol=1e-4)
    assert np.allclose(td_orig.internal_angles_rad, td_trans.internal_angles_rad, atol=1e-4)
    assert np.allclose(td_orig.normalized_radii, td_trans.normalized_radii, atol=1e-4)
    assert np.allclose(td_orig.feature_vector, td_trans.feature_vector, atol=1e-4)


@pytest.mark.parametrize("angle_deg", [30.0, 45.0, 90.0, 180.0, 270.0, 315.0])
def test_triangle_rotation_invariance(constellation_triangle, angle_deg):
    g_orig = build_crater_graph(constellation_triangle, config={"method": "delaunay"})
    td_orig = build_triangle_descriptors(g_orig)[0]

    rad = math.radians(angle_deg)
    c_rot = transform_craters(constellation_triangle, angle_rad=rad)
    g_rot = build_crater_graph(c_rot, config={"method": "delaunay"})
    td_rot = build_triangle_descriptors(g_rot)[0]

    assert np.allclose(td_orig.normalized_sides, td_rot.normalized_sides, atol=1e-4)
    assert np.allclose(td_orig.internal_angles_rad, td_rot.internal_angles_rad, atol=1e-4)
    assert np.allclose(td_orig.normalized_radii, td_rot.normalized_radii, atol=1e-4)
    assert np.allclose(td_orig.feature_vector, td_rot.feature_vector, atol=1e-4)


@pytest.mark.parametrize("scale", [0.4, 0.75, 1.5, 2.0, 3.5])
def test_triangle_scale_invariance(constellation_triangle, scale):
    g_orig = build_crater_graph(constellation_triangle, config={"method": "delaunay"})
    td_orig = build_triangle_descriptors(g_orig)[0]

    c_scale = transform_craters(constellation_triangle, scale=scale)
    g_scale = build_crater_graph(c_scale, config={"method": "delaunay"})
    td_scale = build_triangle_descriptors(g_scale)[0]

    assert np.allclose(td_orig.normalized_sides, td_scale.normalized_sides, atol=1e-4)
    assert np.allclose(td_orig.internal_angles_rad, td_scale.internal_angles_rad, atol=1e-4)
    assert np.allclose(td_orig.normalized_radii, td_scale.normalized_radii, atol=1e-4)
    assert np.allclose(td_orig.feature_vector, td_scale.feature_vector, atol=1e-4)


def test_triangle_combined_similarity_transform(constellation_triangle):
    """
    Combined test: s=2.5, rotation=67.3 deg, tx=150.0, ty=-80.0.
    """
    g_orig = build_crater_graph(constellation_triangle, config={"method": "delaunay"})
    td_orig = build_triangle_descriptors(g_orig)[0]

    c_sim = transform_craters(
        constellation_triangle, scale=2.5, angle_rad=math.radians(67.3), tx=150.0, ty=-80.0
    )
    g_sim = build_crater_graph(c_sim, config={"method": "delaunay"})
    td_sim = build_triangle_descriptors(g_sim)[0]

    assert np.allclose(td_orig.normalized_sides, td_sim.normalized_sides, atol=1e-4)
    assert np.allclose(td_orig.internal_angles_rad, td_sim.internal_angles_rad, atol=1e-4)
    assert np.allclose(td_orig.normalized_radii, td_sim.normalized_radii, atol=1e-4)
    assert np.allclose(td_orig.feature_vector, td_sim.feature_vector, atol=1e-4)


# ---------------------------------------------------------------------------
# 4. Local Star-Neighborhood Descriptors
# ---------------------------------------------------------------------------

def test_local_descriptor_canonical_ordering(constellation_quad):
    """
    Check that local descriptors have deterministic canonical neighbor ordering
    sorted by (normalized_distance, radius_ratio, node_id).
    """
    graph = build_crater_graph(constellation_quad, config={"method": "delaunay"})
    desc = build_crater_invariant_descriptor(graph, node_id=0)

    assert desc.node_id == 0
    assert desc.num_neighbors == len(desc.canonical_neighbor_ids)
    assert len(desc.normalized_distances) == desc.num_neighbors

    # Reference neighbor is first, so its relative angle must be 0.0
    assert desc.relative_angles_rad[0] == pytest.approx(0.0, abs=1e-6)

    # Distances must be sorted monotonically non-decreasing
    for k in range(len(desc.normalized_distances) - 1):
        assert desc.normalized_distances[k] <= desc.normalized_distances[k + 1] + 1e-6


def test_local_descriptor_similarity_invariance(constellation_quad):
    """
    Test local descriptors under combined translation, rotation, and uniform scale.
    """
    g_orig = build_crater_graph(constellation_quad, config={"method": "delaunay"})
    ld_orig = build_local_invariant_descriptors(g_orig)

    c_trans = transform_craters(
        constellation_quad, scale=1.8, angle_rad=math.radians(52.0), tx=120.0, ty=340.0
    )
    g_trans = build_crater_graph(c_trans, config={"method": "delaunay"})
    ld_trans = build_local_invariant_descriptors(g_trans)

    # Compare local descriptor of node 0
    d0_orig = ld_orig[0]
    d0_trans = ld_trans[0]

    assert d0_orig.num_neighbors == d0_trans.num_neighbors
    assert np.allclose(d0_orig.normalized_distances, d0_trans.normalized_distances, atol=1e-3)
    assert np.allclose(d0_orig.radius_ratios, d0_trans.radius_ratios, atol=1e-3)
    assert np.allclose(d0_orig.relative_angles_rad, d0_trans.relative_angles_rad, atol=1e-3)
    assert np.allclose(d0_orig.feature_vector, d0_trans.feature_vector, atol=1e-3)


# ---------------------------------------------------------------------------
# 5. Multiple Scales and Rotations Invariance Suite
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("angle_deg", [0.0, 75.0, 150.0, 240.0])
def test_invariance_grid(constellation_triangle, scale, angle_deg):
    """
    Systematically evaluate similarity invariance across a grid of scales and angles.
    """
    g_orig = build_crater_graph(constellation_triangle, config={"method": "delaunay"})
    tri_orig = build_triangle_descriptors(g_orig)[0]

    c_tf = transform_craters(
        constellation_triangle, scale=scale, angle_rad=math.radians(angle_deg), tx=50.0, ty=-30.0
    )
    g_tf = build_crater_graph(c_tf, config={"method": "delaunay"})
    tri_tf = build_triangle_descriptors(g_tf)[0]

    # Max difference across feature vector elements must be < 1e-3
    max_diff = np.max(np.abs(tri_orig.feature_vector - tri_tf.feature_vector))
    assert max_diff < 1e-3, f"Failed at scale={scale}, angle={angle_deg} with max_diff={max_diff}"


# ---------------------------------------------------------------------------
# 6. Degenerate Geometry & Robustness
# ---------------------------------------------------------------------------

def test_triangle_collinear_degenerate():
    """Collinear points produce area = 0 and should be cleanly rejected."""
    collinear = [
        CraterCandidate(x=50.0, y=50.0, radius=10.0, confidence=0.9),
        CraterCandidate(x=100.0, y=100.0, radius=10.0, confidence=0.9),
        CraterCandidate(x=150.0, y=150.0, radius=10.0, confidence=0.9),
    ]
    graph = build_crater_graph(collinear, config={"method": "delaunay"})
    # Manual attempt on collinear nodes
    td = build_triangle_descriptor(graph, (0, 1, 2), min_area=1.0)
    assert td is None


def test_isolated_and_small_graphs():
    """Graphs with 0, 1, or 2 craters must not crash and produce empty descriptors."""
    # 0 craters
    g0 = build_crater_graph([])
    assert build_local_invariant_descriptors(g0) == {}
    assert build_triangle_descriptors(g0) == []

    # 1 crater
    g1 = build_crater_graph([CraterCandidate(x=50.0, y=50.0, radius=10.0, confidence=0.9)])
    ld1 = build_local_invariant_descriptors(g1)
    assert 0 in ld1
    assert ld1[0].num_neighbors == 0
    assert build_triangle_descriptors(g1) == []

    # 2 craters
    g2 = build_crater_graph([
        CraterCandidate(x=50.0, y=50.0, radius=10.0, confidence=0.9),
        CraterCandidate(x=100.0, y=100.0, radius=10.0, confidence=0.9),
    ])
    ld2 = build_local_invariant_descriptors(g2)
    assert ld2[0].num_neighbors == 1
    assert ld2[1].num_neighbors == 1
    assert build_triangle_descriptors(g2) == []


def test_zero_radius_handling():
    """Craters with zero radius must not generate NaN or Inf in descriptors."""
    craters = [
        CraterCandidate(x=100.0, y=100.0, radius=0.0, confidence=0.9),
        CraterCandidate(x=200.0, y=100.0, radius=0.0, confidence=0.9),
        CraterCandidate(x=150.0, y=200.0, radius=0.0, confidence=0.9),
    ]
    graph = build_crater_graph(craters, config={"method": "delaunay"})
    td = build_triangle_descriptor(graph, (0, 1, 2))
    assert td is not None
    assert np.all(np.isfinite(td.feature_vector))
    assert np.all(td.normalized_radii == (1.0, 1.0, 1.0))

    ld = build_crater_invariant_descriptor(graph, 0)
    assert np.all(np.isfinite(ld.feature_vector))


# ---------------------------------------------------------------------------
# 7. Serialization and Round-trip Fidelity
# ---------------------------------------------------------------------------

def test_triangle_descriptor_serialization(constellation_triangle):
    graph = build_crater_graph(constellation_triangle, config={"method": "delaunay"})
    td = build_triangle_descriptor(graph, (0, 1, 2))
    assert td is not None

    data = td.to_dict()
    assert isinstance(data, dict)
    assert "feature_vector" in data
    assert "nodes" in data

    td_rec = TriangleDescriptor.from_dict(data)
    assert td_rec.nodes == td.nodes
    assert td_rec.perimeter_px == pytest.approx(td.perimeter_px, abs=1e-4)
    assert np.allclose(td_rec.feature_vector, td.feature_vector, atol=1e-4)


def test_local_descriptor_serialization(constellation_quad):
    graph = build_crater_graph(constellation_quad, config={"method": "delaunay"})
    ld = build_crater_invariant_descriptor(graph, 0)

    data = ld.to_dict()
    assert isinstance(data, dict)
    assert "canonical_neighbor_ids" in data
    assert "relative_angles_rad" in data

    ld_rec = LocalCraterDescriptor.from_dict(data)
    assert ld_rec.node_id == ld.node_id
    assert ld_rec.canonical_neighbor_ids == ld.canonical_neighbor_ids
    assert np.allclose(ld_rec.feature_vector, ld.feature_vector, atol=1e-4)


# ---------------------------------------------------------------------------
# 8. Scale Normalization Strategies Comparison
# ---------------------------------------------------------------------------

def test_scale_normalization_strategies(constellation_quad):
    graph = build_crater_graph(constellation_quad, config={"method": "delaunay"})

    # Strategy: median_radius
    ld_median = build_local_invariant_descriptors(
        graph, config={"scale_normalization": "median_radius"}
    )
    # Strategy: mean_radius
    ld_mean = build_local_invariant_descriptors(
        graph, config={"scale_normalization": "mean_radius"}
    )
    # Strategy: pair_mean_radius
    ld_pair = build_local_invariant_descriptors(
        graph, config={"scale_normalization": "pair_mean_radius"}
    )

    assert len(ld_median) == 4
    assert len(ld_mean) == 4
    assert len(ld_pair) == 4

    # All must produce finite non-empty vectors
    for nid in range(4):
        assert np.all(np.isfinite(ld_median[nid].feature_vector))
        assert np.all(np.isfinite(ld_mean[nid].feature_vector))
        assert np.all(np.isfinite(ld_pair[nid].feature_vector))
