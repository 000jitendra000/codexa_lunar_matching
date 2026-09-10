"""
tests/test_crater_graph.py

Phase 6 unit tests for crater graph representation, construction strategies,
degenerate geometry fallbacks, edge attributes, feature extraction, determinism,
and Phase 5 integration.
"""

import math
import pytest
import numpy as np
import cv2

from configs.default import Config
from src.crater_detection.types import CraterCandidate
from src.crater_detection.pipeline import detect_craters
from src.crater_graph.graph import CraterGraph
from src.crater_graph.builder import (
    build_crater_graph,
    compute_reference_scale,
    compute_edge_attributes,
)
from src.crater_graph.features import (
    compute_graph_summary,
    get_coordinates_array,
    get_radii_array,
    get_adjacency_matrix,
    get_pairwise_distance_matrix,
)


# ── Synthetic constellation fixture ──────────────────────────────────────────

@pytest.fixture
def synthetic_constellation():
    """
    Deterministic 5-crater constellation with known geometry.
         C3 (300, 100)
    C1 (100, 100)       C4 (300, 300)
         C2 (100, 300)
              C5 (200, 200)
    """
    return [
        CraterCandidate(x=100.0, y=100.0, radius=15.0, confidence=0.95),  # C0
        CraterCandidate(x=100.0, y=300.0, radius=20.0, confidence=0.90),  # C1
        CraterCandidate(x=300.0, y=100.0, radius=25.0, confidence=0.92),  # C2
        CraterCandidate(x=300.0, y=300.0, radius=18.0, confidence=0.88),  # C3
        CraterCandidate(x=200.0, y=200.0, radius=30.0, confidence=0.98),  # C4
    ]


# ── 1. Graph Representation & Node/Edge Attributes ───────────────────────────

class TestGraphRepresentation:
    def test_empty_graph(self):
        g = build_crater_graph([])
        assert g.num_nodes == 0
        assert g.num_edges == 0
        assert g.nodes() == []
        assert g.edges() == []

    def test_single_crater_graph(self):
        c = CraterCandidate(100.0, 100.0, 20.0, 0.9)
        g = build_crater_graph([c])
        assert g.num_nodes == 1
        assert g.num_edges == 0
        node = g.get_node(0)
        assert node["x"] == 100.0
        assert node["y"] == 100.0
        assert node["radius"] == 20.0
        assert node["confidence"] == 0.9
        assert node["diameter"] == 40.0
        assert math.isclose(node["area"], math.pi * 400.0)
        assert g.get_crater(0) == c

    def test_two_crater_graph(self):
        c1 = CraterCandidate(0.0, 0.0, 10.0, 0.9)
        c2 = CraterCandidate(30.0, 40.0, 20.0, 0.8)
        g = build_crater_graph([c1, c2])
        assert g.num_nodes == 2
        assert g.num_edges == 1
        edge = g.get_edge(0, 1)
        # 3-4-5 triangle: dist = 50.0
        assert math.isclose(edge["distance_px"], 50.0, abs_tol=1e-3)
        # radius ratio = 10 / 20 = 0.5
        assert math.isclose(edge["radius_ratio"], 0.5, abs_tol=1e-3)
        # angle: atan2(40, 30)
        assert math.isclose(edge["relative_angle_rad"], math.atan2(40, 30), abs_tol=1e-3)

    def test_node_neighbors_and_degree(self, synthetic_constellation):
        g = build_crater_graph(synthetic_constellation, config={"method": "delaunay"})
        assert g.num_nodes == 5
        assert g.num_edges > 0
        for nid in g.nodes():
            neighbors = g.neighbors(nid)
            assert len(neighbors) == g.degree(nid)
            for neighbor in neighbors:
                assert g.nx_graph.has_edge(nid, neighbor)

    def test_invalid_node_or_edge_raises(self):
        g = build_crater_graph([])
        with pytest.raises(KeyError):
            g.get_node(99)
        with pytest.raises(KeyError):
            g.get_edge(0, 1)


# ── 2. Construction Strategies (Delaunay, KNN, Radius) ───────────────────────

class TestConstructionStrategies:
    def test_delaunay_construction(self, synthetic_constellation):
        g = build_crater_graph(synthetic_constellation, config={"method": "delaunay"})
        assert g.creation_method == "delaunay"
        assert g.num_nodes == 5
        # 5 points in this planar layout form 8 Delaunay edges
        assert g.num_edges == 8

    def test_knn_construction(self, synthetic_constellation):
        g = build_crater_graph(synthetic_constellation, config={"method": "knn", "k_neighbors": 2})
        assert g.creation_method == "knn"
        assert g.num_nodes == 5
        assert g.num_edges > 0
        # Each node has degree at least k=2
        for nid in g.nodes():
            assert g.degree(nid) >= 2

    def test_knn_k_larger_than_nodes(self, synthetic_constellation):
        # When k >= n-1, should cap gracefully without crashing
        g = build_crater_graph(synthetic_constellation, config={"method": "knn", "k_neighbors": 10})
        assert g.num_nodes == 5
        # Complete graph K5 has 5*4/2 = 10 edges
        assert g.num_edges == 10

    def test_radius_construction(self, synthetic_constellation):
        # Threshold below minimum pair distance -> 0 edges
        g_none = build_crater_graph(synthetic_constellation, config={"method": "radius", "radius_threshold_px": 50.0})
        assert g_none.num_edges == 0

        # Threshold large enough to connect all pairs -> complete graph
        g_all = build_crater_graph(synthetic_constellation, config={"method": "radius", "radius_threshold_px": 500.0})
        assert g_all.num_edges == 10

    def test_max_edge_distance_pruning(self, synthetic_constellation):
        # Prune edges longer than 150 px
        g_unpruned = build_crater_graph(synthetic_constellation, config={"method": "delaunay", "max_edge_distance_px": None})
        g_pruned = build_crater_graph(synthetic_constellation, config={"method": "delaunay", "max_edge_distance_px": 150.0})
        assert g_pruned.num_edges <= g_unpruned.num_edges
        for _, _, d in g_pruned.nx_graph.edges(data=True):
            assert d["distance_px"] <= 150.0

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            build_crater_graph([], config={"method": "spectral_super_cluster"})


# ── 3. Degenerate Geometry Fallbacks ──────────────────────────────────────────

class TestDegenerateGeometry:
    def test_collinear_points_fallback(self):
        # 4 strictly collinear points on the line y = 100
        collinear_craters = [
            CraterCandidate(10.0, 100.0, 5.0, 0.9),
            CraterCandidate(30.0, 100.0, 5.0, 0.9),
            CraterCandidate(60.0, 100.0, 5.0, 0.9),
            CraterCandidate(90.0, 100.0, 5.0, 0.9),
        ]
        # Should not crash with QhullError; gracefully builds 1D Delaunay (3 edges)
        g = build_crater_graph(collinear_craters, config={"method": "delaunay"})
        assert g.num_nodes == 4
        assert g.num_edges == 3

    def test_duplicate_coordinates(self):
        # Points sharing identical x, y coordinates
        duplicates = [
            CraterCandidate(50.0, 50.0, 10.0, 0.9),
            CraterCandidate(50.0, 50.0, 15.0, 0.8),
            CraterCandidate(100.0, 100.0, 12.0, 0.85),
        ]
        g = build_crater_graph(duplicates, config={"method": "delaunay"})
        assert g.num_nodes == 3
        # Should construct edges without crashing
        assert g.num_edges >= 1


# ── 4. Edge Relationships & Normalization ─────────────────────────────────────

class TestRelationshipsAndNormalization:
    def test_normalization_strategies(self, synthetic_constellation):
        # Test median vs mean vs pair_mean
        g_med = build_crater_graph(synthetic_constellation, config={"normalization_scale": "median_radius"})
        g_mean = build_crater_graph(synthetic_constellation, config={"normalization_scale": "mean_radius"})
        g_pair = build_crater_graph(synthetic_constellation, config={"normalization_scale": "pair_mean_radius"})

        assert g_med.num_edges == g_mean.num_edges == g_pair.num_edges

        # Median radius of [15, 18, 20, 25, 30] is 20.0
        e_med = g_med.get_edge(0, 1)
        # Distance between (100, 100) and (100, 300) is 200.0
        assert math.isclose(e_med["distance_px"], 200.0, abs_tol=1e-3)
        assert math.isclose(e_med["normalized_distance"], 200.0 / 20.0, abs_tol=1e-3)

    def test_radius_ratio_calculation(self):
        c1 = CraterCandidate(0, 0, 10.0, 0.9)
        c2 = CraterCandidate(100, 0, 40.0, 0.9)
        attrs = compute_edge_attributes(0, 1, c1, c2, ref_scale=20.0)
        assert attrs["radius_ratio"] == 0.25   # 10 / 40
        assert attrs["distance_px"] == 100.0
        assert attrs["normalized_distance"] == 5.0  # 100 / 20

    def test_zero_radius_safety(self):
        c1 = CraterCandidate(0, 0, 0.0, 0.9)
        c2 = CraterCandidate(100, 0, 0.0, 0.9)
        scale = compute_reference_scale([c1, c2], method="median_radius")
        assert scale == 1.0  # safe fallback
        attrs = compute_edge_attributes(0, 1, c1, c2, ref_scale=scale)
        assert math.isfinite(attrs["normalized_distance"])
        assert math.isfinite(attrs["radius_ratio"])


# ── 5. Feature Utilities & Serialization ──────────────────────────────────────

class TestFeaturesAndSerialization:
    def test_graph_summary(self, synthetic_constellation):
        g = build_crater_graph(synthetic_constellation, config={"method": "delaunay"})
        summary = compute_graph_summary(g)
        assert summary["node_count"] == 5
        assert summary["edge_count"] == 8
        assert summary["creation_method"] == "delaunay"
        assert summary["mean_degree"] > 0
        assert summary["density"] > 0
        assert summary["mean_distance_px"] is not None
        assert summary["is_connected"] is True

    def test_empty_graph_summary(self):
        g = build_crater_graph([])
        summary = compute_graph_summary(g)
        assert summary["node_count"] == 0
        assert summary["edge_count"] == 0
        assert summary["density"] == 0.0
        assert summary["mean_degree"] == 0.0
        assert summary["mean_distance_px"] is None
        assert summary["is_connected"] is False

    def test_coordinates_and_distance_matrix(self, synthetic_constellation):
        g = build_crater_graph(synthetic_constellation)
        coords = get_coordinates_array(g)
        assert coords.shape == (5, 2)
        assert coords[0, 0] == 100.0 and coords[0, 1] == 100.0

        radii = get_radii_array(g)
        assert radii.shape == (5,)
        assert radii[0] == 15.0

        dm = get_pairwise_distance_matrix(g)
        assert dm.shape == (5, 5)
        assert np.allclose(np.diag(dm), 0.0)
        assert np.allclose(dm, dm.T)  # symmetric

        adj = get_adjacency_matrix(g, weighted=False)
        assert adj.shape == (5, 5)
        assert np.allclose(adj, adj.T)

    def test_serialization_roundtrip(self, synthetic_constellation):
        g_orig = build_crater_graph(synthetic_constellation, config={"method": "delaunay"}, image_shape=(512, 512))
        data = g_orig.to_dict()

        # Check serialization format (all primitives, no complex types)
        assert isinstance(data, dict)
        assert isinstance(data["nodes"], list)
        assert isinstance(data["edges"], list)
        assert data["num_nodes"] == 5
        assert data["num_edges"] == 8

        # Reconstruct
        g_reconstructed = CraterGraph.from_dict(data)
        assert g_reconstructed.num_nodes == g_orig.num_nodes
        assert g_reconstructed.num_edges == g_orig.num_edges
        assert g_reconstructed.creation_method == g_orig.creation_method
        assert g_reconstructed.image_shape == (512, 512)

        for u, v in g_orig.edges():
            assert g_reconstructed.nx_graph.has_edge(u, v)
            orig_attr = g_orig.get_edge(u, v)
            recon_attr = g_reconstructed.get_edge(u, v)
            assert math.isclose(orig_attr["distance_px"], recon_attr["distance_px"], abs_tol=1e-3)


# ── 6. Determinism & Phase 5 Integration ─────────────────────────────────────

class TestDeterminismAndIntegration:
    def test_graph_construction_is_deterministic(self, synthetic_constellation):
        # Multiple runs with identical inputs must yield identical topology and edge order
        g1 = build_crater_graph(synthetic_constellation, config={"method": "delaunay"})
        g2 = build_crater_graph(synthetic_constellation, config={"method": "delaunay"})
        assert g1.edges() == g2.edges()
        for u, v in g1.edges():
            assert g1.get_edge(u, v) == g2.get_edge(u, v)

    def test_phase5_to_phase6_pipeline_integration(self):
        # Create synthetic image -> Phase 5 detect_craters -> Phase 6 build_crater_graph
        img = np.full((256, 256), 128, dtype=np.uint8)
        # Draw 4 distinct circles
        cv2.circle(img, (60, 60), 20, 20, -1)
        cv2.circle(img, (60, 60), 20, 220, 2)
        cv2.circle(img, (180, 60), 25, 20, -1)
        cv2.circle(img, (180, 60), 25, 220, 2)
        cv2.circle(img, (120, 180), 30, 20, -1)
        cv2.circle(img, (120, 180), 30, 220, 2)

        # Detect craters via Phase 5 (Mock or Hough)
        det_result = detect_craters(img, config={"detector_type": "mock"})
        assert det_result.count > 0

        # Build graph via Phase 6
        graph = build_crater_graph(det_result.craters, image_shape=det_result.image_shape)
        assert isinstance(graph, CraterGraph)
        assert graph.num_nodes == det_result.count
        assert graph.num_edges > 0
        summary = compute_graph_summary(graph)
        assert summary["node_count"] == det_result.count
