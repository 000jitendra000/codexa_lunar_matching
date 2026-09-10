"""
src/crater_graph package.

Phase 6 Crater Graph representation and construction:
- CraterGraph class wrapping networkx.Graph with typed node and edge attributes
- build_crater_graph() supporting Delaunay, KNN, and Radius strategies
- Analytical features, summary statistics, and distance matrices
- Graph visualization overlay utilities
"""

from src.crater_graph.graph import CraterGraph
from src.crater_graph.builder import (
    build_crater_graph,
    compute_reference_scale,
    compute_edge_attributes,
    SUPPORTED_METHODS,
)
from src.crater_graph.features import (
    compute_graph_summary,
    get_coordinates_array,
    get_radii_array,
    get_adjacency_matrix,
    get_pairwise_distance_matrix,
)
from src.crater_graph.visualization import draw_crater_graph
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
from src.crater_graph.constellation_matcher import (
    ConstellationMatch,
    ConstellationMatchResult,
    compute_local_descriptor_distance,
    count_supporting_triangles,
    match_crater_constellations,
)

__all__ = [
    "CraterGraph",
    "build_crater_graph",
    "compute_reference_scale",
    "compute_edge_attributes",
    "compute_graph_summary",
    "get_coordinates_array",
    "get_radii_array",
    "get_adjacency_matrix",
    "get_pairwise_distance_matrix",
    "draw_crater_graph",
    "SUPPORTED_METHODS",
    "wrap_angle_pi",
    "LocalCraterDescriptor",
    "TriangleDescriptor",
    "find_graph_triangles",
    "build_crater_invariant_descriptor",
    "build_local_invariant_descriptors",
    "build_triangle_descriptor",
    "build_triangle_descriptors",
    "ConstellationMatch",
    "ConstellationMatchResult",
    "compute_local_descriptor_distance",
    "count_supporting_triangles",
    "match_crater_constellations",
]

