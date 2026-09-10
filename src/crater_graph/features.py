"""
src/crater_graph/features.py

Graph feature extraction and analytical utilities for crater graphs:
- Summary statistics (node count, edge count, density, degree, distance distribution)
- Coordinate arrays and radii vectors
- Pairwise distance and adjacency matrices
"""

from typing import Dict, Any, Optional
import math
import numpy as np
import networkx as nx
from src.crater_graph.graph import CraterGraph


def compute_graph_summary(graph: CraterGraph) -> Dict[str, Any]:
    """
    Compute analytical summary statistics for a CraterGraph.

    Args:
        graph: Input CraterGraph instance.

    Returns:
        Dict containing topological and geometric summary metrics.
    """
    n = graph.num_nodes
    m = graph.num_edges

    density = 0.0
    if n >= 2:
        density = (2.0 * m) / (n * (n - 1.0))

    mean_degree = 0.0
    min_deg = 0
    max_deg = 0
    if n >= 1:
        degrees = [graph.degree(i) for i in graph.nodes()]
        mean_degree = float(sum(degrees)) / n
        min_deg = min(degrees)
        max_deg = max(degrees)

    distances = [ed.get("distance_px", 0.0) for _, _, ed in graph.nx_graph.edges(data=True)]
    mean_dist = float(np.mean(distances)) if distances else None
    median_dist = float(np.median(distances)) if distances else None
    min_dist = float(np.min(distances)) if distances else None
    max_dist = float(np.max(distances)) if distances else None

    radii = [graph.get_node(i)["radius"] for i in graph.nodes()]
    mean_radius = float(np.mean(radii)) if radii else None
    median_radius = float(np.median(radii)) if radii else None

    is_connected = False
    if n > 0:
        is_connected = nx.is_connected(graph.nx_graph)

    return {
        "node_count": n,
        "edge_count": m,
        "creation_method": graph.creation_method,
        "density": round(density, 4),
        "mean_degree": round(mean_degree, 2),
        "min_degree": min_deg,
        "max_degree": max_deg,
        "mean_distance_px": round(mean_dist, 2) if mean_dist is not None else None,
        "median_distance_px": round(median_dist, 2) if median_dist is not None else None,
        "min_distance_px": round(min_dist, 2) if min_dist is not None else None,
        "max_distance_px": round(max_dist, 2) if max_dist is not None else None,
        "mean_radius_px": round(mean_radius, 2) if mean_radius is not None else None,
        "median_radius_px": round(median_radius, 2) if median_radius is not None else None,
        "is_connected": is_connected,
    }


def get_coordinates_array(graph: CraterGraph) -> np.ndarray:
    """
    Extract an (N, 2) numpy array of crater center coordinates (x, y)
    ordered by node ID.
    """
    n = graph.num_nodes
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)

    coords = np.zeros((n, 2), dtype=np.float64)
    for i in graph.nodes():
        node = graph.get_node(i)
        coords[i, 0] = node["x"]
        coords[i, 1] = node["y"]
    return coords


def get_radii_array(graph: CraterGraph) -> np.ndarray:
    """
    Extract an (N,) numpy array of crater radii ordered by node ID.
    """
    n = graph.num_nodes
    if n == 0:
        return np.zeros((0,), dtype=np.float64)

    radii = np.zeros(n, dtype=np.float64)
    for i in graph.nodes():
        radii[i] = graph.get_node(i)["radius"]
    return radii


def get_adjacency_matrix(graph: CraterGraph, weighted: bool = False) -> np.ndarray:
    """
    Extract an (N, N) adjacency matrix.
    If weighted=True, edge weights (distance_px) are used; otherwise binary 0/1.
    """
    n = graph.num_nodes
    adj = np.zeros((n, n), dtype=np.float64)
    weight_attr = "weight" if weighted else None

    for u, v, d in graph.nx_graph.edges(data=True):
        val = d.get("weight", 1.0) if weighted else 1.0
        adj[u, v] = val
        adj[v, u] = val
    return adj


def get_pairwise_distance_matrix(graph: CraterGraph) -> np.ndarray:
    """
    Compute an (N, N) pairwise Euclidean distance matrix across all crater centers.
    """
    coords = get_coordinates_array(graph)
    if len(coords) == 0:
        return np.zeros((0, 0), dtype=np.float64)

    from scipy.spatial import distance_matrix
    return distance_matrix(coords, coords)
