"""
src/crater_graph/builder.py

Graph construction algorithms for crater constellations:
- Delaunay triangulation (sparse planar neighborhood, default)
- k-Nearest Neighbors (KNN)
- Radius-neighborhood graph
Includes degenerate geometry fallbacks and robust edge attribute computations.
"""

import math
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
import networkx as nx
from scipy.spatial import Delaunay, distance_matrix, QhullError

from src.crater_detection.types import CraterCandidate
from src.crater_graph.graph import CraterGraph

logger = logging.getLogger(__name__)

SUPPORTED_METHODS = ("delaunay", "knn", "radius")


def compute_reference_scale(craters: List[CraterCandidate], method: str = "median_radius") -> float:
    """
    Compute global reference scale for distance normalization.

    Args:
        craters: List of detected crater candidates.
        method: Normalization strategy ('median_radius', 'mean_radius').

    Returns:
        Positive scale factor in pixels (defaults to 1.0 if insufficient/invalid radii).
    """
    if not craters:
        return 1.0

    radii = [c.radius for c in craters if math.isfinite(c.radius) and c.radius > 0]
    if not radii:
        return 1.0

    if method == "mean_radius":
        scale = float(np.mean(radii))
    else:  # default 'median_radius'
        scale = float(np.median(radii))

    return scale if scale > 0.0 else 1.0


def compute_edge_attributes(u: int, v: int,
                            cu: CraterCandidate, cv: CraterCandidate,
                            ref_scale: float,
                            scale_method: str = "median_radius") -> Dict[str, float]:
    """
    Compute edge relationship attributes between two craters:
    - distance_px: Euclidean distance
    - normalized_distance: distance / scale
    - relative_angle_rad: atan2(y_v - y_u, x_v - x_u) from lower index to higher index
    - radius_ratio: min(r_u, r_v) / max(r_u, r_v) in (0, 1]
    - weight: distance_px
    """
    dx = cv.x - cu.x
    dy = cv.y - cu.y
    dist_px = math.sqrt(dx * dx + dy * dy)

    # Normalization scale
    if scale_method == "pair_mean_radius":
        local_scale = (cu.radius + cv.radius) / 2.0
        scale = local_scale if local_scale > 0.0 else ref_scale
    else:
        scale = ref_scale

    norm_dist = dist_px / scale if scale > 0.0 else dist_px

    # Relative angle in [-pi, pi] directed from u to v (where u < v)
    rel_angle = math.atan2(dy, dx)

    # Radius ratio in (0, 1]
    r_min = min(cu.radius, cv.radius)
    r_max = max(cu.radius, cv.radius)
    if r_max > 0.0:
        radius_ratio = max(0.0, min(1.0, r_min / r_max))
    else:
        radius_ratio = 1.0

    return {
        "distance_px": round(dist_px, 4),
        "normalized_distance": round(norm_dist, 4),
        "relative_angle_rad": round(rel_angle, 4),
        "radius_ratio": round(radius_ratio, 4),
        "weight": round(dist_px, 4),
    }


def _construct_delaunay_edges(pts: np.ndarray) -> Set[Tuple[int, int]]:
    """
    Construct edges via 2D Delaunay triangulation.
    Falls back gracefully for degenerate collinear or near-duplicate points.
    """
    n = len(pts)
    if n < 2:
        return set()
    if n == 2:
        return {(0, 1)}

    # Check for collinearity or Qhull failure
    try:
        tri = Delaunay(pts)
        edges: Set[Tuple[int, int]] = set()
        for simplex in tri.simplices:
            for i in range(3):
                u = int(simplex[i])
                v = int(simplex[(i + 1) % 3])
                edges.add((min(u, v), max(u, v)))
        return edges
    except QhullError as e:
        logger.warning("Delaunay triangulation failed (%s). Falling back to 1D consecutive spatial edges.", e)
        # Sort along primary axis of variation and connect consecutive points
        diff_x = float(np.ptp(pts[:, 0]))
        diff_y = float(np.ptp(pts[:, 1]))
        sort_axis = 0 if diff_x >= diff_y else 1
        sorted_indices = np.argsort(pts[:, sort_axis])
        edges = set()
        for i in range(len(sorted_indices) - 1):
            u = int(sorted_indices[i])
            v = int(sorted_indices[i + 1])
            if u != v:
                edges.add((min(u, v), max(u, v)))
        return edges


def _construct_knn_edges(pts: np.ndarray, k: int) -> Set[Tuple[int, int]]:
    """
    Construct edges connecting each crater to its k nearest neighbors (symmetric union).
    """
    n = len(pts)
    if n < 2:
        return set()

    k_eff = max(1, min(k, n - 1))
    dists = distance_matrix(pts, pts)
    edges: Set[Tuple[int, int]] = set()

    for i in range(n):
        # Sort distances for node i
        neighbors = np.argsort(dists[i])
        # Skip self (index 0) and take next k_eff
        for j in neighbors[1:k_eff + 1]:
            edges.add((min(i, int(j)), max(i, int(j))))

    return edges


def _construct_radius_edges(pts: np.ndarray, radius_px: float) -> Set[Tuple[int, int]]:
    """
    Construct edges connecting any pair of craters within Euclidean radius_px.
    """
    n = len(pts)
    if n < 2:
        return set()

    dists = distance_matrix(pts, pts)
    edges: Set[Tuple[int, int]] = set()

    for i in range(n):
        for j in range(i + 1, n):
            if dists[i, j] <= radius_px:
                edges.add((i, j))

    return edges


def build_crater_graph(craters: List[CraterCandidate],
                       config: Optional[Dict[str, Any]] = None,
                       image_shape: Optional[Tuple[int, int]] = None) -> CraterGraph:
    """
    Build a CraterGraph from a list of CraterCandidate objects.

    Args:
        craters: List of CraterCandidate objects.
        config: Configuration dictionary (defaults to Config.CRATER_GRAPH).
        image_shape: Optional (height, width) of the source image.

    Returns:
        Constructed CraterGraph instance.
    """
    if config is None:
        try:
            from configs.default import Config
            config = Config.CRATER_GRAPH
        except (ImportError, AttributeError):
            config = {}

    method = config.get("method", "delaunay").lower()
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown graph construction method '{method}'. Supported: {SUPPORTED_METHODS}")

    norm_scale_method = config.get("normalization_scale", "median_radius")
    max_edge_dist = config.get("max_edge_distance_px")
    ref_scale = compute_reference_scale(craters, method=norm_scale_method)

    # Initialize graph
    g = nx.Graph()

    # Add nodes
    for i, c in enumerate(craters):
        g.add_node(
            i,
            node_id=i,
            x=float(c.x),
            y=float(c.y),
            radius=float(c.radius),
            confidence=float(c.confidence),
            diameter=float(c.diameter),
            area=float(c.area),
            crater=c,
        )

    n_craters = len(craters)
    if n_craters >= 2:
        pts = np.array([[c.x, c.y] for c in craters], dtype=np.float64)

        # Build raw edges according to chosen strategy
        if method == "delaunay":
            raw_edges = _construct_delaunay_edges(pts)
        elif method == "knn":
            k = int(config.get("k_neighbors", 5))
            raw_edges = _construct_knn_edges(pts, k=k)
        elif method == "radius":
            r_thresh = float(config.get("radius_threshold_px", 150.0))
            raw_edges = _construct_radius_edges(pts, radius_px=r_thresh)
        else:
            raw_edges = set()

        # Add edges with attributes and optional distance pruning
        for u, v in sorted(raw_edges):
            cu = craters[u]
            cv = craters[v]
            attrs = compute_edge_attributes(
                u=u, v=v, cu=cu, cv=cv, ref_scale=ref_scale, scale_method=norm_scale_method
            )

            # Optional distance pruning
            if max_edge_dist is not None and max_edge_dist > 0:
                if attrs["distance_px"] > max_edge_dist:
                    continue

            g.add_edge(u, v, **attrs)

    logger.debug("Built CraterGraph (%s): %d nodes, %d edges", method, g.number_of_nodes(), g.number_of_edges())

    return CraterGraph(
        nx_graph=g,
        creation_method=method,
        image_shape=image_shape,
        params=dict(config),
    )
