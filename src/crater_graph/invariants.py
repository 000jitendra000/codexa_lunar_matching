"""
src/crater_graph/invariants.py

Scale- and rotation-robust normalized relationship descriptors for crater graphs:
- LocalCraterDescriptor: Deterministic local star-neighborhood descriptor.
- TriangleDescriptor: Permutation-invariant, perimeter-normalized 3-clique descriptor
  preserving vertex-radius-angle correspondence.
- Utilities: wrap_angle_pi, find_graph_triangles, and builder functions.

Mathematical Design:
- Translation invariance: Inter-crater relative vectors (x_j - x_i, y_j - y_i).
- Rotation invariance: Relative angular separations relative to a canonically selected
  reference neighbor and triangle internal angles via Law of Cosines.
- Scale normalization: Global median radius, pairwise mean radius, and triangle perimeter.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Set
import math
import itertools
import numpy as np

from src.crater_detection.types import CraterCandidate
from src.crater_graph.graph import CraterGraph


def wrap_angle_pi(angle: float) -> float:
    """
    Wrap an angle in radians to the canonical interval (-pi, pi].

    Args:
        angle: Input angle in radians.

    Returns:
        Wrapped angle in (-pi, pi].
    """
    if not math.isfinite(angle):
        return 0.0
    w = (angle + math.pi) % (2.0 * math.pi)
    if w < 0.0:
        w += 2.0 * math.pi
    res = w - math.pi
    # Map -pi to +pi for unique canonical representation in (-pi, pi]
    if abs(res + math.pi) < 1e-12:
        return math.pi
    return res



@dataclass
class LocalCraterDescriptor:
    """
    Local neighborhood descriptor for a crater node.

    Attributes:
        node_id: Center node identifier in the source graph.
        crater: Original CraterCandidate of the center node.
        num_neighbors: Number of connected neighbors.
        canonical_neighbor_ids: Sorted list of neighbor node IDs in canonical order.
        normalized_distances: Scale-normalized distances to each canonical neighbor.
        radius_ratios: Radius ratios min(r_i, r_j) / max(r_i, r_j) for each neighbor.
        relative_angles_rad: Relative angular separation of each neighbor relative
                             to the canonical reference neighbor (first neighbor).
        feature_vector: Fixed-length 1D numpy array for fast vector comparison.
    """
    node_id: int
    crater: CraterCandidate
    num_neighbors: int
    canonical_neighbor_ids: List[int]
    normalized_distances: List[float]
    radius_ratios: List[float]
    relative_angles_rad: List[float]
    feature_vector: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))

    def to_dict(self) -> Dict[str, Any]:
        """Convert descriptor to a JSON-friendly primitive dictionary."""
        return {
            "node_id": int(self.node_id),
            "crater": self.crater.to_dict(),
            "num_neighbors": int(self.num_neighbors),
            "canonical_neighbor_ids": [int(nid) for nid in self.canonical_neighbor_ids],
            "normalized_distances": [round(float(d), 5) for d in self.normalized_distances],
            "radius_ratios": [round(float(r), 5) for r in self.radius_ratios],
            "relative_angles_rad": [round(float(a), 5) for a in self.relative_angles_rad],
            "feature_vector": [round(float(v), 5) for v in self.feature_vector.tolist()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LocalCraterDescriptor":
        """Reconstruct descriptor from a primitive dictionary."""
        crater = CraterCandidate.from_dict(data["crater"])
        feat = np.array(data.get("feature_vector", []), dtype=np.float64)
        return cls(
            node_id=int(data["node_id"]),
            crater=crater,
            num_neighbors=int(data["num_neighbors"]),
            canonical_neighbor_ids=[int(nid) for nid in data["canonical_neighbor_ids"]],
            normalized_distances=[float(d) for d in data["normalized_distances"]],
            radius_ratios=[float(r) for r in data["radius_ratios"]],
            relative_angles_rad=[float(a) for a in data["relative_angles_rad"]],
            feature_vector=feat,
        )


@dataclass
class TriangleDescriptor:
    """
    Canonical, permutation-invariant triangle descriptor for a 3-clique in CraterGraph.

    The descriptor is constructed by evaluating all 6 vertex permutations and selecting
    the lexicographically smallest representation while preserving the strict correspondence:
        vertex v_i <-> opposite side s_i <-> internal angle alpha_i <-> normalized radius r_i.

    Attributes:
        nodes: Tuple of 3 node IDs (v1, v2, v3) in canonical permutation order.
        side_lengths_px: Raw side lengths (s1, s2, s3) in pixels opposite to (v1, v2, v3).
        perimeter_px: Perimeter P = s1 + s2 + s3.
        normalized_sides: Perimeter-normalized side lengths (s1/P, s2/P, s3/P).
        internal_angles_rad: Internal angles (alpha1, alpha2, alpha3) at vertices (v1, v2, v3).
        normalized_radii: Radii at (v1, v2, v3) normalized by max(r_v1, r_v2, r_v3).
        area_px: Triangle area in square pixels.
        feature_vector: 1D numpy array of 9 features:
            [s1/P, s2/P, s3/P, alpha1, alpha2, alpha3, r1_norm, r2_norm, r3_norm].
    """
    nodes: Tuple[int, int, int]
    side_lengths_px: Tuple[float, float, float]
    perimeter_px: float
    normalized_sides: Tuple[float, float, float]
    internal_angles_rad: Tuple[float, float, float]
    normalized_radii: Tuple[float, float, float]
    area_px: float
    feature_vector: np.ndarray = field(default_factory=lambda: np.zeros(9, dtype=np.float64))

    def to_dict(self) -> Dict[str, Any]:
        """Convert triangle descriptor to a JSON-friendly primitive dictionary."""
        return {
            "nodes": [int(n) for n in self.nodes],
            "side_lengths_px": [round(float(s), 4) for s in self.side_lengths_px],
            "perimeter_px": round(float(self.perimeter_px), 4),
            "normalized_sides": [round(float(s), 5) for s in self.normalized_sides],
            "internal_angles_rad": [round(float(a), 5) for a in self.internal_angles_rad],
            "normalized_radii": [round(float(r), 5) for r in self.normalized_radii],
            "area_px": round(float(self.area_px), 4),
            "feature_vector": [round(float(v), 5) for v in self.feature_vector.tolist()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TriangleDescriptor":
        """Reconstruct triangle descriptor from a primitive dictionary."""
        feat = np.array(data.get("feature_vector", []), dtype=np.float64)
        return cls(
            nodes=tuple(int(n) for n in data["nodes"]),
            side_lengths_px=tuple(float(s) for s in data["side_lengths_px"]),
            perimeter_px=float(data["perimeter_px"]),
            normalized_sides=tuple(float(s) for s in data["normalized_sides"]),
            internal_angles_rad=tuple(float(a) for a in data["internal_angles_rad"]),
            normalized_radii=tuple(float(r) for r in data["normalized_radii"]),
            area_px=float(data["area_px"]),
            feature_vector=feat,
        )


def find_graph_triangles(graph: CraterGraph) -> List[Tuple[int, int, int]]:
    """
    Find all triangles (3-cliques) in the CraterGraph.

    Returns:
        List of sorted node ID triples (u, v, w) with u < v < w.
    """
    triangles: List[Tuple[int, int, int]] = []
    nx_g = graph.nx_graph
    nodes = graph.nodes()

    for u in nodes:
        nbrs_u = set(nx_g.neighbors(u))
        for v in nbrs_u:
            if v > u:
                common = nbrs_u.intersection(nx_g.neighbors(v))
                for w in common:
                    if w > v:
                        triangles.append((u, v, w))

    return sorted(triangles)


def build_crater_invariant_descriptor(
    graph: CraterGraph,
    node_id: int,
    config: Optional[Dict[str, Any]] = None,
    ref_scale: Optional[float] = None,
) -> LocalCraterDescriptor:
    """
    Build a scale- and rotation-robust local neighborhood descriptor for a given crater node.

    Neighbor canonicalization:
    1. Neighbors are evaluated with their Euclidean distance, scale-normalized distance,
       radius ratio, and polar angle.
    2. Neighbors are canonically ordered by:
       (round(normalized_distance, 5), round(1.0 - radius_ratio, 5), neighbor_node_id).
    3. The first neighbor in this deterministic order is selected as the canonical reference neighbor.
    4. Relative angles for all neighbors are computed as wrap_angle_pi(theta_j - theta_ref).

    Args:
        graph: Input CraterGraph instance.
        node_id: Index of the crater node.
        config: Optional configuration dictionary (defaults to Config.CRATER_INVARIANTS).
        ref_scale: Optional precomputed reference scale in pixels.

    Returns:
        LocalCraterDescriptor for the node.
    """
    if config is None:
        try:
            from configs.default import Config
            config = getattr(Config, "CRATER_INVARIANTS", {})
        except (ImportError, AttributeError):
            config = {}

    max_nbrs = int(config.get("max_neighbors", 8))
    norm_method = config.get("scale_normalization", "median_radius")

    crater = graph.get_crater(node_id)
    c_x, c_y, c_r = crater.x, crater.y, crater.radius

    # Compute reference scale if not provided
    if ref_scale is None:
        from src.crater_graph.builder import compute_reference_scale
        craters_all = graph.craters()
        ref_scale = compute_reference_scale(craters_all, method=norm_method)
    if ref_scale <= 1e-9:
        ref_scale = 1.0

    raw_nbr_ids = graph.neighbors(node_id)
    if not raw_nbr_ids:
        # Isolated node
        zero_feat = np.zeros(3 * max_nbrs, dtype=np.float64)
        return LocalCraterDescriptor(
            node_id=node_id,
            crater=crater,
            num_neighbors=0,
            canonical_neighbor_ids=[],
            normalized_distances=[],
            radius_ratios=[],
            relative_angles_rad=[],
            feature_vector=zero_feat,
        )

    # Compute raw properties for each neighbor
    nbr_data = []
    for nbr_id in raw_nbr_ids:
        nbr_crater = graph.get_crater(nbr_id)
        dx = nbr_crater.x - c_x
        dy = nbr_crater.y - c_y
        dist_px = math.sqrt(dx * dx + dy * dy)

        # Scale normalization
        if norm_method == "pair_mean_radius":
            pair_scale = (c_r + nbr_crater.radius) / 2.0
            scale = pair_scale if pair_scale > 1e-9 else ref_scale
        else:
            scale = ref_scale
        norm_dist = dist_px / scale if scale > 1e-9 else dist_px

        # Radius ratio
        r_min = min(c_r, nbr_crater.radius)
        r_max = max(c_r, nbr_crater.radius)
        rad_ratio = (r_min / r_max) if r_max > 1e-9 else 1.0

        # Absolute polar angle in [-pi, pi]
        theta = math.atan2(dy, dx)

        nbr_data.append({
            "id": nbr_id,
            "dist_px": dist_px,
            "norm_dist": norm_dist,
            "radius_ratio": rad_ratio,
            "theta": theta,
        })

    # Canonical neighbor ordering:
    # 1. normalized distance (rounded to 5 decimal places)
    # 2. radius ratio descending (larger ratio preferred)
    # 3. stable neighbor node_id as deterministic tie-breaker
    nbr_data.sort(
        key=lambda d: (
            round(d["norm_dist"], 5),
            round(1.0 - d["radius_ratio"], 5),
            d["id"]
        )
    )

    # Reference neighbor is the first in canonical order
    ref_theta = nbr_data[0]["theta"]

    canonical_ids: List[int] = []
    norm_distances: List[float] = []
    rad_ratios: List[float] = []
    rel_angles: List[float] = []

    for d in nbr_data:
        canonical_ids.append(d["id"])
        norm_distances.append(round(d["norm_dist"], 5))
        rad_ratios.append(round(d["radius_ratio"], 5))
        # Angle relative to reference neighbor in [-pi, pi]
        rel_ang = wrap_angle_pi(d["theta"] - ref_theta)
        rel_angles.append(round(rel_ang, 5))

    # Construct fixed-length feature vector (truncated/zero-padded to max_nbrs)
    # Shape: (3 * max_nbrs,) -> [norm_distances..., rad_ratios..., rel_angles...]
    feat = np.zeros(3 * max_nbrs, dtype=np.float64)
    k_take = min(len(canonical_ids), max_nbrs)
    feat[0:k_take] = norm_distances[:k_take]
    feat[max_nbrs:max_nbrs + k_take] = rad_ratios[:k_take]
    feat[2 * max_nbrs:2 * max_nbrs + k_take] = rel_angles[:k_take]

    return LocalCraterDescriptor(
        node_id=node_id,
        crater=crater,
        num_neighbors=len(canonical_ids),
        canonical_neighbor_ids=canonical_ids,
        normalized_distances=norm_distances,
        radius_ratios=rad_ratios,
        relative_angles_rad=rel_angles,
        feature_vector=feat,
    )


def build_local_invariant_descriptors(
    graph: CraterGraph,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[int, LocalCraterDescriptor]:
    """
    Build local invariant descriptors for all nodes in CraterGraph.

    Returns:
        Dict mapping node_id -> LocalCraterDescriptor.
    """
    if config is None:
        try:
            from configs.default import Config
            config = getattr(Config, "CRATER_INVARIANTS", {})
        except (ImportError, AttributeError):
            config = {}

    from src.crater_graph.builder import compute_reference_scale
    norm_method = config.get("scale_normalization", "median_radius")
    ref_scale = compute_reference_scale(graph.craters(), method=norm_method)

    descriptors: Dict[int, LocalCraterDescriptor] = {}
    for nid in graph.nodes():
        descriptors[nid] = build_crater_invariant_descriptor(
            graph=graph, node_id=nid, config=config, ref_scale=ref_scale
        )
    return descriptors


def _compute_triangle_permutation(
    c1: CraterCandidate, c2: CraterCandidate, c3: CraterCandidate,
    v1: int, v2: int, v3: int
) -> Optional[Dict[str, Any]]:
    """
    Compute triangle representation for a specific ordered permutation (v1, v2, v3).

    v1, v2, v3 are the vertices in counter-clockwise / clockwise ordering.
    s1 is side opposite v1 (between v2 and v3).
    s2 is side opposite v2 (between v3 and v1).
    s3 is side opposite v3 (between v1 and v2).
    alpha1 is internal angle at v1.
    alpha2 is internal angle at v2.
    alpha3 is internal angle at v3.
    r1, r2, r3 are radii at v1, v2, v3.
    """
    # Side lengths
    s1 = math.sqrt((c2.x - c3.x) ** 2 + (c2.y - c3.y) ** 2)
    s2 = math.sqrt((c3.x - c1.x) ** 2 + (c3.y - c1.y) ** 2)
    s3 = math.sqrt((c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2)

    perimeter = s1 + s2 + s3
    if perimeter <= 1e-9:
        return None

    # Triangle area via cross product
    area = 0.5 * abs(c1.x * (c2.y - c3.y) + c2.x * (c3.y - c1.y) + c3.x * (c1.y - c2.y))

    # Normalized sides
    s1_norm = s1 / perimeter
    s2_norm = s2 / perimeter
    s3_norm = s3 / perimeter

    # Internal angles via Law of Cosines
    # cos(alpha1) = (s2^2 + s3^2 - s1^2) / (2 * s2 * s3)
    def calc_angle(opp: float, adj1: float, adj2: float) -> float:
        denom = 2.0 * adj1 * adj2
        if denom <= 1e-12:
            return 0.0
        val = max(-1.0, min(1.0, (adj1 * adj1 + adj2 * adj2 - opp * opp) / denom))
        return math.acos(val)

    alpha1 = calc_angle(s1, s2, s3)
    alpha2 = calc_angle(s2, s3, s1)
    alpha3 = calc_angle(s3, s1, s2)

    # Normalized crater radii
    r_max = max(c1.radius, c2.radius, c3.radius)
    if r_max > 1e-9:
        r1_norm = c1.radius / r_max
        r2_norm = c2.radius / r_max
        r3_norm = c3.radius / r_max
    else:
        r1_norm = 1.0
        r2_norm = 1.0
        r3_norm = 1.0

    # Strict feature vector maintaining correspondence:
    # index 0, 3, 6 <-> vertex v1 (opposite side s1, internal angle alpha1, radius r1)
    # index 1, 4, 7 <-> vertex v2 (opposite side s2, internal angle alpha2, radius r2)
    # index 2, 5, 8 <-> vertex v3 (opposite side s3, internal angle alpha3, radius r3)
    feat = np.array([
        s1_norm, s2_norm, s3_norm,
        alpha1, alpha2, alpha3,
        r1_norm, r2_norm, r3_norm,
    ], dtype=np.float64)

    # Deterministic comparison key: rounded to 5 decimal places for robust sorting
    # tie-broken by vertex node IDs
    sort_key = (
        round(s1_norm, 5),
        round(s2_norm, 5),
        round(s3_norm, 5),
        round(alpha1, 5),
        round(alpha2, 5),
        round(alpha3, 5),
        round(r1_norm, 5),
        round(r2_norm, 5),
        round(r3_norm, 5),
        v1, v2, v3
    )

    return {
        "nodes": (v1, v2, v3),
        "side_lengths_px": (s1, s2, s3),
        "perimeter_px": perimeter,
        "normalized_sides": (s1_norm, s2_norm, s3_norm),
        "internal_angles_rad": (alpha1, alpha2, alpha3),
        "normalized_radii": (r1_norm, r2_norm, r3_norm),
        "area_px": area,
        "feature_vector": feat,
        "sort_key": sort_key,
    }


def build_triangle_descriptor(
    graph: CraterGraph,
    nodes: Tuple[int, int, int],
    min_area: float = 1.0,
) -> Optional[TriangleDescriptor]:
    """
    Construct a canonical, permutation-invariant TriangleDescriptor for 3 vertices.

    Generates all 6 vertex permutations, evaluates their complete descriptor representation
    (preserving side-angle-radius correspondence), and selects the lexicographically
    smallest representation.

    Args:
        graph: Source CraterGraph.
        nodes: Triple of node IDs (u, v, w).
        min_area: Minimum area in square pixels; triangles smaller than this are filtered.

    Returns:
        Canonical TriangleDescriptor, or None if degenerate.
    """
    if len(nodes) != 3:
        raise ValueError(f"Triangle descriptor requires exactly 3 nodes, got {nodes}.")

    u, v, w = nodes
    cu = graph.get_crater(u)
    cv = graph.get_crater(v)
    cw = graph.get_crater(w)

    candidates_map = {u: cu, v: cv, w: cw}
    permutations = list(itertools.permutations([u, v, w]))

    valid_reps = []
    for p1, p2, p3 in permutations:
        rep = _compute_triangle_permutation(
            c1=candidates_map[p1],
            c2=candidates_map[p2],
            c3=candidates_map[p3],
            v1=p1, v2=p2, v3=p3,
        )
        if rep is not None and rep["area_px"] >= min_area:
            valid_reps.append(rep)

    if not valid_reps:
        return None

    # Pick the lexicographically smallest canonical representation
    best = min(valid_reps, key=lambda r: r["sort_key"])

    return TriangleDescriptor(
        nodes=best["nodes"],
        side_lengths_px=best["side_lengths_px"],
        perimeter_px=best["perimeter_px"],
        normalized_sides=best["normalized_sides"],
        internal_angles_rad=best["internal_angles_rad"],
        normalized_radii=best["normalized_radii"],
        area_px=best["area_px"],
        feature_vector=best["feature_vector"],
    )


def build_triangle_descriptors(
    graph: CraterGraph,
    config: Optional[Dict[str, Any]] = None,
) -> List[TriangleDescriptor]:
    """
    Build canonical TriangleDescriptors for all valid 3-cliques in CraterGraph.

    Args:
        graph: Input CraterGraph instance.
        config: Configuration dictionary (defaults to Config.CRATER_INVARIANTS).

    Returns:
        List of canonical TriangleDescriptor instances, sorted deterministically.
    """
    if config is None:
        try:
            from configs.default import Config
            config = getattr(Config, "CRATER_INVARIANTS", {})
        except (ImportError, AttributeError):
            config = {}

    if not config.get("triangle_enabled", True):
        return []

    min_area = float(config.get("min_triangle_area", 1.0))
    raw_triangles = find_graph_triangles(graph)

    descriptors: List[TriangleDescriptor] = []
    for tri_nodes in raw_triangles:
        td = build_triangle_descriptor(graph, tri_nodes, min_area=min_area)
        if td is not None:
            descriptors.append(td)

    # Sort deterministically by feature vector
    descriptors.sort(key=lambda d: tuple(round(float(x), 5) for x in d.feature_vector))
    return descriptors
