"""
src/crater_graph/constellation_matcher.py

Local crater constellation matching module using invariant descriptors:
- ConstellationMatch: Data structure for a matched crater pair with confidence and support.
- ConstellationMatchResult: Container for matching results with summary metadata.
- compute_local_descriptor_distance(): Transparent, configurable metric comparing
  compatible non-padded portions of LocalCraterDescriptor.
- count_supporting_triangles(): Efficient O(N_tri) triangle support lookup.
- match_crater_constellations(): Full matching pipeline with ratio test and mutual consistency.

IMPORTANT ARCHITECTURAL RULE:
No image coordinates, pixel scales, or orientation angles are used for candidate generation.
Matching operates strictly on scale- and rotation-invariant descriptor information.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set
import math
from collections import defaultdict
import numpy as np

from src.crater_graph.graph import CraterGraph
from src.crater_graph.invariants import (
    wrap_angle_pi,
    LocalCraterDescriptor,
    TriangleDescriptor,
    build_local_invariant_descriptors,
    build_triangle_descriptors,
)


@dataclass
class ConstellationMatch:
    """
    Candidate correspondence between a source crater and a target crater.

    Attributes:
        source_node_id: Node ID in the source CraterGraph.
        target_node_id: Node ID in the target CraterGraph.
        descriptor_distance: Weighted distance between invariant descriptors.
        confidence: Compatibility/confidence score in [0.0, 1.0].
        matched_neighbor_count: Number of compatible neighbors evaluated.
        supporting_triangle_count: Number of mutually compatible 3-cliques supporting this match.
        component_errors: Detailed breakdown of component discrepancies (dist, radius, angle, count).
    """
    source_node_id: int
    target_node_id: int
    descriptor_distance: float
    confidence: float
    matched_neighbor_count: int
    supporting_triangle_count: int = 0
    component_errors: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize match to primitive dictionary."""
        return {
            "source_node_id": int(self.source_node_id),
            "target_node_id": int(self.target_node_id),
            "descriptor_distance": round(float(self.descriptor_distance), 5),
            "confidence": round(float(self.confidence), 5),
            "matched_neighbor_count": int(self.matched_neighbor_count),
            "supporting_triangle_count": int(self.supporting_triangle_count),
            "component_errors": {k: round(float(v), 5) for k, v in self.component_errors.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConstellationMatch":
        """Deserialize match from primitive dictionary."""
        return cls(
            source_node_id=int(data["source_node_id"]),
            target_node_id=int(data["target_node_id"]),
            descriptor_distance=float(data["descriptor_distance"]),
            confidence=float(data["confidence"]),
            matched_neighbor_count=int(data["matched_neighbor_count"]),
            supporting_triangle_count=int(data.get("supporting_triangle_count", 0)),
            component_errors=dict(data.get("component_errors", {})),
        )


@dataclass
class ConstellationMatchResult:
    """
    Overall result of crater constellation correspondence matching.

    Attributes:
        matches: List of selected ConstellationMatch correspondences.
        source_descriptor_count: Total descriptors available in source graph.
        target_descriptor_count: Total descriptors available in target graph.
        candidate_count: Number of raw candidate pairs before ratio/mutual filtering.
        metadata: Configuration, timing, and diagnostic statistics.
    """
    matches: List[ConstellationMatch]
    source_descriptor_count: int
    target_descriptor_count: int
    candidate_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def match_count(self) -> int:
        return len(self.matches)

    def get_match_pairs(self) -> List[Tuple[int, int]]:
        """Return list of (source_node_id, target_node_id) tuples."""
        return [(m.source_node_id, m.target_node_id) for m in self.matches]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize matching result to primitive dictionary."""
        return {
            "match_count": len(self.matches),
            "source_descriptor_count": int(self.source_descriptor_count),
            "target_descriptor_count": int(self.target_descriptor_count),
            "candidate_count": int(self.candidate_count),
            "metadata": self.metadata,
            "matches": [m.to_dict() for m in self.matches],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConstellationMatchResult":
        """Deserialize matching result from primitive dictionary."""
        matches = [ConstellationMatch.from_dict(m) for m in data.get("matches", [])]
        return cls(
            matches=matches,
            source_descriptor_count=int(data.get("source_descriptor_count", 0)),
            target_descriptor_count=int(data.get("target_descriptor_count", 0)),
            candidate_count=int(data.get("candidate_count", 0)),
            metadata=dict(data.get("metadata", {})),
        )


def compute_local_descriptor_distance(
    desc_a: LocalCraterDescriptor,
    desc_b: LocalCraterDescriptor,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute transparent, deterministic geometric distance between two LocalCraterDescriptors.

    Handling Variable Neighborhood Sizes & Zero Padding:
    - Descriptors are zero-padded up to max_neighbors in their feature_vector.
    - Zero padding is STRICTLY EXCLUDED from comparison.
    - Only the K_compat = min(desc_a.num_neighbors, desc_b.num_neighbors) valid
      canonical neighbors are compared.
    - If K_compat < min_neighbors_for_match (default 2), descriptors are rejected (inf distance).
    - Topological discrepancy is explicitly penalized via the relative neighbor count difference.

    Args:
        desc_a: LocalCraterDescriptor from source graph.
        desc_b: LocalCraterDescriptor from target graph.
        config: Optional configuration dictionary.

    Returns:
        Tuple of (total_descriptor_distance, component_errors_dict).
    """
    if config is None:
        try:
            from configs.default import Config
            config = getattr(Config, "CRATER_MATCHING", {})
        except (ImportError, AttributeError):
            config = {}

    w_dist = float(config.get("distance_weight", 0.35))
    w_radius = float(config.get("radius_weight", 0.25))
    w_angle = float(config.get("angle_weight", 0.25))
    w_count = float(config.get("neighbor_count_weight", 0.15))
    min_nbrs = int(config.get("min_neighbors_for_match", 2))

    ka = desc_a.num_neighbors
    kb = desc_b.num_neighbors
    k_compat = min(ka, kb)

    # Reject if insufficient compatible neighbors
    if k_compat < min_nbrs:
        return float("inf"), {
            "error_dist": float("inf"),
            "error_radius": float("inf"),
            "error_angle": float("inf"),
            "error_count": 1.0,
        }

    # Compare strictly the first k_compat non-padded entries
    dists_a = desc_a.normalized_distances[:k_compat]
    dists_b = desc_b.normalized_distances[:k_compat]
    rads_a = desc_a.radius_ratios[:k_compat]
    rads_b = desc_b.radius_ratios[:k_compat]
    angs_a = desc_a.relative_angles_rad[:k_compat]
    angs_b = desc_b.relative_angles_rad[:k_compat]

    # Mean errors across compatible entries
    e_dist = sum(abs(da - db) for da, db in zip(dists_a, dists_b)) / float(k_compat)
    e_radius = sum(abs(ra - rb) for ra, rb in zip(rads_a, rads_b)) / float(k_compat)
    # Angular error in [0, pi]
    e_angle = sum(abs(wrap_angle_pi(aa - ab)) for aa, ab in zip(angs_a, angs_b)) / float(k_compat)

    # Neighbor count discrepancy penalty in [0.0, 1.0]
    e_count = abs(ka - kb) / float(max(ka, kb, 1))

    total_dist = (
        w_dist * e_dist +
        w_radius * e_radius +
        w_angle * e_angle +
        w_count * e_count
    )

    component_errors = {
        "error_dist": e_dist,
        "error_radius": e_radius,
        "error_angle": e_angle,
        "error_count": e_count,
    }

    return total_dist, component_errors


def count_supporting_triangles(
    candidate_matches: List[Tuple[int, int]],
    triangles_a: List[TriangleDescriptor],
    triangles_b: List[TriangleDescriptor],
    threshold: float = 0.1,
) -> Dict[Tuple[int, int], int]:
    """
    Efficiently compute supporting triangle counts for candidate matches.

    Uses an O(N_tri) hash map lookup on target triangles rather than a combinatorial search.
    For each triangle in Image A whose vertices all belong to candidate matches:
    checks if their corresponding target nodes form a compatible triangle in Image B.

    Args:
        candidate_matches: List of (source_node_id, target_node_id) candidate pairs.
        triangles_a: List of canonical TriangleDescriptors from graph A.
        triangles_b: List of canonical TriangleDescriptors from graph B.
        threshold: Maximum L_inf difference between triangle feature vectors.

    Returns:
        Dict mapping (source_id, target_id) -> supporting_triangle_count.
    """
    support_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    if not triangles_a or not triangles_b or not candidate_matches:
        return dict(support_counts)

    # Map target 3-cliques: frozenset({u, v, w}) -> TriangleDescriptor
    b_tri_map: Dict[frozenset, TriangleDescriptor] = {}
    for tb in triangles_b:
        b_tri_map[frozenset(tb.nodes)] = tb

    # Candidate mapping: source_node_id -> target_node_id
    src_to_tgt: Dict[int, int] = {s: t for s, t in candidate_matches}

    for ta in triangles_a:
        u_a, v_a, w_a = ta.nodes
        if u_a in src_to_tgt and v_a in src_to_tgt and w_a in src_to_tgt:
            u_b = src_to_tgt[u_a]
            v_b = src_to_tgt[v_a]
            w_b = src_to_tgt[w_a]

            # All 3 target vertices must be distinct
            if len({u_b, v_b, w_b}) == 3:
                tgt_set = frozenset([u_b, v_b, w_b])
                if tgt_set in b_tri_map:
                    tb = b_tri_map[tgt_set]
                    # Verify feature vector compatibility
                    diff = float(np.max(np.abs(ta.feature_vector - tb.feature_vector)))
                    if diff <= threshold:
                        support_counts[(u_a, u_b)] += 1
                        support_counts[(v_a, v_b)] += 1
                        support_counts[(w_a, w_b)] += 1

    return dict(support_counts)


def match_crater_constellations(
    graph_a: CraterGraph,
    graph_b: CraterGraph,
    descriptors_a: Optional[Dict[int, LocalCraterDescriptor]] = None,
    descriptors_b: Optional[Dict[int, LocalCraterDescriptor]] = None,
    triangles_a: Optional[List[TriangleDescriptor]] = None,
    triangles_b: Optional[List[TriangleDescriptor]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> ConstellationMatchResult:
    """
    Match crater constellations between Image A and Image B using invariant descriptors.

    Pipeline:
    1. Extract local and triangle descriptors for Graph A and Graph B if not supplied.
    2. Compute pairwise descriptor distances between all compatible nodes (excluding padding).
    3. Filter candidate pairs with distance <= max_descriptor_distance.
    4. Apply Lowe's ratio test (best vs second-best target distance per source node).
    5. Apply mutual (bidirectional) consistency: (A->B and B->A agreement).
    6. Compute triangle support for candidate matches using 3-cliques.
    7. Calculate confidence scores combining descriptor distance and triangle support.
    8. Deterministically rank matches by (-confidence, distance, source_id, target_id).

    Args:
        graph_a: CraterGraph for Image A.
        graph_b: CraterGraph for Image B.
        descriptors_a: Optional precomputed local descriptors for A.
        descriptors_b: Optional precomputed local descriptors for B.
        triangles_a: Optional precomputed triangle descriptors for A.
        triangles_b: Optional precomputed triangle descriptors for B.
        config: Configuration dictionary (defaults to Config.CRATER_MATCHING).

    Returns:
        ConstellationMatchResult containing verified correspondences.
    """
    if config is None:
        try:
            from configs.default import Config
            config = getattr(Config, "CRATER_MATCHING", {})
        except (ImportError, AttributeError):
            config = {}

    max_dist_thresh = float(config.get("max_descriptor_distance", 0.5))
    ratio_thresh = config.get("ratio_test_threshold", 0.85)
    mutual_consistency = bool(config.get("mutual_consistency", True))
    tri_support_enabled = bool(config.get("triangle_support_enabled", True))
    tri_dist_thresh = float(config.get("triangle_distance_threshold", 0.15))
    sigma_dist = float(config.get("confidence_sigma", 0.3))

    # Precompute descriptors if needed
    if descriptors_a is None:
        descriptors_a = build_local_invariant_descriptors(graph_a)
    if descriptors_b is None:
        descriptors_b = build_local_invariant_descriptors(graph_b)

    num_desc_a = len(descriptors_a)
    num_desc_b = len(descriptors_b)

    if num_desc_a == 0 or num_desc_b == 0:
        return ConstellationMatchResult(
            matches=[],
            source_descriptor_count=num_desc_a,
            target_descriptor_count=num_desc_b,
            candidate_count=0,
            metadata={"status": "empty_inputs"},
        )

    # Precompute triangles if triangle support is enabled
    if tri_support_enabled:
        if triangles_a is None:
            triangles_a = build_triangle_descriptors(graph_a)
        if triangles_b is None:
            triangles_b = build_triangle_descriptors(graph_b)
    else:
        triangles_a = []
        triangles_b = []

    # Step 1: Compute pairwise distances
    # a_candidates[src_id] = [(dist, tgt_id, comp_errors), ...]
    # b_candidates[tgt_id] = [(dist, src_id, comp_errors), ...]
    a_candidates: Dict[int, List[Tuple[float, int, Dict[str, float]]]] = defaultdict(list)
    b_candidates: Dict[int, List[Tuple[float, int, Dict[str, float]]]] = defaultdict(list)
    total_raw_candidates = 0

    src_nodes = sorted(descriptors_a.keys())
    tgt_nodes = sorted(descriptors_b.keys())

    for u in src_nodes:
        da = descriptors_a[u]
        for v in tgt_nodes:
            db = descriptors_b[v]
            dist, comp_err = compute_local_descriptor_distance(da, db, config=config)
            if dist <= max_dist_thresh:
                total_raw_candidates += 1
                a_candidates[u].append((dist, v, comp_err))
                b_candidates[v].append((dist, u, comp_err))

    # Step 2: Sort candidates deterministically
    for u in a_candidates:
        a_candidates[u].sort(key=lambda item: (round(item[0], 5), item[1]))
    for v in b_candidates:
        b_candidates[v].sort(key=lambda item: (round(item[0], 5), item[1]))

    # Step 3: Apply ratio test on source candidates
    filtered_a_to_b: Dict[int, Tuple[float, int, Dict[str, float]]] = {}
    for u, cand_list in a_candidates.items():
        if not cand_list:
            continue
        best_dist, best_v, best_err = cand_list[0]

        if ratio_thresh is not None and len(cand_list) >= 2:
            second_dist = cand_list[1][0]
            # If best is too close to second best, match is ambiguous
            if second_dist > 1e-9 and (best_dist / second_dist) > ratio_thresh:
                continue

        filtered_a_to_b[u] = (best_dist, best_v, best_err)

    # Step 4: Apply mutual consistency if enabled
    confirmed_pairs: List[Tuple[int, int, float, Dict[str, float]]] = []
    for u, (best_dist, best_v, comp_err) in filtered_a_to_b.items():
        if mutual_consistency:
            # Check if u is also the best match for best_v in b_candidates
            b_list = b_candidates.get(best_v, [])
            if not b_list:
                continue
            best_u_for_v = b_list[0][1]
            if best_u_for_v != u:
                # Asymmetric match -> reject
                continue
        confirmed_pairs.append((u, best_v, best_dist, comp_err))

    # Deterministic order of confirmed pairs: (u, v)
    confirmed_pairs.sort(key=lambda p: (p[0], p[1]))

    # Step 5: Compute supporting triangles for confirmed pairs
    candidate_pair_tuples = [(u, v) for u, v, _, _ in confirmed_pairs]
    support_counts = count_supporting_triangles(
        candidate_matches=candidate_pair_tuples,
        triangles_a=triangles_a,
        triangles_b=triangles_b,
        threshold=tri_dist_thresh,
    )

    # Step 6: Construct ConstellationMatch objects with confidence
    matches: List[ConstellationMatch] = []
    for u, v, dist, comp_err in confirmed_pairs:
        tri_count = support_counts.get((u, v), 0)
        da = descriptors_a[u]
        db = descriptors_b[v]
        k_compat = min(da.num_neighbors, db.num_neighbors)

        # Confidence formulation
        base_conf = math.exp(-dist / max(1e-4, sigma_dist))
        tri_bonus = min(0.25, float(tri_count) * 0.08)
        confidence = min(1.0, max(0.0, base_conf + tri_bonus))

        matches.append(
            ConstellationMatch(
                source_node_id=u,
                target_node_id=v,
                descriptor_distance=dist,
                confidence=confidence,
                matched_neighbor_count=k_compat,
                supporting_triangle_count=tri_count,
                component_errors=comp_err,
            )
        )

    # Step 7: Deterministic final ordering:
    # 1. descending confidence (rounded to 5 decimals)
    # 2. ascending descriptor distance
    # 3. source_node_id, target_node_id
    matches.sort(
        key=lambda m: (
            round(-m.confidence, 5),
            round(m.descriptor_distance, 5),
            m.source_node_id,
            m.target_node_id,
        )
    )

    return ConstellationMatchResult(
        matches=matches,
        source_descriptor_count=num_desc_a,
        target_descriptor_count=num_desc_b,
        candidate_count=total_raw_candidates,
        metadata={
            "mutual_consistency": mutual_consistency,
            "ratio_test_threshold": ratio_thresh,
            "max_descriptor_distance": max_dist_thresh,
            "triangle_support_enabled": tri_support_enabled,
            "source_triangles_count": len(triangles_a),
            "target_triangles_count": len(triangles_b),
        },
    )
