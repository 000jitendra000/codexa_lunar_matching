"""
src/matching/correspondence_fusion.py

Correspondence fusion module for the Cross-Sensor Lunar Matching hybrid engine:
- Correspondence: Common unified data structure for both classical crater and learned deep correspondences.
- crater_matches_to_correspondences(): Adapter converting Phase 8 ConstellationMatch objects.
- learned_matches_to_correspondences(): Adapter converting LearnedMatchResult objects.
- fuse_correspondences(): Deterministic fusion procedure removing duplicates, invalid points,
  and balancing sources without performing geometric RANSAC.
- correspondences_to_arrays(): Utility converting correspondences to coordinate arrays for Phase 10 RANSAC.

ARCHITECTURAL RULES:
- Both branches map to the identical Correspondence schema.
- Fusion is strictly deterministic.
- Does NOT perform geometric filtering or RANSAC (reserved for Phase 10 / final stage).
- Preserves source provenance ('crater' vs 'learned').
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import math
import numpy as np

from src.crater_graph.graph import CraterGraph
from src.crater_graph.constellation_matcher import ConstellationMatch
from src.matching.learned_matcher import LearnedCorrespondence, LearnedMatchResult


@dataclass
class Correspondence:
    """
    Common unified representation for a point-to-point candidate correspondence.

    Attributes:
        point_a: (2,) array [x, y] coordinate in Image A.
        point_b: (2,) array [x, y] coordinate in Image B.
        confidence: Confidence score in [0.0, 1.0].
        source: Provenance label ('crater', 'learned', etc.).
        source_index: Original 0-indexed position within the generating branch.
        metadata: Branch-specific diagnostic metadata.
    """
    point_a: np.ndarray
    point_b: np.ndarray
    confidence: float
    source: str
    source_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.point_a = np.asarray(self.point_a, dtype=np.float64).flatten()
        self.point_b = np.asarray(self.point_b, dtype=np.float64).flatten()
        if self.point_a.shape != (2,):
            raise ValueError(f"point_a must have shape (2,), got {self.point_a.shape}.")
        if self.point_b.shape != (2,):
            raise ValueError(f"point_b must have shape (2,), got {self.point_b.shape}.")
        self.confidence = float(self.confidence)
        self.source = str(self.source)
        self.source_index = int(self.source_index)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize correspondence to primitive dictionary."""
        return {
            "point_a": [round(float(v), 4) for v in self.point_a],
            "point_b": [round(float(v), 4) for v in self.point_b],
            "confidence": round(float(self.confidence), 4),
            "source": self.source,
            "source_index": int(self.source_index),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Correspondence":
        """Deserialize correspondence from dictionary."""
        return cls(
            point_a=np.array(data["point_a"], dtype=np.float64),
            point_b=np.array(data["point_b"], dtype=np.float64),
            confidence=float(data["confidence"]),
            source=str(data.get("source", "unknown")),
            source_index=int(data.get("source_index", 0)),
            metadata=dict(data.get("metadata", {})),
        )


def crater_matches_to_correspondences(
    graph_a: CraterGraph,
    graph_b: CraterGraph,
    matches: List[ConstellationMatch],
) -> List[Correspondence]:
    """
    Convert Phase 8 ConstellationMatch objects to common Correspondence representation.

    Args:
        graph_a: Source CraterGraph.
        graph_b: Target CraterGraph.
        matches: List of ConstellationMatch candidate correspondences.

    Returns:
        List of Correspondence objects with source='crater'.
    """
    correspondences: List[Correspondence] = []
    for idx, m in enumerate(matches):
        ca = graph_a.get_crater(m.source_node_id)
        cb = graph_b.get_crater(m.target_node_id)
        correspondences.append(
            Correspondence(
                point_a=np.array([ca.x, ca.y], dtype=np.float64),
                point_b=np.array([cb.x, cb.y], dtype=np.float64),
                confidence=float(m.confidence),
                source="crater",
                source_index=idx,
                metadata={
                    "source_crater_id": m.source_node_id,
                    "target_crater_id": m.target_node_id,
                    "descriptor_distance": m.descriptor_distance,
                    "supporting_triangles": m.supporting_triangle_count,
                },
            )
        )
    return correspondences


def learned_matches_to_correspondences(
    learned_result: LearnedMatchResult,
) -> List[Correspondence]:
    """
    Convert LearnedMatchResult correspondences to common Correspondence representation.

    Args:
        learned_result: LearnedMatchResult from BaseLearnedMatcher.

    Returns:
        List of Correspondence objects with source='learned'.
    """
    correspondences: List[Correspondence] = []
    for idx, lc in enumerate(learned_result.correspondences):
        correspondences.append(
            Correspondence(
                point_a=lc.point_a.copy(),
                point_b=lc.point_b.copy(),
                confidence=float(lc.confidence),
                source="learned",
                source_index=idx,
                metadata={"backend": learned_result.backend},
            )
        )
    return correspondences


def fuse_correspondences(
    crater_correspondences: List[Correspondence],
    learned_correspondences: List[Correspondence],
    config: Optional[Dict[str, Any]] = None,
    image_shape_a: Optional[Tuple[int, int]] = None,
    image_shape_b: Optional[Tuple[int, int]] = None,
    **kwargs,
) -> List[Correspondence]:
    """
    Deterministically fuse candidate correspondences from crater and learned branches.

    Fusion Steps:
    1. Filter invalid coordinates (NaN, Inf, or out-of-bounds if image shapes provided).
    2. Source balancing: truncate each branch to max_correspondences_per_source by confidence.
    3. Concatenate and sort candidate pool by confidence descending.
    4. Remove spatial duplicates within duplicate_tolerance_px, retaining the higher-confidence match.
       Deterministic tie-break: prefer 'crater' over 'learned', then lower source_index.

    Args:
        crater_correspondences: List of crater branch correspondences.
        learned_correspondences: List of learned branch correspondences.
        config: Optional configuration dictionary.
        image_shape_a: Optional (height, width) of Image A.
        image_shape_b: Optional (height, width) of Image B.
        **kwargs: Overrides for configuration settings.

    Returns:
        List of fused, deduplicated, and sorted Correspondence objects.
    """
    cfg: Dict[str, Any] = {}
    if config is not None:
        cfg.update(config)
    else:
        try:
            from configs.default import Config
            cfg.update(getattr(Config, "CORRESPONDENCE_FUSION", {}))
        except (ImportError, AttributeError):
            pass
    cfg.update(kwargs)

    duplicate_tol = float(cfg.get("duplicate_tolerance_px", 4.0))
    max_per_source = int(cfg.get("max_correspondences_per_source", 300))
    crater_mult = float(cfg.get("crater_weight_multiplier", 1.0))
    learned_mult = float(cfg.get("learned_weight_multiplier", 1.0))

    def is_valid(c: Correspondence) -> bool:
        if not math.isfinite(c.confidence) or c.confidence < 0.0:
            return False
        pa, pb = c.point_a, c.point_b
        if not (math.isfinite(pa[0]) and math.isfinite(pa[1])):
            return False
        if not (math.isfinite(pb[0]) and math.isfinite(pb[1])):
            return False
        if image_shape_a is not None:
            ha, wa = image_shape_a
            if pa[0] < 0.0 or pa[1] < 0.0 or pa[0] >= wa or pa[1] >= ha:
                return False
        if image_shape_b is not None:
            hb, wb = image_shape_b
            if pb[0] < 0.0 or pb[1] < 0.0 or pb[0] >= wb or pb[1] >= hb:
                return False
        return True

    # 1. Validate inputs
    valid_crater = [c for c in crater_correspondences if is_valid(c)]
    valid_learned = [c for c in learned_correspondences if is_valid(c)]

    # 2. Source balancing: sort and cap per source
    if max_per_source > 0:
        valid_crater = sorted(valid_crater, key=lambda c: c.confidence, reverse=True)[:max_per_source]
        valid_learned = sorted(valid_learned, key=lambda c: c.confidence, reverse=True)[:max_per_source]

    # Apply confidence multipliers
    pool: List[Correspondence] = []
    for c in valid_crater:
        c_copy = Correspondence(
            point_a=c.point_a.copy(),
            point_b=c.point_b.copy(),
            confidence=min(1.0, c.confidence * crater_mult),
            source=c.source,
            source_index=c.source_index,
            metadata=dict(c.metadata),
        )
        pool.append(c_copy)

    for c in valid_learned:
        c_copy = Correspondence(
            point_a=c.point_a.copy(),
            point_b=c.point_b.copy(),
            confidence=min(1.0, c.confidence * learned_mult),
            source=c.source,
            source_index=c.source_index,
            metadata=dict(c.metadata),
        )
        pool.append(c_copy)

    if not pool:
        return []

    # 3. Deterministic candidate ordering before deduplication:
    #    Primary: confidence descending
    #    Secondary: prefer 'crater' over 'learned'
    #    Tertiary: source_index ascending
    def sort_key(c: Correspondence):
        source_order = 0 if c.source == "crater" else 1
        return (-c.confidence, source_order, c.source_index)

    pool = sorted(pool, key=sort_key)

    # 4. Duplicate removal within spatial tolerance
    fused: List[Correspondence] = []
    tol_sq = duplicate_tol * duplicate_tol

    for cand in pool:
        is_duplicate = False
        for accepted in fused:
            d_a_sq = float(np.sum((cand.point_a - accepted.point_a) ** 2))
            d_b_sq = float(np.sum((cand.point_b - accepted.point_b) ** 2))
            if d_a_sq <= tol_sq and d_b_sq <= tol_sq:
                is_duplicate = True
                break
        if not is_duplicate:
            fused.append(cand)

    return fused


def correspondences_to_arrays(
    correspondences: List[Correspondence],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract coordinate and weight arrays from a list of Correspondence objects.

    Returns:
        Tuple of:
            points_a: (N, 2) float64 array of coordinates in Image A.
            points_b: (N, 2) float64 array of coordinates in Image B.
            weights: (N,) float64 array of confidence weights.
    """
    if not correspondences:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.float64)

    pts_a = np.array([c.point_a for c in correspondences], dtype=np.float64)
    pts_b = np.array([c.point_b for c in correspondences], dtype=np.float64)
    weights = np.array([max(c.confidence, 1e-4) for c in correspondences], dtype=np.float64)
    return pts_a, pts_b, weights
