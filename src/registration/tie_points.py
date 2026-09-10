"""
src/registration/tie_points.py

Uniform spatial tie-point selection module for Milestone B (Registration & Quality Engine).
Enforces a spatially uniform distribution of corresponding points across the overlap domain
using a deterministic regular grid.

ARCHITECTURAL RULES:
- Deterministic regular spatial grid over Image A's coordinate space.
- Retains the highest-confidence candidate per grid cell.
- Enforces an optional global cap (max_points) while preserving spatial distribution.
- Does not bias against or favor any correspondence source (crater vs learned).
- Computes transparent spatial coverage metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from src.matching.correspondence_fusion import Correspondence
from src.registration.inliers import ExtractedInliers


@dataclass
class TiePointSelectionResult:
    """
    Result container for uniform spatial tie-point selection.

    Attributes:
        selected_correspondences: List of chosen Correspondence instances.
        points_a: (M, 2) float64 array of coordinates in Image A.
        points_b: (M, 2) float64 array of coordinates in Image B.
        confidences: (M,) float64 array of confidence scores.
        sources: List of provenance strings ('crater', 'learned').
        num_input_inliers: Number of inliers provided to selection.
        num_selected: Number of final tie points retained.
        num_occupied_cells: Number of distinct grid cells containing inliers.
        total_cells: Total cells in the partitioning grid (rows * cols).
        coverage: Fractional grid cell occupancy in [0.0, 1.0].
        metadata: Detailed diagnostics (cell coordinates, bbox coverage, config).
    """
    selected_correspondences: List[Correspondence]
    points_a: np.ndarray
    points_b: np.ndarray
    confidences: np.ndarray
    sources: List[str]
    num_input_inliers: int
    num_selected: int
    num_occupied_cells: int
    total_cells: int
    coverage: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tie-point selection result to primitive dictionary."""
        return {
            "num_input_inliers": int(self.num_input_inliers),
            "num_selected": int(self.num_selected),
            "num_occupied_cells": int(self.num_occupied_cells),
            "total_cells": int(self.total_cells),
            "coverage": round(float(self.coverage), 4),
            "sources": list(self.sources),
            "confidences": [round(float(c), 4) for c in self.confidences.tolist()],
            "metadata": self.metadata,
        }


def select_uniform_tie_points(
    inliers: ExtractedInliers,
    image_shape: Tuple[int, int],
    config: Optional[Dict[str, Any]] = None,
) -> TiePointSelectionResult:
    """
    Select a spatially uniform subset of verified inliers using a regular spatial grid.

    Args:
        inliers: ExtractedInliers container from extract_verified_inliers.
        image_shape: (height, width) of Image A (reference coordinate frame).
        config: Optional configuration dictionary. Defaults to Config.TIE_POINT_SELECTION.

    Returns:
        TiePointSelectionResult containing uniformly distributed tie points and metrics.
    """
    cfg: Dict[str, Any] = {}
    if config is not None:
        cfg.update(config)
    else:
        try:
            from configs.default import Config
            cfg.update(getattr(Config, "TIE_POINT_SELECTION", {}))
        except (ImportError, AttributeError):
            pass

    enabled: bool = bool(cfg.get("enabled", True))
    grid_rows: int = max(1, int(cfg.get("grid_rows", 6)))
    grid_cols: int = max(1, int(cfg.get("grid_cols", 6)))
    max_points: int = max(1, int(cfg.get("max_points", 30)))
    total_cells = grid_rows * grid_cols

    height, width = int(image_shape[0]), int(image_shape[1])
    num_input = inliers.num_inliers

    if num_input == 0 or height <= 0 or width <= 0:
        return TiePointSelectionResult(
            selected_correspondences=[],
            points_a=np.empty((0, 2), dtype=np.float64),
            points_b=np.empty((0, 2), dtype=np.float64),
            confidences=np.empty((0,), dtype=np.float64),
            sources=[],
            num_input_inliers=num_input,
            num_selected=0,
            num_occupied_cells=0,
            total_cells=total_cells,
            coverage=0.0,
            metadata={"empty_input": True},
        )

    # If uniform selection is disabled, take top inliers up to max_points
    if not enabled:
        indices = np.argsort(-inliers.confidences)[:max_points]
        selected_corrs = [inliers.correspondences[i] for i in indices]
        pts_a = inliers.points_a[indices]
        pts_b = inliers.points_b[indices]
        confs = inliers.confidences[indices]
        srcs = [inliers.sources[i] for i in indices]

        return TiePointSelectionResult(
            selected_correspondences=selected_corrs,
            points_a=pts_a,
            points_b=pts_b,
            confidences=confs,
            sources=srcs,
            num_input_inliers=num_input,
            num_selected=len(selected_corrs),
            num_occupied_cells=0,
            total_cells=total_cells,
            coverage=0.0,
            metadata={"enabled": False},
        )

    cell_h = height / grid_rows
    cell_w = width / grid_cols

    # Assign points to cells
    # cell_map: (r, c) -> list of (inlier_idx, confidence)
    cell_map: Dict[Tuple[int, int], List[Tuple[int, float]]] = {}

    for idx in range(num_input):
        x, y = inliers.points_a[idx]

        # Calculate cell coordinates with bounds clamping
        r = int(y / cell_h)
        c = int(x / cell_w)
        r = min(max(r, 0), grid_rows - 1)
        c = min(max(c, 0), grid_cols - 1)

        cell_key = (r, c)
        if cell_key not in cell_map:
            cell_map[cell_key] = []
        cell_map[cell_key].append((idx, float(inliers.confidences[idx])))

    num_occupied = len(cell_map)
    coverage = float(num_occupied / total_cells)

    # Pick the strongest candidate per cell (tie-break by lowest index for determinism)
    best_per_cell: List[Tuple[Tuple[int, int], int, float]] = []
    for cell_key, candidates in cell_map.items():
        # Sort by confidence descending, then index ascending
        candidates.sort(key=lambda item: (-item[1], item[0]))
        best_idx, best_conf = candidates[0]
        best_per_cell.append((cell_key, best_idx, best_conf))

    # If occupied cells exceed max_points, pick the top max_points cells by candidate confidence
    if len(best_per_cell) > max_points:
        # Sort by confidence descending, then cell (r, c) ascending
        best_per_cell.sort(key=lambda item: (-item[2], item[0]))
        best_per_cell = best_per_cell[:max_points]

    # Deterministic spatial order: sort by cell (r, c) then index
    best_per_cell.sort(key=lambda item: (item[0], item[1]))

    selected_indices = [item[1] for item in best_per_cell]
    selected_corrs = [inliers.correspondences[i] for i in selected_indices]
    pts_a = inliers.points_a[selected_indices]
    pts_b = inliers.points_b[selected_indices]
    confs = inliers.confidences[selected_indices]
    srcs = [inliers.sources[i] for i in selected_indices]

    # Compute bounding-box coverage
    if len(pts_a) > 0:
        min_x, min_y = np.min(pts_a, axis=0)
        max_x, max_y = np.max(pts_a, axis=0)
        bbox_coverage = float(((max_x - min_x) * (max_y - min_y)) / (width * height))
    else:
        bbox_coverage = 0.0

    metadata = {
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "max_points": max_points,
        "bbox_coverage": round(bbox_coverage, 4),
        "occupied_cells": [list(item[0]) for item in best_per_cell],
    }

    return TiePointSelectionResult(
        selected_correspondences=selected_corrs,
        points_a=pts_a,
        points_b=pts_b,
        confidences=confs,
        sources=srcs,
        num_input_inliers=num_input,
        num_selected=len(selected_corrs),
        num_occupied_cells=num_occupied,
        total_cells=total_cells,
        coverage=round(coverage, 4),
        metadata=metadata,
    )
