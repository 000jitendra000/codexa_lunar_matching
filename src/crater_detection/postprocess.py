"""
src/crater_detection/postprocess.py

Post-processing layer for crater candidates:
- Bounding-box to circular crater conversion
- Confidence filtering
- Geometry and boundary validation
- Non-Maximum Suppression (NMS)
- Candidate sorting and maximum count truncation
"""

import math
import logging
from typing import List, Tuple, Optional
from src.crater_detection.types import CraterCandidate

logger = logging.getLogger(__name__)


def bbox_to_crater(xmin: float, ymin: float, xmax: float, ymax: float,
                   confidence: float) -> CraterCandidate:
    """
    Convert bounding box coordinates to a circular CraterCandidate.

    Approximation documentation:
    Planetary craters are approximately circular when viewed from near-nadir.
    Given a detection bounding box [xmin, ymin, xmax, ymax]:
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        radius = min(width, height) / 2
    Note: min(width, height) / 2 is used to avoid overestimating crater radius
    for oblique or elongated detections.
    """
    return CraterCandidate.from_bbox(xmin, ymin, xmax, ymax, confidence)


def filter_by_confidence(craters: List[CraterCandidate],
                         min_confidence: float) -> List[CraterCandidate]:
    """
    Filter out crater candidates with confidence below the threshold.
    """
    if min_confidence <= 0.0:
        return list(craters)
    filtered = [c for c in craters if c.confidence >= min_confidence]
    logger.debug("Confidence filter (>= %.3f): %d -> %d candidates",
                 min_confidence, len(craters), len(filtered))
    return filtered


def filter_invalid(craters: List[CraterCandidate],
                   image_shape: Tuple[int, int],
                   min_radius: float = 2.0,
                   max_radius: Optional[float] = None) -> List[CraterCandidate]:
    """
    Remove invalid crater detections:
    - Centers outside image dimensions
    - Radius below min_radius or above max_radius
    - Non-finite coordinates or radius
    """
    h, w = image_shape[:2]
    valid: List[CraterCandidate] = []

    for c in craters:
        # Check finite
        if not (math.isfinite(c.x) and math.isfinite(c.y) and math.isfinite(c.radius)):
            continue

        # Check radius constraints
        if c.radius < min_radius:
            continue
        if max_radius is not None and c.radius > max_radius:
            continue

        # Check center is within image boundary [0, w) and [0, h)
        if not (0.0 <= c.x < float(w) and 0.0 <= c.y < float(h)):
            continue

        valid.append(c)

    logger.debug("Validation filter (shape=%s, r=[%.1f, %s]): %d -> %d candidates",
                 image_shape, min_radius, max_radius, len(craters), len(valid))
    return valid


def compute_box_iou(box_a: Tuple[float, float, float, float],
                    box_b: Tuple[float, float, float, float]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes
    defined as (xmin, ymin, xmax, ymax).
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h

    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def compute_crater_iou(crater_a: CraterCandidate, crater_b: CraterCandidate) -> float:
    """Compute IoU between two crater candidate bounding boxes."""
    return compute_box_iou(crater_a.bounds, crater_b.bounds)


def apply_crater_nms(craters: List[CraterCandidate],
                     iou_threshold: float = 0.5) -> List[CraterCandidate]:
    """
    Standard greedy Non-Maximum Suppression (NMS) for crater candidates.

    Sorts candidates descending by confidence score, greedily keeps the highest
    scoring candidate, and suppresses any subsequent candidate whose IoU exceeds
    the iou_threshold.
    """
    if len(craters) <= 1:
        return list(craters)

    # Sort descending by confidence
    sorted_craters = sorted(craters, key=lambda c: c.confidence, reverse=True)
    kept: List[CraterCandidate] = []

    for cand in sorted_craters:
        suppress = False
        for k in kept:
            iou = compute_crater_iou(cand, k)
            if iou >= iou_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(cand)

    logger.debug("NMS (iou_thresh=%.2f): %d -> %d candidates",
                 iou_threshold, len(craters), len(kept))
    return kept


def postprocess_craters(craters: List[CraterCandidate],
                        image_shape: Tuple[int, int],
                        min_confidence: float = 0.25,
                        min_radius_px: float = 2.0,
                        max_radius_px: Optional[float] = None,
                        nms_iou_threshold: Optional[float] = 0.45,
                        max_detections: Optional[int] = 300) -> List[CraterCandidate]:
    """
    Full post-processing pipeline for raw crater detections:
    1. Filter by confidence threshold
    2. Filter invalid geometry / boundary out-of-bounds
    3. Apply Non-Maximum Suppression (if nms_iou_threshold is provided)
    4. Sort by confidence descending
    5. Truncate to max_detections (if specified)
    """
    # 1. Confidence filter
    cands = filter_by_confidence(craters, min_confidence)

    # 2. Geometric / boundary filter
    cands = filter_invalid(cands, image_shape, min_radius=min_radius_px, max_radius=max_radius_px)

    # 3. NMS
    if nms_iou_threshold is not None and nms_iou_threshold > 0.0:
        cands = apply_crater_nms(cands, iou_threshold=nms_iou_threshold)

    # 4. Sort descending by confidence
    cands.sort(key=lambda c: c.confidence, reverse=True)

    # 5. Limit max detections
    if max_detections is not None and max_detections > 0:
        cands = cands[:max_detections]

    return cands
