"""
src/crater_detection package.

Phase 5 crater candidate extraction:
- Standardized CraterCandidate & CraterDetectionResult
- BaseCraterDetector, MockCraterDetector, HoughCraterDetector, YOLOCraterDetector
- Postprocessing (confidence filtering, boundary validation, NMS)
- CraterDetectionPipeline & detect_craters()
- Evaluation metrics infrastructure
"""

from src.crater_detection.types import CraterCandidate, CraterDetectionResult
from src.crater_detection.postprocess import (
    bbox_to_crater,
    filter_by_confidence,
    filter_invalid,
    compute_crater_iou,
    apply_crater_nms,
    postprocess_craters,
)
from src.crater_detection.detector import (
    BaseCraterDetector,
    MockCraterDetector,
    HoughCraterDetector,
    YOLOCraterDetector,
    build_crater_detector,
)
from src.crater_detection.pipeline import CraterDetectionPipeline, detect_craters
from src.crater_detection.evaluation import compute_crater_detection_metrics

__all__ = [
    "CraterCandidate",
    "CraterDetectionResult",
    "bbox_to_crater",
    "filter_by_confidence",
    "filter_invalid",
    "compute_crater_iou",
    "apply_crater_nms",
    "postprocess_craters",
    "BaseCraterDetector",
    "MockCraterDetector",
    "HoughCraterDetector",
    "YOLOCraterDetector",
    "build_crater_detector",
    "CraterDetectionPipeline",
    "detect_craters",
    "compute_crater_detection_metrics",
]
