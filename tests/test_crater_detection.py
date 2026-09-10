"""
tests/test_crater_detection.py

Phase 5 unit tests for crater candidate extraction, postprocessing,
detector abstractions, pipeline integration, and evaluation metrics.
All test images are generated synthetically; no fake data is placed in data/.
"""

import math
import pytest
import numpy as np
import cv2

from configs.default import Config
from src.crater_detection.types import CraterCandidate, CraterDetectionResult
from src.crater_detection.postprocess import (
    bbox_to_crater,
    filter_by_confidence,
    filter_invalid,
    compute_box_iou,
    compute_crater_iou,
    apply_crater_nms,
    postprocess_craters,
)
from src.crater_detection.detector import (
    MockCraterDetector,
    HoughCraterDetector,
    YOLOCraterDetector,
    build_crater_detector,
)
from src.crater_detection.pipeline import CraterDetectionPipeline, detect_craters
from src.crater_detection.evaluation import compute_crater_detection_metrics


# ── Synthetic image fixture helpers ──────────────────────────────────────────

def _create_synthetic_crater_canvas(h: int = 256, w: int = 256) -> np.ndarray:
    """Create synthetic lunar surface texture with crater-like circular structures."""
    rng = np.random.default_rng(123)
    base = np.full((h, w), 128, dtype=np.uint8)
    noise = rng.integers(-15, 15, (h, w), dtype=np.int16)
    img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Draw a simulated crater (sunlit rim on one side, shadowed interior/rim on other)
    cx, cy, r = 128, 128, 30
    cv2.circle(img, (cx, cy), r, 40, thickness=-1)       # dark interior floor
    cv2.circle(img, (cx, cy), r, 230, thickness=3)       # bright outer rim
    cv2.ellipse(img, (cx, cy), (r, r), 0, 180, 360, 25, thickness=4)  # shadowed inner rim
    return img


# ── 1. CraterCandidate Types & Validation ────────────────────────────────────

class TestCraterCandidateTypes:
    def test_valid_construction_and_properties(self):
        c = CraterCandidate(x=100.0, y=150.0, radius=25.0, confidence=0.85)
        assert c.x == 100.0
        assert c.y == 150.0
        assert c.radius == 25.0
        assert c.confidence == 0.85
        assert c.diameter == 50.0
        assert math.isclose(c.area, math.pi * 25.0 * 25.0)
        assert c.bounds == (75.0, 125.0, 125.0, 175.0)

    def test_invalid_coordinates_raise(self):
        with pytest.raises(ValueError):
            CraterCandidate(x=float("nan"), y=10.0, radius=5.0, confidence=0.9)
        with pytest.raises(ValueError):
            CraterCandidate(x=10.0, y=float("inf"), radius=5.0, confidence=0.9)

    def test_negative_radius_raises(self):
        with pytest.raises(ValueError):
            CraterCandidate(x=10.0, y=10.0, radius=-1.0, confidence=0.9)

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError):
            CraterCandidate(x=10.0, y=10.0, radius=5.0, confidence=-0.1)
        with pytest.raises(ValueError):
            CraterCandidate(x=10.0, y=10.0, radius=5.0, confidence=1.5)

    def test_serialization_roundtrip(self):
        c = CraterCandidate(x=42.5, y=84.2, radius=12.3, confidence=0.9123,
                            bbox=(30.2, 71.9, 54.8, 96.5))
        data = c.to_dict()
        reconstructed = CraterCandidate.from_dict(data)
        assert math.isclose(c.x, reconstructed.x, abs_tol=1e-2)
        assert math.isclose(c.y, reconstructed.y, abs_tol=1e-2)
        assert math.isclose(c.radius, reconstructed.radius, abs_tol=1e-2)
        assert math.isclose(c.confidence, reconstructed.confidence, abs_tol=1e-3)
        assert reconstructed.bbox is not None

    def test_from_bbox_conversion(self):
        # 40x20 bounding box
        c = CraterCandidate.from_bbox(xmin=10.0, ymin=20.0, xmax=50.0, ymax=40.0, confidence=0.88)
        assert c.x == 30.0   # (10 + 50) / 2
        assert c.y == 30.0   # (20 + 40) / 2
        assert c.radius == 10.0  # min(40, 20) / 2 = 10.0
        assert c.confidence == 0.88
        assert c.bbox == (10.0, 20.0, 50.0, 40.0)

    def test_invalid_bbox_raises(self):
        with pytest.raises(ValueError):
            CraterCandidate(x=10, y=10, radius=5, confidence=0.9, bbox=(50, 50, 10, 60))


# ── 2. Post-processing Layer ──────────────────────────────────────────────────

class TestPostprocessing:
    def test_filter_by_confidence(self):
        c1 = CraterCandidate(10, 10, 5, 0.90)
        c2 = CraterCandidate(20, 20, 5, 0.50)
        c3 = CraterCandidate(30, 30, 5, 0.20)
        filtered = filter_by_confidence([c1, c2, c3], min_confidence=0.40)
        assert len(filtered) == 2
        assert c3 not in filtered

    def test_filter_invalid_boundary(self):
        shape = (100, 100)
        valid = CraterCandidate(50, 50, 10, 0.9)
        out_x = CraterCandidate(105, 50, 10, 0.9)
        out_y = CraterCandidate(50, -5, 10, 0.9)
        res = filter_invalid([valid, out_x, out_y], shape)
        assert len(res) == 1
        assert res[0] == valid

    def test_filter_invalid_radius(self):
        shape = (100, 100)
        c_tiny = CraterCandidate(50, 50, 1.0, 0.9)      # below min_radius 3.0
        c_valid = CraterCandidate(50, 50, 10.0, 0.9)
        c_huge = CraterCandidate(50, 50, 80.0, 0.9)     # above max_radius 50.0
        res = filter_invalid([c_tiny, c_valid, c_huge], shape, min_radius=3.0, max_radius=50.0)
        assert len(res) == 1
        assert res[0] == c_valid

    def test_iou_calculation(self):
        # Identical boxes
        b1 = (10.0, 10.0, 30.0, 30.0)
        assert math.isclose(compute_box_iou(b1, b1), 1.0)

        # Disjoint boxes
        b2 = (50.0, 50.0, 70.0, 70.0)
        assert compute_box_iou(b1, b2) == 0.0

        # Known overlap: two 20x20 boxes shifted by 10
        # Overlap is 10x20 = 200, Union = 400 + 400 - 200 = 600 -> IoU = 200/600 = 1/3
        b3 = (20.0, 10.0, 40.0, 30.0)
        assert math.isclose(compute_box_iou(b1, b3), 1.0 / 3.0, abs_tol=1e-4)

    def test_apply_crater_nms(self):
        # Two heavily overlapping candidates
        c_high = CraterCandidate(50.0, 50.0, 15.0, 0.95)
        c_low = CraterCandidate(52.0, 51.0, 15.0, 0.70)
        # One distant candidate
        c_dist = CraterCandidate(150.0, 150.0, 10.0, 0.80)

        kept = apply_crater_nms([c_low, c_dist, c_high], iou_threshold=0.5)
        assert len(kept) == 2
        assert c_high in kept
        assert c_dist in kept
        assert c_low not in kept

    def test_postprocess_craters_full_pipeline(self):
        c1 = CraterCandidate(50.0, 50.0, 10.0, 0.95)
        c2 = CraterCandidate(51.0, 50.0, 10.0, 0.70)   # suppressed by NMS
        c3 = CraterCandidate(200.0, 200.0, 1.0, 0.90)  # suppressed by min_radius
        c4 = CraterCandidate(120.0, 120.0, 15.0, 0.10) # suppressed by min_confidence
        c5 = CraterCandidate(300.0, 50.0, 10.0, 0.85)  # suppressed by boundary (w=256)

        results = postprocess_craters(
            [c1, c2, c3, c4, c5],
            image_shape=(256, 256),
            min_confidence=0.25,
            min_radius_px=3.0,
            nms_iou_threshold=0.45,
            max_detections=10,
        )
        assert len(results) == 1
        assert results[0] == c1

    def test_empty_candidates_handling(self):
        assert filter_by_confidence([], 0.5) == []
        assert filter_invalid([], (100, 100)) == []
        assert apply_crater_nms([], 0.5) == []
        assert postprocess_craters([], (100, 100)) == []


# ── 3. Detector Abstraction & Implementations ─────────────────────────────────

class TestDetectors:
    def test_mock_detector_deterministic(self):
        img = np.zeros((200, 200), dtype=np.uint8)
        detector = MockCraterDetector()
        res = detector.detect_with_result(img)
        assert isinstance(res, CraterDetectionResult)
        assert res.detector_name == "mock"
        assert res.count > 0
        assert res.image_shape == (200, 200)
        assert res.elapsed_sec >= 0.0

    def test_mock_detector_preset_craters(self):
        preset = [
            CraterCandidate(30, 30, 8, 0.9),
            CraterCandidate(80, 80, 12, 0.85),
        ]
        detector = MockCraterDetector(preset_craters=preset)
        craters = detector.detect(np.zeros((100, 100), dtype=np.uint8))
        assert len(craters) == 2

    def test_hough_detector_synthetic_crater(self):
        canvas = _create_synthetic_crater_canvas(256, 256)
        cfg = {
            "hough_params": {
                "dp": 1.2,
                "min_dist": 20,
                "param1": 60,
                "param2": 25,
                "min_radius": 15,
                "max_radius": 50,
            },
            "confidence_threshold": 0.20,
        }
        detector = HoughCraterDetector(cfg=cfg)
        craters = detector.detect(canvas)
        assert isinstance(craters, list)
        # Detects the prominent circular feature drawn on canvas
        assert len(craters) >= 1
        found = craters[0]
        # Should be roughly centered around (128, 128) with radius ~30
        assert math.isclose(found.x, 128, abs_tol=15)
        assert math.isclose(found.y, 128, abs_tol=15)
        assert math.isclose(found.radius, 30, abs_tol=10)

    def test_hough_detector_empty_image(self):
        blank = np.full((128, 128), 128, dtype=np.uint8)
        detector = HoughCraterDetector()
        craters = detector.detect(blank)
        assert len(craters) == 0

    def test_yolo_detector_missing_model_raises_clearly(self):
        # Without ultralytics or valid weights file, YOLOCraterDetector must fail cleanly
        detector = YOLOCraterDetector(cfg={"model_path": "non_existent_weights.pt"})
        with pytest.raises((RuntimeError, FileNotFoundError)) as exc_info:
            detector.detect(np.zeros((64, 64), dtype=np.uint8))
        err_msg = str(exc_info.value)
        assert "ultralytics" in err_msg or "weights file not found" in err_msg

    def test_detector_none_image_raises(self):
        with pytest.raises(ValueError):
            MockCraterDetector().detect(None)
        with pytest.raises(ValueError):
            HoughCraterDetector().detect(None)
        with pytest.raises(ValueError):
            YOLOCraterDetector().detect(None)

    def test_factory_build_crater_detector(self):
        m = build_crater_detector({"detector_type": "mock"})
        assert isinstance(m, MockCraterDetector)

        h = build_crater_detector({"detector_type": "hough"})
        assert isinstance(h, HoughCraterDetector)

        y = build_crater_detector({"detector_type": "yolo"})
        assert isinstance(y, YOLOCraterDetector)

        with pytest.raises(ValueError):
            build_crater_detector({"detector_type": "unsupported_detector_algo"})


# ── 4. Pipeline Integration ───────────────────────────────────────────────────

class TestPipelineIntegration:
    def test_pipeline_process_runs(self):
        canvas = _create_synthetic_crater_canvas(128, 128)
        pipeline = CraterDetectionPipeline(config={"detector_type": "mock"})
        result = pipeline.process(canvas)
        assert isinstance(result, CraterDetectionResult)
        assert result.detector_name == "mock"
        assert result.count > 0
        assert result.image_shape == (128, 128)

    def test_detect_craters_convenience_function(self):
        canvas = _create_synthetic_crater_canvas(128, 128)
        result = detect_craters(canvas, config={"detector_type": "mock"})
        assert result.count > 0
        d = result.to_dict()
        assert "craters" in d
        assert d["detector_name"] == "mock"


# ── 5. Evaluation Infrastructure ──────────────────────────────────────────────

class TestEvaluationMetrics:
    def test_perfect_match(self):
        gt = [CraterCandidate(10, 10, 5, 1.0), CraterCandidate(50, 50, 8, 1.0)]
        preds = [CraterCandidate(10, 10, 5, 0.95), CraterCandidate(50, 50, 8, 0.90)]
        m = compute_crater_detection_metrics(gt, preds, iou_threshold=0.5)
        assert m["true_positives"] == 2
        assert m["false_positives"] == 0
        assert m["false_negatives"] == 0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1_score"] == 1.0
        assert math.isclose(m["mean_iou"], 1.0)

    def test_partial_match(self):
        gt = [
            CraterCandidate(10, 10, 5, 1.0),  # matched
            CraterCandidate(50, 50, 8, 1.0),  # unmatched (FN)
        ]
        preds = [
            CraterCandidate(11, 10, 5, 0.9),   # matches gt[0]
            CraterCandidate(150, 150, 5, 0.8), # unmatched (FP)
        ]
        m = compute_crater_detection_metrics(gt, preds, iou_threshold=0.5)
        assert m["true_positives"] == 1
        assert m["false_positives"] == 1
        assert m["false_negatives"] == 1
        assert m["precision"] == 0.5
        assert m["recall"] == 0.5
        assert m["f1_score"] == 0.5

    def test_empty_gt_and_preds(self):
        m = compute_crater_detection_metrics([], [])
        assert m["true_positives"] == 0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1_score"] == 1.0

    def test_empty_gt_with_preds(self):
        preds = [CraterCandidate(10, 10, 5, 0.9)]
        m = compute_crater_detection_metrics([], preds)
        assert m["true_positives"] == 0
        assert m["false_positives"] == 1
        assert m["precision"] == 0.0

    def test_empty_preds_with_gt(self):
        gt = [CraterCandidate(10, 10, 5, 1.0)]
        m = compute_crater_detection_metrics(gt, [])
        assert m["true_positives"] == 0
        assert m["false_negatives"] == 1
        assert m["recall"] == 0.0
