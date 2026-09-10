"""
src/crater_detection/detector.py

Detector abstraction for crater candidate extraction:
- BaseCraterDetector (Abstract Base Class)
- MockCraterDetector (deterministic for testing and pipeline verification)
- HoughCraterDetector (classical computer-vision baseline using cv2.HoughCircles)
- YOLOCraterDetector (deep learning wrapper for YOLO models, isolated from core graph code)
- build_crater_detector (factory function)
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import cv2

from src.crater_detection.types import CraterCandidate, CraterDetectionResult
from src.crater_detection.postprocess import postprocess_craters

logger = logging.getLogger(__name__)

SUPPORTED_DETECTORS = ("mock", "hough", "yolo")


def _ensure_grayscale_uint8(image: np.ndarray) -> np.ndarray:
    """Ensure image is single-channel uint8 in range [0, 255]."""
    if image is None:
        raise ValueError("Image must not be None.")
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Image must be a numpy.ndarray, got {type(image)}.")

    img = image
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.dtype == np.uint8:
        return img
    elif np.issubdtype(img.dtype, np.floating):
        clipped = np.clip(img, 0.0, 1.0)
        return (clipped * 255.0).astype(np.uint8)
    else:
        return img.astype(np.uint8)


class BaseCraterDetector(ABC):
    """
    Abstract Base Class for all crater detection models.
    Downstream components (crater graph, constellation matcher) depend exclusively
    on this interface, never directly on specific machine learning frameworks.
    """

    def __init__(self, name: str = "base_detector"):
        self.name = name

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[CraterCandidate]:
        """
        Detect craters in an input image.

        Args:
            image: 2D or 3D numpy array representing the input image.

        Returns:
            List of standardized CraterCandidate objects in image pixel coordinates.
        """
        pass

    def detect_with_result(self, image: np.ndarray) -> CraterDetectionResult:
        """
        Execute detection and return a CraterDetectionResult with timing and metadata.
        """
        if image is None:
            raise ValueError("Input image must not be None.")

        t0 = time.perf_counter()
        craters = self.detect(image)
        elapsed = time.perf_counter() - t0

        h, w = image.shape[:2]
        return CraterDetectionResult(
            craters=craters,
            detector_name=self.name,
            image_shape=(h, w),
            elapsed_sec=elapsed,
        )


class MockCraterDetector(BaseCraterDetector):
    """
    Deterministic mock detector for unit testing, offline development,
    and software pipeline verification without neural network weights.
    """

    def __init__(self, preset_craters: Optional[List[CraterCandidate]] = None,
                 cfg: Optional[Dict[str, Any]] = None):
        super().__init__(name="mock")
        self.preset_craters = preset_craters
        self.cfg = cfg or {}

    def detect(self, image: np.ndarray) -> List[CraterCandidate]:
        if image is None:
            raise ValueError("Input image must not be None.")

        img_u8 = _ensure_grayscale_uint8(image)
        h, w = img_u8.shape[:2]

        if self.preset_craters is not None:
            raw = list(self.preset_craters)
        else:
            # Deterministic synthetic candidates arranged in a geometric layout
            raw = [
                CraterCandidate(x=w * 0.25, y=h * 0.25, radius=min(w, h) * 0.08, confidence=0.95),
                CraterCandidate(x=w * 0.75, y=h * 0.25, radius=min(w, h) * 0.06, confidence=0.88),
                CraterCandidate(x=w * 0.50, y=h * 0.50, radius=min(w, h) * 0.12, confidence=0.98),
                CraterCandidate(x=w * 0.30, y=h * 0.75, radius=min(w, h) * 0.05, confidence=0.75),
                CraterCandidate(x=w * 0.70, y=h * 0.80, radius=min(w, h) * 0.07, confidence=0.82),
            ]

        min_conf = float(self.cfg.get("confidence_threshold", 0.25))
        min_r = float(self.cfg.get("min_radius_px", 2.0))
        max_r = self.cfg.get("max_radius_px")
        nms_iou = self.cfg.get("nms_iou_threshold", 0.45)
        max_det = self.cfg.get("max_detections", 300)

        return postprocess_craters(
            raw,
            image_shape=(h, w),
            min_confidence=min_conf,
            min_radius_px=min_r,
            max_radius_px=max_r,
            nms_iou_threshold=nms_iou,
            max_detections=max_det,
        )


class HoughCraterDetector(BaseCraterDetector):
    """
    Classical circular feature baseline using OpenCV's HoughCircles.
    NOTE: This is a classical baseline / fallback, NOT a replacement for a learned detector.
    It detects circular edge structures and approximates confidence from local gradient contrast.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        super().__init__(name="hough")
        self.cfg = cfg or {}
        hp = self.cfg.get("hough_params", {})

        self.dp = float(hp.get("dp", 1.2))
        self.min_dist = float(hp.get("min_dist", 20.0))
        self.param1 = float(hp.get("param1", 50.0))  # Canny high threshold
        self.param2 = float(hp.get("param2", 30.0))  # Accumulator threshold
        self.min_radius = int(hp.get("min_radius", 5))
        self.max_radius = int(hp.get("max_radius", 100))

    def detect(self, image: np.ndarray) -> List[CraterCandidate]:
        if image is None:
            raise ValueError("Input image must not be None.")

        img_u8 = _ensure_grayscale_uint8(image)
        h, w = img_u8.shape[:2]

        # Light Gaussian blur to reduce high-frequency noise before circle detection
        blurred = cv2.GaussianBlur(img_u8, (5, 5), 1.5)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )

        raw_candidates: List[CraterCandidate] = []
        if circles is not None and len(circles) > 0:
            circles = np.squeeze(circles, axis=0)
            if circles.ndim == 1:
                circles = circles[np.newaxis, :]

            for cx, cy, r in circles:
                cx, cy, r = float(cx), float(cy), float(r)
                if r <= 0.0:
                    continue

                # Estimate confidence from local edge intensity contrast around the circle
                conf = self._estimate_circle_confidence(img_u8, cx, cy, r)
                raw_candidates.append(CraterCandidate(x=cx, y=cy, radius=r, confidence=conf))

        min_conf = float(self.cfg.get("confidence_threshold", 0.25))
        min_r = float(self.cfg.get("min_radius_px", self.min_radius))
        max_r = self.cfg.get("max_radius_px", self.max_radius)
        nms_iou = self.cfg.get("nms_iou_threshold", 0.45)
        max_det = self.cfg.get("max_detections", 300)

        return postprocess_craters(
            raw_candidates,
            image_shape=(h, w),
            min_confidence=min_conf,
            min_radius_px=min_r,
            max_radius_px=max_r,
            nms_iou_threshold=nms_iou,
            max_detections=max_det,
        )

    def _estimate_circle_confidence(self, img: np.ndarray, cx: float, cy: float, r: float) -> float:
        """
        Estimate a normalized [0.3, 0.95] confidence score based on the contrast
        between the crater interior and its rim neighborhood.
        """
        h, w = img.shape[:2]
        ix, iy, ir = int(round(cx)), int(round(cy)), int(round(r))

        # Sample inner patch
        inner_r = max(1, int(ir * 0.6))
        y1, y2 = max(0, iy - inner_r), min(h, iy + inner_r)
        x1, x2 = max(0, ix - inner_r), min(w, ix + inner_r)

        if y2 <= y1 or x2 <= x1:
            return 0.50

        patch = img[y1:y2, x1:x2]
        std_val = float(np.std(patch))
        # Standard deviation in interior reflects shadowing/depth -> map to [0.35, 0.92]
        conf = 0.35 + min(0.57, (std_val / 64.0) * 0.57)
        return round(float(np.clip(conf, 0.20, 0.95)), 4)


class YOLOCraterDetector(BaseCraterDetector):
    """
    Wrapper for YOLO-based crater detection models (e.g. YOLOv8 / YOLOv5).
    Isolates neural network dependencies from the rest of the matching pipeline.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        super().__init__(name="yolo")
        self.cfg = cfg or {}
        self.model_path = self.cfg.get("model_path")
        self.device = self.cfg.get("device", "cpu")
        self.confidence_threshold = float(self.cfg.get("confidence_threshold", 0.25))
        self.input_size = int(self.cfg.get("input_size", 640))
        self._model = None

    def _load_model(self):
        """Lazy loader for YOLO model with informative failure modes."""
        if self._model is not None:
            return

        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "YOLOCraterDetector requires the 'ultralytics' package. "
                "Install it using 'pip install ultralytics' or configure detector_type='hough' or 'mock'."
            ) from e

        if not self.model_path:
            raise FileNotFoundError(
                "YOLO model_path is not configured in Config.CRATER_DETECTION['model_path']. "
                "Provide a valid path to model weights or select another detector."
            )

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"YOLO model weights file not found at '{self.model_path}'. "
                "Please place the trained weights file at the configured path."
            )

        logger.info("Loading YOLO crater detector from: %s (device=%s)", self.model_path, self.device)
        self._model = YOLO(self.model_path)

    def detect(self, image: np.ndarray) -> List[CraterCandidate]:
        if image is None:
            raise ValueError("Input image must not be None.")

        # Ensure model is ready (will raise if package/weights missing)
        self._load_model()

        img_u8 = _ensure_grayscale_uint8(image)
        h, w = img_u8.shape[:2]

        # Convert to 3-channel for standard YOLO architectures
        img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)

        results = self._model.predict(
            img_bgr,
            conf=self.confidence_threshold,
            imgsz=self.input_size,
            device=self.device,
            verbose=False,
        )

        raw_candidates: List[CraterCandidate] = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cand = CraterCandidate.from_bbox(
                    xmin=float(xyxy[0]),
                    ymin=float(xyxy[1]),
                    xmax=float(xyxy[2]),
                    ymax=float(xyxy[3]),
                    confidence=conf,
                )
                raw_candidates.append(cand)

        min_r = float(self.cfg.get("min_radius_px", 2.0))
        max_r = self.cfg.get("max_radius_px")
        nms_iou = self.cfg.get("nms_iou_threshold", 0.45)
        max_det = self.cfg.get("max_detections", 300)

        return postprocess_craters(
            raw_candidates,
            image_shape=(h, w),
            min_confidence=self.confidence_threshold,
            min_radius_px=min_r,
            max_radius_px=max_r,
            nms_iou_threshold=nms_iou,
            max_detections=max_det,
        )


def build_crater_detector(config: Optional[Dict[str, Any]] = None) -> BaseCraterDetector:
    """
    Factory function to instantiate the configured crater detector.

    Args:
        config: Dictionary containing crater detection settings.
                If None, loads Config.CRATER_DETECTION.

    Returns:
        An instance of BaseCraterDetector (Mock, Hough, or YOLO).
    """
    cfg = {}
    try:
        from configs.default import Config
        cfg.update(Config.CRATER_DETECTION)
    except (ImportError, AttributeError):
        pass
    if config:
        cfg.update(config)

    method = cfg.get("detector_type", "hough").lower()

    if method == "mock":
        return MockCraterDetector(cfg=cfg)
    elif method == "hough":
        return HoughCraterDetector(cfg=cfg)
    elif method == "yolo":
        return YOLOCraterDetector(cfg=cfg)
    else:
        raise ValueError(
            f"Unknown crater detector_type '{method}'. Supported: {SUPPORTED_DETECTORS}"
        )
