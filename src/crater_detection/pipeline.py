"""
src/crater_detection/pipeline.py

High-level crater detection pipeline.
Integrates the Phase 3 PreprocessingPipeline with the crater detector abstraction.
No preprocessing logic is duplicated.
"""

import time
import logging
from typing import Optional, Dict, Any
import numpy as np

from src.preprocessing.pipeline import PreprocessingPipeline
from src.crater_detection.types import CraterDetectionResult
from src.crater_detection.detector import BaseCraterDetector, build_crater_detector

logger = logging.getLogger(__name__)


class CraterDetectionPipeline:
    """
    End-to-end pipeline:
    Raw Image -> Preprocessing (grayscale, CLAHE, normalization) -> Detector -> Post-processing -> Result
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 detector: Optional[BaseCraterDetector] = None):
        """
        Initialize pipeline.

        Args:
            config: Full or crater-specific config dictionary.
                    If None, loads Config defaults.
            detector: Optional pre-built BaseCraterDetector instance.
        """
        self.config = config or {}

        # 1. Initialize Preprocessing Pipeline
        # Preprocessing enhances crater rims (CLAHE) and normalizes sensor differences.
        pp_cfg = self.config.get("preprocessing")
        if pp_cfg is None:
            try:
                from configs.default import Config
                pp_cfg = dict(Config.PREPROCESSING)
            except (ImportError, AttributeError):
                pp_cfg = {"grayscale": True, "clahe_enabled": True, "normalization_method": "minmax"}

        # Pyramid is typically disabled for standard crater detection
        pp_cfg = dict(pp_cfg)
        pp_cfg["pyramid_enabled"] = False
        self.preprocessing = PreprocessingPipeline(pp_cfg)

        # 2. Initialize Crater Detector
        if detector is not None:
            self.detector = detector
        else:
            crater_cfg = self.config.get("crater_detection", self.config)
            self.detector = build_crater_detector(crater_cfg)

        logger.info("CraterDetectionPipeline initialized with detector: %s", self.detector.name)

    def process(self, image: np.ndarray) -> CraterDetectionResult:
        """
        Run the complete crater detection pipeline on an image.

        Args:
            image: Raw input image (numpy array, color or grayscale).

        Returns:
            CraterDetectionResult with detected candidates and metadata.
        """
        if image is None:
            raise ValueError("Input image must not be None.")

        t0 = time.perf_counter()

        # Step 1: Preprocess
        pp_res = self.preprocessing.process(image)
        processed_img = pp_res["image"]

        # Ensure image is uint8 [0, 255] for detector consistency
        if np.issubdtype(processed_img.dtype, np.floating):
            clipped = np.clip(processed_img, 0.0, 1.0)
            img_u8 = (clipped * 255.0).astype(np.uint8)
        else:
            img_u8 = processed_img.astype(np.uint8)

        # Step 2: Detect & Postprocess
        result = self.detector.detect_with_result(img_u8)

        total_elapsed = time.perf_counter() - t0
        result.elapsed_sec = total_elapsed
        logger.info("Pipeline completed: detected %d craters in %.3fs (detector=%s)",
                    len(result.craters), total_elapsed, self.detector.name)
        return result

    def detect(self, image: np.ndarray):
        """
        Run detection pipeline and return list of detected CraterCandidate objects.
        """
        return self.process(image).craters


def detect_craters(image: np.ndarray,
                   config: Optional[Dict[str, Any]] = None,
                   detector: Optional[BaseCraterDetector] = None) -> CraterDetectionResult:
    """
    Convenience function to detect craters in an image.

    Args:
        image: Input image array.
        config: Optional configuration dictionary.
        detector: Optional pre-configured detector instance.

    Returns:
        CraterDetectionResult.
    """
    pipeline = CraterDetectionPipeline(config=config, detector=detector)
    return pipeline.process(image)
