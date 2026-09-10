"""
src/matching/feature_detector.py

SIFT and AKAZE feature detectors.
Both expose the same interface: detect_and_compute(image) -> (keypoints, descriptors).
Configuration lives in Config.CLASSICAL_MATCHING.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

SUPPORTED_DETECTORS = ("sift", "akaze")


class SIFTDetector:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError(
                "OpenCV build does not have 'SIFT_create'. "
                "Ensure 'opencv-python>=4.8.0,<5.0.0' is installed."
            )
        sift_cfg = cfg.get("sift", {})
        self._detector = cv2.SIFT_create(
            nfeatures=sift_cfg.get("n_features", 0),
            nOctaveLayers=sift_cfg.get("n_octave_layers", 3),
            contrastThreshold=sift_cfg.get("contrast_threshold", 0.04),
            edgeThreshold=sift_cfg.get("edge_threshold", 10),
            sigma=sift_cfg.get("sigma", 1.6),
        )

    def detect_and_compute(self, image: np.ndarray):
        if image is None:
            raise ValueError("image must not be None.")
        img_u8 = _ensure_uint8(image)
        kps, descs = self._detector.detectAndCompute(img_u8, None)
        logger.debug("SIFT: detected %d keypoints", len(kps))
        return kps, descs


class AKAZEDetector:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        if not hasattr(cv2, "AKAZE_create"):
            raise RuntimeError(
                "OpenCV build does not have 'AKAZE_create'. "
                "Ensure a compatible version such as 'opencv-python>=4.8.0,<5.0.0' is installed."
            )
        akaze_cfg = cfg.get("akaze", {})
        default_desc_type = getattr(cv2, "AKAZE_DESCRIPTOR_MLDB", 5)
        self._detector = cv2.AKAZE_create(
            descriptor_type=akaze_cfg.get("descriptor_type", default_desc_type),
            descriptor_size=akaze_cfg.get("descriptor_size", 0),
            descriptor_channels=akaze_cfg.get("descriptor_channels", 3),
            threshold=akaze_cfg.get("threshold", 0.001),
            nOctaves=akaze_cfg.get("n_octaves", 4),
            nOctaveLayers=akaze_cfg.get("n_octave_layers", 4),
        )

    def detect_and_compute(self, image: np.ndarray):
        if image is None:
            raise ValueError("image must not be None.")
        img_u8 = _ensure_uint8(image)
        kps, descs = self._detector.detectAndCompute(img_u8, None)
        logger.debug("AKAZE: detected %d keypoints", len(kps))
        return kps, descs


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    if image.dtype == np.float32 or image.dtype == np.float64:
        clipped = np.clip(image, 0.0, 1.0)
        return (clipped * 255).astype(np.uint8)
    return image.astype(np.uint8)


def build_detector(method: str, cfg: dict):
    method = method.lower()
    if method == "sift":
        return SIFTDetector(cfg)
    if method == "akaze":
        return AKAZEDetector(cfg)
    raise ValueError(f"Unknown detector '{method}'. Supported: {SUPPORTED_DETECTORS}")
