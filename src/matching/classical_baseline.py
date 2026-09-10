import time
import logging
import numpy as np

from src.matching.feature_detector import build_detector
from src.matching.descriptor_matcher import match_descriptors
from src.matching.geometric_verification import verify_geometry
from src.matching.metrics import compute_metrics

logger = logging.getLogger(__name__)


def run_classical_baseline(image_a, image_b, config=None, method="sift"):
    """
    Run the full classical feature-matching baseline pipeline.

    Inputs:
        image_a, image_b : Preprocessed grayscale uint8 numpy arrays.
        config           : Config.CLASSICAL_MATCHING dict (optional, defaults to Config.CLASSICAL_MATCHING).
        method           : 'sift' or 'akaze'.

    Returns:
        dict with keypoints_a/b, matches, inlier_matches, transform,
             metrics, status, method.

    NOTE: This function accepts only images — no coordinates, no metadata.
    """
    if image_a is None or image_b is None:
        raise ValueError("Both image_a and image_b must be provided.")

    if isinstance(config, str):
        method = config
        config = None

    if config is None:
        from configs.default import Config
        config = Config.CLASSICAL_MATCHING

    t0 = time.perf_counter()

    # 1. Detect & compute
    detector = build_detector(method, config)
    kps_a, descs_a = detector.detect_and_compute(image_a)
    kps_b, descs_b = detector.detect_and_compute(image_b)

    # 2. Descriptor matching + ratio test
    if descs_a is None or descs_b is None or len(descs_a) < 2 or len(descs_b) < 2:
        logger.warning("Not enough keypoints detected (A=%d, B=%d). Cannot match.",
                       len(kps_a) if kps_a else 0,
                       len(kps_b) if kps_b else 0)
        return _no_match_result(method, kps_a, kps_b)

    ratio = config.get("ratio_threshold", 0.75)
    match_result = match_descriptors(descs_a, descs_b, method, ratio_threshold=ratio)

    # 3. Geometric verification (RANSAC)
    ransac_cfg = config.get("ransac", {})
    geo_result = verify_geometry(
        kps_a, kps_b,
        match_result["good_matches"],
        model=ransac_cfg.get("model", "affine"),
        ransac_threshold=ransac_cfg.get("threshold", 5.0),
        confidence=ransac_cfg.get("confidence", 0.99),
        max_iters=ransac_cfg.get("max_iters", 2000),
    )

    elapsed = time.perf_counter() - t0

    # 4. Metrics
    metrics = compute_metrics(kps_a, kps_b, match_result["good_matches"],
                              geo_result, elapsed_sec=elapsed)

    return {
        "keypoints_a": kps_a,
        "keypoints_b": kps_b,
        "matches": match_result["good_matches"],
        "inlier_matches": geo_result["inlier_matches"],
        "transform": geo_result["transform"],
        "geo_result": geo_result,
        "metrics": metrics,
        "status": geo_result["status"],
        "method": method,
    }


def _no_match_result(method, kps_a, kps_b):
    from src.matching.geometric_verification import _failed
    geo = _failed("affine")
    metrics = compute_metrics(kps_a or [], kps_b or [], [], geo)
    return dict(keypoints_a=kps_a, keypoints_b=kps_b, matches=[],
                inlier_matches=[], transform=None, geo_result=geo,
                metrics=metrics, status="failed", method=method)
