import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def _build_flann():
    return cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))


def _build_bfh():
    return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def match_descriptors(descs_a, descs_b, method, ratio_threshold=0.75):
    """
    Match descriptors and apply Lowe ratio test.

    SIFT: FLANN / L2  |  AKAZE: BruteForce / Hamming
    Returns dict with raw_matches, good_matches, raw_count, good_count.
    """
    if descs_a is None or descs_b is None:
        raise ValueError("Descriptors must not be None.")
    if len(descs_a) < 2 or len(descs_b) < 2:
        raise ValueError(
            "Insufficient descriptors for knnMatch "
            "(A={}, B={}): need >= 2.".format(len(descs_a), len(descs_b))
        )

    if method == "sift":
        raw_knn = _build_flann().knnMatch(
            descs_a.astype(np.float32), descs_b.astype(np.float32), k=2
        )
    elif method == "akaze":
        if np.issubdtype(descs_a.dtype, np.floating) or np.issubdtype(descs_b.dtype, np.floating):
            # Floating-point descriptors (e.g. KAZE / float variant) -> L2 norm
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            raw_knn = matcher.knnMatch(
                descs_a.astype(np.float32), descs_b.astype(np.float32), k=2
            )
        else:
            # Binary descriptors (e.g. MLDB / uint8) -> Hamming norm
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            raw_knn = matcher.knnMatch(descs_a, descs_b, k=2)
    else:
        raise ValueError("Unknown matching method: {}".format(method))

    good = []
    for pair in raw_knn:
        if len(pair) == 2 and pair[0].distance < ratio_threshold * pair[1].distance:
            good.append(pair[0])

    logger.debug("%s: %d raw knn -> %d good after ratio test", method, len(raw_knn), len(good))
    return dict(raw_matches=raw_knn, good_matches=good,
                raw_count=len(raw_knn), good_count=len(good))
