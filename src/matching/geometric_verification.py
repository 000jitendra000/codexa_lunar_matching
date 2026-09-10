import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = ("affine", "homography")
MIN_MATCHES = {"affine": 3, "homography": 4}


def verify_geometry(kps_a, kps_b, good_matches, model="affine",
                    ransac_threshold=5.0, confidence=0.99, max_iters=2000):
    """
    Estimate affine or homography transform using RANSAC.

    Returns dict: transform, inlier_mask, inlier_matches,
                  inlier_count, status ('ok'/'failed'), model.
    """
    if model not in SUPPORTED_MODELS:
        raise ValueError("Unknown model '{}'. Supported: {}".format(model, SUPPORTED_MODELS))

    required = MIN_MATCHES[model]
    if len(good_matches) < required:
        logger.warning("Insufficient matches for %s RANSAC: %d < %d",
                       model, len(good_matches), required)
        return _failed(model)

    pts_a = np.float32([kps_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_b = np.float32([kps_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    if model == "affine":
        transform, mask = cv2.estimateAffine2D(
            pts_a, pts_b, method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            confidence=confidence, maxIters=max_iters,
        )
    else:
        transform, mask = cv2.findHomography(
            pts_a, pts_b, cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            confidence=confidence, maxIters=max_iters,
        )

    if transform is None or mask is None:
        logger.warning("RANSAC could not estimate %s transform.", model)
        return _failed(model)

    inlier_mask = mask.ravel().astype(bool)
    inlier_matches = [m for m, f in zip(good_matches, inlier_mask) if f]
    logger.debug("%s RANSAC: %d/%d inliers", model, len(inlier_matches), len(good_matches))

    return dict(transform=transform, inlier_mask=inlier_mask,
                inlier_matches=inlier_matches, inlier_count=len(inlier_matches),
                status="ok", model=model)


def _failed(model):
    return dict(transform=None, inlier_mask=None, inlier_matches=[],
                inlier_count=0, status="failed", model=model)
