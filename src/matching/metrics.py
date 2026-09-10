import cv2
import numpy as np
import logging
import math

logger = logging.getLogger(__name__)

_NO_RESULT = float("nan")


def compute_metrics(kps_a, kps_b, good_matches, geo_result, elapsed_sec=None):
    """
    Compute matching quality metrics.

    Args:
        kps_a / kps_b:  Keypoints from each image.
        good_matches:   Matches after ratio test.
        geo_result:     Output from geometric_verification.verify_geometry().
        elapsed_sec:    Optional wall-clock time for the full pipeline.

    Returns:
        dict with numeric metrics and a validity flag.
    """
    kp_count_a = len(kps_a) if kps_a else 0
    kp_count_b = len(kps_b) if kps_b else 0
    n_candidates = len(good_matches)
    n_inliers = geo_result.get("inlier_count", 0)
    status = geo_result.get("status", "failed")

    inlier_ratio = (n_inliers / n_candidates) if n_candidates > 0 else 0.0

    rmse = _NO_RESULT
    if status == "ok":
        rmse = _reprojection_rmse(kps_a, kps_b,
                                  geo_result["inlier_matches"],
                                  geo_result["transform"],
                                  geo_result["model"])

    metrics = {
        "keypoints_a": kp_count_a,
        "keypoints_b": kp_count_b,
        "candidate_matches": n_candidates,
        "inlier_count": n_inliers,
        "inlier_ratio": round(inlier_ratio, 4),
        "reprojection_rmse_px": round(rmse, 4) if not math.isnan(rmse) else None,
        "geometric_status": status,
        "transform_model": geo_result.get("model"),
        "elapsed_sec": round(elapsed_sec, 3) if elapsed_sec is not None else None,
        "valid": status == "ok" and n_inliers >= 4,
    }
    return metrics


def _reprojection_rmse(kps_a, kps_b, inlier_matches, transform, model):
    """Compute RMSE of inlier reprojection errors under the estimated transform."""
    if not inlier_matches or transform is None:
        return _NO_RESULT

    pts_a = np.float32([kps_a[m.queryIdx].pt for m in inlier_matches]).reshape(-1, 1, 2)
    pts_b = np.float32([kps_b[m.trainIdx].pt for m in inlier_matches]).reshape(-1, 1, 2)

    if model == "affine":
        warped = cv2.transform(pts_a, transform)          # (N,1,2)
    else:
        warped = cv2.perspectiveTransform(pts_a, transform)

    diffs = (warped - pts_b).reshape(-1, 2)
    errors = np.linalg.norm(diffs, axis=1)
    return float(np.sqrt(np.mean(errors ** 2)))
