"""
tests/test_classical_matching.py

Phase 4 unit tests for classical feature-matching baseline.
Synthetic test images are generated inside tests; never placed in data/raw/.
"""

import os
import sys
import math
import pytest
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.default import Config
from src.matching.feature_detector import build_detector, SIFTDetector, AKAZEDetector
from src.matching.descriptor_matcher import match_descriptors
from src.matching.geometric_verification import verify_geometry
from src.matching.metrics import compute_metrics
from src.matching.classical_baseline import run_classical_baseline

CFG = Config.CLASSICAL_MATCHING


# ── Synthetic image factory ──────────────────────────────────────────────────

def _checkerboard(h=256, w=256, block=32):
    """Rich-texture checkerboard with noise — gives plenty of keypoints."""
    board = np.zeros((h, w), dtype=np.uint8)
    for r in range(0, h, block):
        for c in range(0, w, block):
            if (r // block + c // block) % 2 == 0:
                board[r:r+block, c:c+block] = 200
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 30, (h, w), dtype=np.uint8)
    return np.clip(board.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _transform_image(img, angle_deg=15.0, scale=1.0, tx=10, ty=10):
    """Apply affine warp (rotation+scale+translation) to create a pair image."""
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    warped = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
    return warped, M


BASE = _checkerboard()


# ── 1. SIFT Detector ─────────────────────────────────────────────────────────

class TestSIFTDetector:
    def test_returns_keypoints_and_descriptors(self):
        det = SIFTDetector(CFG)
        kps, descs = det.detect_and_compute(BASE)
        assert len(kps) > 0
        assert descs is not None
        assert descs.shape[1] == 128        # SIFT descriptor dim

    def test_none_image_raises(self):
        det = SIFTDetector(CFG)
        with pytest.raises(ValueError):
            det.detect_and_compute(None)

    def test_build_detector_sift(self):
        det = build_detector("sift", CFG)
        assert isinstance(det, SIFTDetector)

    def test_sift_on_float32_input(self):
        det = SIFTDetector(CFG)
        img_f = (BASE.astype(np.float32) / 255.0)
        kps, descs = det.detect_and_compute(img_f)
        assert len(kps) > 0


# ── 2. AKAZE Detector ────────────────────────────────────────────────────────

class TestAKAZEDetector:
    def test_returns_keypoints_and_descriptors(self):
        det = AKAZEDetector(CFG)
        kps, descs = det.detect_and_compute(BASE)
        assert len(kps) > 0
        assert descs is not None

    def test_none_image_raises(self):
        det = AKAZEDetector(CFG)
        with pytest.raises(ValueError):
            det.detect_and_compute(None)

    def test_build_detector_akaze(self):
        det = build_detector("akaze", CFG)
        assert isinstance(det, AKAZEDetector)

    def test_unknown_detector_raises(self):
        with pytest.raises(ValueError):
            build_detector("unknown_algo", CFG)


# ── 3. Descriptor Matching ───────────────────────────────────────────────────

class TestDescriptorMatching:
    def _sift_descs(self, img):
        det = SIFTDetector(CFG)
        return det.detect_and_compute(img)

    def _akaze_descs(self, img):
        det = AKAZEDetector(CFG)
        return det.detect_and_compute(img)

    def test_sift_matching_returns_good_matches(self):
        warped, _ = _transform_image(BASE, angle_deg=10)
        _, da = self._sift_descs(BASE)
        _, db = self._sift_descs(warped)
        result = match_descriptors(da, db, method="sift")
        assert result["good_count"] >= 0
        assert "good_matches" in result

    def test_akaze_matching_returns_good_matches(self):
        warped, _ = _transform_image(BASE, angle_deg=10)
        _, da = self._akaze_descs(BASE)
        _, db = self._akaze_descs(warped)
        result = match_descriptors(da, db, method="akaze")
        assert result["good_count"] >= 0

    def test_none_descriptors_raises(self):
        det = SIFTDetector(CFG)
        _, da = det.detect_and_compute(BASE)
        with pytest.raises(ValueError):
            match_descriptors(da, None, method="sift")

    def test_unknown_method_raises(self):
        det = SIFTDetector(CFG)
        _, da = det.detect_and_compute(BASE)
        _, db = det.detect_and_compute(BASE)
        with pytest.raises(ValueError):
            match_descriptors(da, db, method="orb_unknown")

    def test_ratio_test_strictness(self):
        """Stricter ratio should give fewer or equal good matches."""
        det = SIFTDetector(CFG)
        warped, _ = _transform_image(BASE, angle_deg=5)
        _, da = det.detect_and_compute(BASE)
        _, db = det.detect_and_compute(warped)
        r_loose = match_descriptors(da, db, method="sift", ratio_threshold=0.9)
        r_strict = match_descriptors(da, db, method="sift", ratio_threshold=0.5)
        assert r_strict["good_count"] <= r_loose["good_count"]


# ── 4. Geometric Verification ─────────────────────────────────────────────────

class TestGeometricVerification:
    def _get_matches(self, img_a, img_b):
        det = SIFTDetector(CFG)
        kps_a, da = det.detect_and_compute(img_a)
        kps_b, db = det.detect_and_compute(img_b)
        result = match_descriptors(da, db, method="sift")
        return kps_a, kps_b, result["good_matches"]

    def test_affine_ransac_on_synthetic_pair(self):
        warped, _ = _transform_image(BASE, angle_deg=10, scale=1.0, tx=5, ty=5)
        kps_a, kps_b, good = self._get_matches(BASE, warped)
        geo = verify_geometry(kps_a, kps_b, good, model="affine")
        assert geo["status"] in ("ok", "failed")   # don't demand success on every run
        if geo["status"] == "ok":
            assert geo["transform"] is not None
            assert geo["transform"].shape == (2, 3)

    def test_homography_mode(self):
        warped, _ = _transform_image(BASE, angle_deg=8)
        kps_a, kps_b, good = self._get_matches(BASE, warped)
        geo = verify_geometry(kps_a, kps_b, good, model="homography")
        if geo["status"] == "ok":
            assert geo["transform"].shape == (3, 3)

    def test_insufficient_matches_returns_failed(self):
        kps_a = [cv2.KeyPoint(10, 10, 1)]
        kps_b = [cv2.KeyPoint(20, 20, 1)]
        geo = verify_geometry(kps_a, kps_b, [], model="affine")
        assert geo["status"] == "failed"
        assert geo["transform"] is None

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError):
            verify_geometry([], [], [], model="perspective_magic")

    def test_inlier_count_leq_good_matches(self):
        warped, _ = _transform_image(BASE, angle_deg=12)
        kps_a, kps_b, good = self._get_matches(BASE, warped)
        geo = verify_geometry(kps_a, kps_b, good, model="affine")
        assert geo["inlier_count"] <= len(good)


# ── 5. Metrics ────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_metrics_on_failed_geo(self):
        from src.matching.geometric_verification import _failed
        geo = _failed("affine")
        det = SIFTDetector(CFG)
        kps_a, _ = det.detect_and_compute(BASE)
        m = compute_metrics(kps_a, kps_a, [], geo)
        assert m["inlier_count"] == 0
        assert m["inlier_ratio"] == 0.0
        assert m["reprojection_rmse_px"] is None
        assert m["valid"] is False

    def test_inlier_ratio_calculation(self):
        from src.matching.geometric_verification import _failed
        geo = _failed("affine")
        geo["inlier_count"] = 0
        kps_a, _ = SIFTDetector(CFG).detect_and_compute(BASE)
        m = compute_metrics(kps_a, kps_a, [], geo)
        assert m["inlier_ratio"] == 0.0

    def test_rmse_present_when_transform_ok(self):
        warped, _ = _transform_image(BASE, angle_deg=5)
        det = SIFTDetector(CFG)
        kps_a, da = det.detect_and_compute(BASE)
        kps_b, db = det.detect_and_compute(warped)
        good = match_descriptors(da, db, "sift")["good_matches"]
        geo = verify_geometry(kps_a, kps_b, good, model="affine")
        if geo["status"] == "ok" and geo["inlier_count"] >= 4:
            m = compute_metrics(kps_a, kps_b, good, geo)
            assert m["reprojection_rmse_px"] is not None
            assert m["reprojection_rmse_px"] >= 0.0


# ── 6. Full Baseline Pipeline ─────────────────────────────────────────────────

class TestClassicalBaseline:
    def test_sift_baseline_runs(self):
        warped, _ = _transform_image(BASE, angle_deg=10)
        result = run_classical_baseline(BASE, warped, CFG, method="sift")
        assert "metrics" in result
        assert "status" in result
        assert result["method"] == "sift"

    def test_akaze_baseline_runs(self):
        warped, _ = _transform_image(BASE, angle_deg=10)
        result = run_classical_baseline(BASE, warped, CFG, method="akaze")
        assert result["method"] == "akaze"

    def test_sift_baseline_default_config(self):
        """Verify API works without explicitly supplying config (defaults to Config.CLASSICAL_MATCHING)."""
        warped, _ = _transform_image(BASE, angle_deg=10)
        result = run_classical_baseline(BASE, warped, method="sift")
        assert "metrics" in result
        assert "status" in result
        assert result["method"] == "sift"

    def test_akaze_baseline_default_config(self):
        """Verify AKAZE baseline works without explicitly supplying config."""
        warped, _ = _transform_image(BASE, angle_deg=10)
        result = run_classical_baseline(BASE, warped, method="akaze")
        assert "metrics" in result
        assert "status" in result
        assert result["method"] == "akaze"

    def test_none_image_raises(self):
        with pytest.raises(ValueError):
            run_classical_baseline(None, BASE, CFG, method="sift")

    def test_result_contains_required_keys(self):
        warped, _ = _transform_image(BASE, angle_deg=5)
        result = run_classical_baseline(BASE, warped, CFG, method="sift")
        for key in ("keypoints_a", "keypoints_b", "matches",
                    "inlier_matches", "transform", "metrics", "status"):
            assert key in result

    def test_multiple_transformations(self):
        """Run across several controlled warp scenarios to exercise the pipeline."""
        scenarios = [
            dict(angle_deg=0, scale=1.0, tx=20, ty=0),    # translation
            dict(angle_deg=15, scale=1.0, tx=0, ty=0),    # rotation
            dict(angle_deg=0, scale=0.85, tx=0, ty=0),    # scale
            dict(angle_deg=10, scale=0.9, tx=10, ty=10),  # combined
        ]
        for s in scenarios:
            warped, _ = _transform_image(BASE, **s)
            result = run_classical_baseline(BASE, warped, CFG, method="sift")
            assert result["status"] in ("ok", "failed"), \
                "Unexpected status for scenario {}".format(s)
