"""
tests/test_streamlit_app.py

Unit tests for Streamlit application helper functions and adapter logic.
Mocks heavy matcher inference to execute quickly in CI test suites.
"""

import io
import os
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

import importlib.util
_spec = importlib.util.spec_from_file_location("match_engine_module", os.path.join(os.path.dirname(__file__), "..", "pages", "1_Match_Engine.py"))
match_engine_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(match_engine_module)

decode_uploaded_image = match_engine_module.decode_uploaded_image
format_match_result = match_engine_module.format_match_result
get_matcher = match_engine_module.get_matcher

from src.matching.hybrid_matcher import HybridMatchResult, HybridMatcher
from src.matching.match_acceptance import MatchAcceptanceResult
from src.matching.transformation import SimilarityTransform2D
from src.registration.registration_engine import RegistrationResult


class TestStreamlitAppHelpers(unittest.TestCase):
    """Test suite for Streamlit helper functions."""

    def test_decode_uploaded_image_valid(self):
        """Test decoding a valid JPEG image buffer."""
        img = np.ones((100, 100), dtype=np.uint8) * 128
        _, encoded = cv2.imencode(".jpg", img)
        buffer = io.BytesIO(encoded.tobytes())

        decoded = decode_uploaded_image(buffer)
        self.assertIsNotNone(decoded)
        self.assertIsInstance(decoded, np.ndarray)
        self.assertEqual(decoded.shape, (100, 100))

    def test_decode_uploaded_image_invalid(self):
        """Test decoding corrupt/invalid bytes returns None."""
        buffer = io.BytesIO(b"not_an_image_file")
        decoded = decode_uploaded_image(buffer)
        self.assertIsNone(decoded)

    def test_decode_uploaded_image_none(self):
        """Test passing None returns None."""
        self.assertIsNone(decode_uploaded_image(None))

    def test_format_match_result_accepted(self):
        """Test formatting an accepted match result into dictionary format."""
        transform = SimilarityTransform2D(scale=1.05, rotation_rad=0.1, translation_x=12.5, translation_y=-4.2)
        acceptance = MatchAcceptanceResult(
            accepted=True,
            status="ACCEPTED",
            reason="All 6 hard evidence criteria passed.",
            checks={
                "minimum_inliers": {"passed": True, "value": 25, "threshold": 10},
                "maximum_rmse": {"passed": True, "value": 1.2, "threshold": 3.0},
            },
            acceptance_score=0.92,
        )
        dummy_corrs = [MagicMock()] * 40
        inliers = list(range(25))
        outliers = list(range(25, 40))

        hybrid_res = HybridMatchResult(
            matched=True,
            correspondences=dummy_corrs,
            inlier_indices=inliers,
            outlier_indices=outliers,
            transform=transform,
            confidence=0.88,
            rmse=1.2,
            inlier_ratio=0.625,
            acceptance=acceptance,
        )

        reg_res = RegistrationResult(
            success=True,
            registered_image=np.zeros((10, 10), dtype=np.uint8),
            valid_mask=np.zeros((10, 10), dtype=np.uint8),
            transform=transform,
            selected_tie_points=[],
            num_inliers=25,
            num_tie_points=25,
            rmse=1.2,
            mean_error=1.0,
            median_error=0.9,
            max_error=2.1,
            coverage=0.75,
            confidence=0.91,
            quality="EXCELLENT",
            metadata={},
        )

        result_dict = format_match_result(hybrid_res, reg_res)

        self.assertTrue(result_dict["matched"])
        self.assertEqual(result_dict["acceptance_status"], "ACCEPTED")
        self.assertAlmostEqual(result_dict["acceptance_score"], 0.92)
        self.assertEqual(result_dict["inliers"], 25)
        self.assertEqual(result_dict["correspondences"], 40)
        self.assertAlmostEqual(result_dict["inlier_ratio"], 0.625)
        self.assertAlmostEqual(result_dict["rmse"], 1.2)
        self.assertAlmostEqual(result_dict["spatial_coverage"], 0.75)
        self.assertEqual(result_dict["quality_category"], "EXCELLENT")
        self.assertEqual(len(result_dict["checks"]), 2)
        self.assertIsNotNone(result_dict["transform"])
        self.assertAlmostEqual(result_dict["transform"]["scale"], 1.05)

    def test_format_match_result_rejected(self):
        """Test formatting a rejected match result into dictionary format."""
        acceptance = MatchAcceptanceResult(
            accepted=False,
            status="REJECTED",
            reason="Insufficient inliers: 3 < 10 threshold.",
            checks={
                "minimum_inliers": {"passed": False, "value": 3, "threshold": 10},
            },
            acceptance_score=0.0,
        )
        dummy_corrs = [MagicMock()] * 35
        inliers = [0, 1, 2]
        outliers = list(range(3, 35))

        hybrid_res = HybridMatchResult(
            matched=False,
            correspondences=dummy_corrs,
            inlier_indices=inliers,
            outlier_indices=outliers,
            transform=None,
            confidence=0.2,
            rmse=2.5,
            inlier_ratio=0.086,
            acceptance=acceptance,
        )

        result_dict = format_match_result(hybrid_res, None)

        self.assertFalse(result_dict["matched"])
        self.assertEqual(result_dict["acceptance_status"], "REJECTED")
        self.assertEqual(result_dict["acceptance_reason"], "Insufficient inliers: 3 < 10 threshold.")
        self.assertAlmostEqual(result_dict["acceptance_score"], 0.0)
        self.assertIsNone(result_dict["transform"])

    @patch.object(match_engine_module, "HybridMatcher")
    def test_get_matcher_caching(self, mock_matcher_cls):
        """Test get_matcher returns instance created by HybridMatcher."""
        mock_instance = MagicMock()
        mock_matcher_cls.return_value = mock_instance

        matcher = get_matcher()
        self.assertIsNotNone(matcher)


if __name__ == "__main__":
    unittest.main()
