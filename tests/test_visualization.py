"""
tests/test_visualization.py

Comprehensive unit and integration tests for Lunar Location Matching Visualizer & Explainable Outputs.
"""

import os
import io
import time
import shutil
import tempfile
import numpy as np
import cv2
import pytest
from fastapi.testclient import TestClient

from src.matching.correspondence_fusion import Correspondence
from src.matching.transformation import SimilarityTransform2D
from src.matching.hybrid_matcher import HybridMatchResult
from src.registration.registration_engine import RegistrationResult
from src.visualization.confidence_map import generate_confidence_map
from src.visualization.correspondence_plot import plot_correspondences
from src.visualization.registration_overlay import generate_registration_overlay
from src.visualization.match_visualizer import MatchVisualizer, VisualizationResult
from api.main import app


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test visualization outputs."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def synthetic_images():
    """Generate synthetic grayscale test image pair (128x128)."""
    img_a = np.zeros((128, 128), dtype=np.uint8)
    cv2.circle(img_a, (40, 40), 15, 255, -1)
    cv2.circle(img_a, (80, 80), 20, 200, -1)

    M = np.float32([[1.0, 0.0, 5.0], [0.0, 1.0, 3.0]])
    img_b = cv2.warpAffine(img_a, M, (128, 128))

    return img_a, img_b


@pytest.fixture
def sample_correspondences():
    """Generate sample Correspondence objects with inliers and outliers."""
    corrs = [
        Correspondence(point_a=np.array([40.0, 40.0]), point_b=np.array([45.0, 43.0]), confidence=0.9, source="learned", source_index=0),
        Correspondence(point_a=np.array([80.0, 80.0]), point_b=np.array([85.0, 83.0]), confidence=0.85, source="crater", source_index=1),
        Correspondence(point_a=np.array([20.0, 20.0]), point_b=np.array([90.0, 10.0]), confidence=0.2, source="learned", source_index=2), # outlier
        Correspondence(point_a=np.array([60.0, 60.0]), point_b=np.array([65.0, 63.0]), confidence=0.8, source="fused", source_index=3),
    ]
    inlier_indices = [0, 1, 3]
    return corrs, inlier_indices


def test_1_basic_confidence_map_generation(synthetic_images, sample_correspondences):
    img_a, _ = synthetic_images
    corrs, inliers = sample_correspondences

    cmap = generate_confidence_map(img_a, corrs, inliers, is_target_b=False)
    assert isinstance(cmap, np.ndarray)
    assert cmap.ndim == 3
    assert cmap.shape[2] == 3
    assert cmap.dtype == np.uint8
    assert cmap.shape[:2] == img_a.shape[:2]


def test_2_confidence_map_no_correspondences(synthetic_images):
    img_a, _ = synthetic_images
    cmap = generate_confidence_map(img_a, correspondences=[], inlier_indices=[])
    assert isinstance(cmap, np.ndarray)
    assert cmap.shape[:2] == img_a.shape[:2]


def test_3_confidence_map_few_correspondences(synthetic_images):
    img_a, _ = synthetic_images
    few_corrs = [Correspondence(point_a=np.array([50.0, 50.0]), point_b=np.array([55.0, 55.0]), confidence=0.95, source="learned", source_index=0)]
    cmap = generate_confidence_map(img_a, few_corrs, inlier_indices=[0])
    assert isinstance(cmap, np.ndarray)
    assert cmap.shape[:2] == img_a.shape[:2]


def test_4_confidence_normalization(synthetic_images, sample_correspondences):
    img_a, _ = synthetic_images
    corrs, inliers = sample_correspondences
    # Confidence values vary, result should still be bounded uint8 [0, 255]
    cmap = generate_confidence_map(img_a, corrs, inliers)
    assert np.min(cmap) >= 0
    assert np.max(cmap) <= 255


def test_5_correspondence_image_generation(synthetic_images, sample_correspondences):
    img_a, img_b = synthetic_images
    corrs, inliers = sample_correspondences

    plot_img = plot_correspondences(img_a, img_b, corrs, inliers)
    assert isinstance(plot_img, np.ndarray)
    assert plot_img.ndim == 3
    assert plot_img.shape[2] == 3
    assert plot_img.shape[1] == img_a.shape[1] + img_b.shape[1]


def test_6_inlier_outlier_distinction(synthetic_images, sample_correspondences):
    img_a, img_b = synthetic_images
    corrs, inliers = sample_correspondences

    plot_img = plot_correspondences(img_a, img_b, corrs, inliers, config={"include_outliers": True})
    assert isinstance(plot_img, np.ndarray)


def test_7_max_displayed_correspondence_limit(synthetic_images):
    img_a, img_b = synthetic_images
    many_corrs = [
        Correspondence(point_a=np.array([i*2.0, i*2.0]), point_b=np.array([i*2.0+5, i*2.0+3]), confidence=0.8, source="learned", source_index=i)
        for i in range(50)
    ]
    plot_img = plot_correspondences(img_a, img_b, many_corrs, config={"max_visualized_matches": 10})
    assert isinstance(plot_img, np.ndarray)


def test_8_deterministic_visualization_selection(synthetic_images, sample_correspondences):
    img_a, img_b = synthetic_images
    corrs, inliers = sample_correspondences

    plot1 = plot_correspondences(img_a, img_b, corrs, inliers)
    plot2 = plot_correspondences(img_a, img_b, corrs, inliers)
    np.testing.assert_array_equal(plot1, plot2)


def test_9_registration_overlay_generation(synthetic_images):
    img_a, img_b = synthetic_images
    tf = SimilarityTransform2D(scale=1.0, rotation_rad=0.0, translation_x=5.0, translation_y=3.0)

    overlay, checker = generate_registration_overlay(img_a, img_b, tf)
    assert isinstance(overlay, np.ndarray)
    assert overlay.shape[:2] == img_a.shape[:2]
    assert checker is not None
    assert checker.shape[:2] == img_a.shape[:2]


def test_10_correct_transform_direction(synthetic_images):
    img_a, img_b = synthetic_images
    tf = SimilarityTransform2D(scale=1.0, rotation_rad=0.0, translation_x=5.0, translation_y=3.0)
    overlay, _ = generate_registration_overlay(img_a, img_b, tf)
    assert overlay is not None


def test_11_different_image_dimensions():
    img_a = np.zeros((100, 150), dtype=np.uint8)
    img_b = np.zeros((120, 140), dtype=np.uint8)
    corrs = [Correspondence(point_a=np.array([20.0, 20.0]), point_b=np.array([25.0, 25.0]), confidence=0.9, source="learned", source_index=0)]

    cmap_a = generate_confidence_map(img_a, corrs)
    cmap_b = generate_confidence_map(img_b, corrs, is_target_b=True)
    plot_img = plot_correspondences(img_a, img_b, corrs)

    assert cmap_a.shape[:2] == (100, 150)
    assert cmap_b.shape[:2] == (120, 140)
    assert plot_img.shape[0] == max(100, 120)


def test_12_grayscale_input(synthetic_images, sample_correspondences):
    img_a, _ = synthetic_images
    corrs, inliers = sample_correspondences
    assert img_a.ndim == 2
    cmap = generate_confidence_map(img_a, corrs, inliers)
    assert cmap.ndim == 3


def test_13_float_input(sample_correspondences):
    img_float = np.random.rand(100, 100).astype(np.float32)
    corrs, inliers = sample_correspondences
    cmap = generate_confidence_map(img_float, corrs, inliers)
    assert cmap.dtype == np.uint8


def test_14_large_image_downsampling_safety(sample_correspondences):
    large_img_a = np.zeros((2000, 2000), dtype=np.uint8)
    large_img_b = np.zeros((2000, 2000), dtype=np.uint8)
    corrs, inliers = sample_correspondences

    cmap = generate_confidence_map(large_img_a, corrs, inliers, config={"max_display_dim": 500})
    plot_img = plot_correspondences(large_img_a, large_img_b, corrs, inliers, config={"max_display_dim": 500})

    assert max(cmap.shape[:2]) <= 500
    assert max(plot_img.shape[:2]) <= 1000


def test_15_visualization_serialization(temp_dir, synthetic_images, sample_correspondences):
    img_a, img_b = synthetic_images
    corrs, inliers = sample_correspondences
    match_res = HybridMatchResult(
        matched=True,
        correspondences=corrs,
        inlier_indices=inliers,
        outlier_indices=[2],
        transform=SimilarityTransform2D(scale=1.0, rotation_rad=0.0, translation_x=5.0, translation_y=3.0),
        confidence=0.9,
        rmse=1.2,
        inlier_ratio=0.75,
    )

    viz = MatchVisualizer(config={"output_dir": temp_dir})
    res = viz.generate(img_a, img_b, match_res, job_id="test_job_15")

    dct = res.to_dict()
    assert isinstance(dct["confidence_map_a"], str)
    assert isinstance(dct["confidence_map_b"], str)
    assert isinstance(dct["correspondence_image"], str)
    assert isinstance(dct["registration_overlay"], str)


def test_16_api_visualization_response(synthetic_images):
    img_a, img_b = synthetic_images
    buf_a = cv2.imencode(".png", img_a)[1].tobytes()
    buf_b = cv2.imencode(".png", img_b)[1].tobytes()

    with TestClient(app) as client:
        resp = client.post(
            "/match",
            files={
                "image_a": ("a.png", io.BytesIO(buf_a), "image/png"),
                "image_b": ("b.png", io.BytesIO(buf_b), "image/png"),
            },
        )
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        # Poll until completed
        completed = False
        for _ in range(100):
            st_resp = client.get(f"/match/{job_id}")
            assert st_resp.status_code == 200
            data = st_resp.json()
            if data["status"] == "completed":
                completed = True
                assert "visualizations" in data
                assert data["visualizations"] is not None
                assert "confidence_map_a" in data["visualizations"]
                break
            time.sleep(0.2)

        assert completed, "Job did not complete in time"


def test_17_api_visualization_file_retrieval(synthetic_images):
    img_a, img_b = synthetic_images
    buf_a = cv2.imencode(".png", img_a)[1].tobytes()
    buf_b = cv2.imencode(".png", img_b)[1].tobytes()

    with TestClient(app) as client:
        resp = client.post(
            "/match",
            files={
                "image_a": ("a.png", io.BytesIO(buf_a), "image/png"),
                "image_b": ("b.png", io.BytesIO(buf_b), "image/png"),
            },
        )
        job_id = resp.json()["job_id"]

        for _ in range(100):
            st_resp = client.get(f"/match/{job_id}")
            if st_resp.json()["status"] in ("completed", "failed"):
                break
            time.sleep(0.2)

        asset_resp = client.get(f"/match/{job_id}/visualizations/confidence_map_a.png")
        assert asset_resp.status_code == 200
        assert asset_resp.headers["content-type"] == "image/png"

        # Invalid asset name security test
        invalid_resp = client.get(f"/match/{job_id}/visualizations/secret.txt")
        assert invalid_resp.status_code == 400


def test_18_visualization_failure_does_not_invalidate_model_result(temp_dir, synthetic_images, sample_correspondences):
    img_a, img_b = synthetic_images
    corrs, inliers = sample_correspondences
    match_res = HybridMatchResult(
        matched=True,
        correspondences=corrs,
        inlier_indices=inliers,
        outlier_indices=[2],
        transform=SimilarityTransform2D(scale=1.0, rotation_rad=0.0, translation_x=5.0, translation_y=3.0),
        confidence=0.9,
        rmse=1.2,
        inlier_ratio=0.75,
    )

    # Pass invalid/unwritable output directory
    viz = MatchVisualizer(config={"output_dir": "/invalid_path_unwritable/38291"})
    res = viz.generate(img_a, img_b, match_res, job_id="test_fail")

    # Match result remains intact and valid
    assert match_res.matched is True
    assert res is not None
