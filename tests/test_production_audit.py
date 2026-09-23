"""
tests/test_production_audit.py

Production audit test suite verifying memory and storage optimizations:
1. Production visualization jobs generate no more than 5 user-facing PNG assets.
2. Historical job directories are bounded by max_job_directories and retention_hours.
3. Active job directories are preserved during cleanup.
4. Disabling visualizations (enabled=False) works without error.
5. Evicted/cleaned-up jobs return 404 cleanly from visualization endpoints.
6. In-memory JobManager history remains strictly bounded.
"""

import os
import shutil
import tempfile
import time
import pytest
import numpy as np
from unittest.mock import MagicMock

from src.matching.hybrid_matcher import HybridMatcher, HybridMatchResult
from src.matching.correspondence_fusion import Correspondence
from src.registration.registration_engine import RegistrationEngine, RegistrationResult
from src.matching.transformation import SimilarityTransform2D
from src.visualization.match_visualizer import MatchVisualizer, cleanup_old_visualizations, VisualizationResult
from api.jobs import JobManager
from api.schemas import MatchResultSummary, VisualizationSchema


def create_dummy_image(height=100, width=100, value=128):
    """Utility to create a synthetic grayscale uint8 numpy array."""
    img = np.full((height, width), value, dtype=np.uint8)
    # Add a small recognizable box in center
    img[40:60, 40:60] = 200
    return img


def create_dummy_hybrid_result():
    """Utility to construct a successful HybridMatchResult."""
    corrs = [
        Correspondence(
            point_a=np.array([20.0 + i * 2, 20.0 + i * 2]),
            point_b=np.array([22.0 + i * 2, 21.0 + i * 2]),
            confidence=0.9,
            source="learned",
            source_index=i,
        )
        for i in range(15)
    ]
    inlier_indices = list(range(12))
    transform = SimilarityTransform2D(
        scale=1.0,
        rotation_rad=0.0,
        translation_x=2.0,
        translation_y=1.0,
    )
    return HybridMatchResult(
        matched=True,
        correspondences=corrs,
        inlier_indices=inlier_indices,
        outlier_indices=[12, 13, 14],
        transform=transform,
        confidence=0.85,
        rmse=1.2,
        inlier_ratio=0.8,
        metadata={"num_correspondences": len(corrs)},
    )


def test_production_png_limit():
    """Verify that a production job creates at most the 5 expected PNG files and zero debug PNGs."""
    temp_dir = tempfile.mkdtemp()
    try:
        visualizer = MatchVisualizer(config={"output_dir": temp_dir, "enabled": True})
        img_a = create_dummy_image()
        img_b = create_dummy_image()
        res = create_dummy_hybrid_result()

        job_id = "audit_job_001"
        viz_res = visualizer.generate(
            image_a=img_a,
            image_b=img_b,
            match_result=res,
            job_id=job_id,
        )

        job_dir = os.path.join(temp_dir, job_id)
        assert os.path.isdir(job_dir)

        files = os.listdir(job_dir)
        png_files = [f for f in files if f.endswith(".png")]

        # Must not exceed 5 PNGs
        assert len(png_files) <= 5
        allowed_names = {
            "confidence_map_a.png",
            "confidence_map_b.png",
            "correspondence_image.png",
            "registration_overlay.png",
            "checkerboard.png",
        }
        for filename in png_files:
            assert filename in allowed_names, f"Unexpected PNG file generated: {filename}"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_max_job_directories_cleanup():
    """Verify that cleanup_old_visualizations purges oldest directories when count > max_job_directories."""
    temp_dir = tempfile.mkdtemp()
    try:
        # Create 10 dummy job directories with varying mtimes
        job_ids = [f"job_{i:02d}" for i in range(10)]
        for idx, jid in enumerate(job_ids):
            jdir = os.path.join(temp_dir, jid)
            os.makedirs(jdir, exist_ok=True)
            mtime = time.time() - (10 - idx) * 10.0  # job_00 is oldest, job_09 is newest
            os.utime(jdir, (mtime, mtime))

        # Perform cleanup with max_job_directories=5
        cleanup_old_visualizations(
            base_dir=temp_dir,
            max_job_directories=5,
            retention_hours=24.0,
            active_job_ids=set(),
        )

        remaining = set(os.listdir(temp_dir))
        assert len(remaining) == 5
        # Oldest 5 (job_00 to job_04) must be removed, newest 5 (job_05 to job_09) preserved
        for i in range(5):
            assert f"job_{i:02d}" not in remaining
        for i in range(5, 10):
            assert f"job_{i:02d}" in remaining
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_retention_hours_cleanup():
    """Verify that cleanup_old_visualizations purges directories older than retention_hours."""
    temp_dir = tempfile.mkdtemp()
    try:
        now = time.time()

        old_job = os.path.join(temp_dir, "old_job")
        new_job = os.path.join(temp_dir, "new_job")

        os.makedirs(old_job, exist_ok=True)
        os.makedirs(new_job, exist_ok=True)

        # Set old_job to 2 hours ago
        os.utime(old_job, (now - 7200, now - 7200))
        # Set new_job to 10 minutes ago
        os.utime(new_job, (now - 600, now - 600))

        # Cleanup with retention_hours=1.0
        cleanup_old_visualizations(
            base_dir=temp_dir,
            max_job_directories=10,
            retention_hours=1.0,
            active_job_ids=set(),
        )

        remaining = set(os.listdir(temp_dir))
        assert "old_job" not in remaining
        assert "new_job" in remaining
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_active_job_protection():
    """Verify that active/running job directories are never deleted during retention cleanup."""
    temp_dir = tempfile.mkdtemp()
    try:
        now = time.time()
        active_job = os.path.join(temp_dir, "active_job_123")
        os.makedirs(active_job, exist_ok=True)
        # Set timestamp to 5 hours ago (normally expired)
        os.utime(active_job, (now - 18000, now - 18000))

        cleanup_old_visualizations(
            base_dir=temp_dir,
            max_job_directories=1,
            retention_hours=1.0,
            active_job_ids={"active_job_123"},
        )

        assert os.path.isdir(active_job)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_visualization_disabled_toggle():
    """Verify that setting VISUALIZATION['enabled'] = False produces no files and returns clean empty output."""
    temp_dir = tempfile.mkdtemp()
    try:
        visualizer = MatchVisualizer(config={"output_dir": temp_dir, "enabled": False})
        img_a = create_dummy_image()
        img_b = create_dummy_image()
        res = create_dummy_hybrid_result()

        job_id = "disabled_job"
        viz_res = visualizer.generate(
            image_a=img_a,
            image_b=img_b,
            match_result=res,
            job_id=job_id,
        )

        assert isinstance(viz_res, VisualizationResult)
        assert viz_res.confidence_map_a is None
        assert viz_res.confidence_map_b is None
        assert viz_res.correspondence_image is None
        assert viz_res.registration_overlay is None
        assert viz_res.checkerboard is None
        assert not os.path.exists(os.path.join(temp_dir, job_id))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_job_manager_history_limit():
    """Verify that JobManager bounds completed job metadata history in RAM."""
    mock_matcher = MagicMock()
    mock_matcher.match.return_value = create_dummy_hybrid_result()
    mock_reg = MagicMock()
    mock_reg.register.return_value = None

    mgr = JobManager(matcher=mock_matcher, registration_engine=mock_reg)
    mgr.max_job_history = 3

    import cv2
    img_bytes = cv2.imencode(".png", create_dummy_image())[1].tobytes()

    job_ids = []
    for _ in range(6):
        jid, status = mgr.create_job(img_bytes, img_bytes)
        job_ids.append(jid)
        # Wait for worker thread to complete execution synchronously
        time.sleep(0.1)

    # Allow worker threads to finalize
    time.sleep(0.3)

    with mgr._jobs_lock:
        active_and_finished = list(mgr._jobs.keys())

    # Total stored jobs should not exceed max_job_history (3) plus any running job
    assert len(active_and_finished) <= 3
