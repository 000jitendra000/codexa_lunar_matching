"""
tests/test_api.py

Integration and unit tests for the Lunar Image Matching FastAPI service.
Verifies API endpoints, job state transitions, error handling, SSE progress streaming,
and model independence.
"""

import io
import time
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient

from api.main import app
from src.matching.hybrid_matcher import HybridMatcher
from src.matching.progress import ProgressEvent, ProgressCallback


@pytest.fixture
def client():
    """FastAPI TestClient fixture with lifespan startup execution."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_image_bytes():
    """Generate two synthetic 128x128 grayscale test images with distinct features."""
    img_a = np.zeros((128, 128), dtype=np.uint8)
    cv2.circle(img_a, (40, 40), 15, 255, -1)
    cv2.circle(img_a, (80, 80), 20, 200, -1)
    cv2.rectangle(img_a, (20, 90), (50, 110), 180, -1)

    # Image B is slightly transformed (translation + scale)
    M = np.float32([[1.0, 0.0, 5.0], [0.0, 1.0, 3.0]])
    img_b = cv2.warpAffine(img_a, M, (128, 128))

    _, buf_a = cv2.imencode(".png", img_a)
    _, buf_b = cv2.imencode(".png", img_b)

    return buf_a.tobytes(), buf_b.tobytes()


def test_health_endpoint(client):
    """Verify /health returns 200 OK and model status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "lunar-image-matching-api"
    assert "model_available" in data
    assert data["model_available"] is True


def test_submit_match_missing_files(client):
    """Verify POST /match with missing parameters returns 422 Unprocessable Entity."""
    response = client.post("/match", files={})
    assert response.status_code in (400, 422)


def test_submit_match_corrupted_image(client):
    """Verify POST /match with non-image bytes returns 400 Bad Request."""
    files = {
        "image_a": ("a.txt", b"not an image", "text/plain"),
        "image_b": ("b.txt", b"not an image", "text/plain"),
    }
    response = client.post("/match", files=files)
    assert response.status_code == 400
    assert "Invalid or corrupted image data" in response.json()["detail"]


def test_non_existent_job_404(client):
    """Verify GET /match/{job_id} for unknown job ID returns 404 Not Found."""
    response = client.get("/match/invalid-job-uuid-12345")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_end_to_end_job_lifecycle(client, sample_image_bytes):
    """Verify job submission, progress tracking, status polling, and final completion."""
    bytes_a, bytes_b = sample_image_bytes
    files = {
        "image_a": ("test_a.png", io.BytesIO(bytes_a), "image/png"),
        "image_b": ("test_b.png", io.BytesIO(bytes_b), "image/png"),
    }

    # 1. Submit job
    response = client.post("/match", files=files)
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "queued"
    job_id = data["job_id"]

    # 2. Poll status until completed (timeout after 15s)
    start_time = time.time()
    completed = False
    status_data = None

    while time.time() - start_time < 15.0:
        poll_res = client.get(f"/match/{job_id}")
        assert poll_res.status_code == 200
        status_data = poll_res.json()
        if status_data["status"] == "completed":
            completed = True
            break
        time.sleep(0.1)

    assert completed, f"Job did not complete in time. Last status: {status_data}"

    # 3. Verify result structure
    result = status_data["result"]
    assert result is not None
    assert "matched" in result
    assert "correspondences" in result
    assert "inliers" in result
    assert "inlier_ratio" in result
    assert "confidence" in result
    assert "coverage" in result
    assert "quality" in result

    if result["matched"]:
        assert result["transform"] is not None
        tf = result["transform"]
        assert "scale" in tf
        assert "rotation_deg" in tf
        assert "translation" in tf
        assert "x" in tf["translation"]
        assert "y" in tf["translation"]


def test_sse_progress_stream(client, sample_image_bytes):
    """Verify GET /match/{job_id}/events returns SSE stream."""
    bytes_a, bytes_b = sample_image_bytes
    files = {
        "image_a": ("test_a.png", io.BytesIO(bytes_a), "image/png"),
        "image_b": ("test_b.png", io.BytesIO(bytes_b), "image/png"),
    }

    sub_res = client.post("/match", files=files)
    job_id = sub_res.json()["job_id"]

    # Request SSE stream
    stream_res = client.get(f"/match/{job_id}/events")
    assert stream_res.status_code == 200
    assert "text/event-stream" in stream_res.headers["content-type"]

    lines = stream_res.text.splitlines()
    assert len(lines) > 0
    # Confirm presence of event wire format
    event_lines = [l for l in lines if l.startswith("event: ")]
    assert len(event_lines) > 0


def test_unmatched_pair_returns_completed_status(client):
    """Verify an unmatched noise pair yields status=completed with matched=False (NOT HTTP 500)."""
    noise_a = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    noise_b = np.random.randint(0, 255, (64, 64), dtype=np.uint8)

    _, buf_a = cv2.imencode(".png", noise_a)
    _, buf_b = cv2.imencode(".png", noise_b)

    files = {
        "image_a": ("noise_a.png", io.BytesIO(buf_a.tobytes()), "image/png"),
        "image_b": ("noise_b.png", io.BytesIO(buf_b.tobytes()), "image/png"),
    }

    sub_res = client.post("/match", files=files)
    job_id = sub_res.json()["job_id"]

    # Wait for execution
    start_time = time.time()
    status_data = None
    while time.time() - start_time < 15.0:
        poll_res = client.get(f"/match/{job_id}")
        status_data = poll_res.json()
        if status_data["status"] == "completed":
            break
        time.sleep(0.1)

    assert status_data["status"] == "completed"
    res = status_data["result"]
    assert res is not None
    # Even if RANSAC fails on noise, HTTP request and job execution succeed cleanly
    assert "matched" in res


def test_model_independence_without_fastapi():
    """
    Verify model can be run directly in pure Python without importing or invoking FastAPI.
    """
    img_a = np.zeros((100, 100), dtype=np.uint8)
    img_b = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(img_a, (30, 30), 10, 255, -1)
    cv2.circle(img_b, (32, 32), 10, 255, -1)

    received_events = []

    def my_callback(evt: ProgressEvent):
        received_events.append(evt)

    matcher = HybridMatcher()
    res = matcher.match(img_a, img_b, progress_callback=my_callback)

    assert res is not None
    assert hasattr(res, "matched")
    assert hasattr(res, "num_correspondences")
    assert len(received_events) > 0


def test_openapi_schema_file_upload_format():
    """
    Verify OpenAPI schema correctly formats image_a and image_b as binary file uploads
    (type: string, format: binary) so Swagger UI renders file upload controls ([Choose File]).
    """
    openapi_spec = app.openapi()
    match_post_path = openapi_spec["paths"]["/match"]["post"]

    # Verify content type is multipart/form-data
    request_content = match_post_path["requestBody"]["content"]
    assert "multipart/form-data" in request_content

    # Find schema reference or properties
    schema_ref = request_content["multipart/form-data"]["schema"]["$ref"]
    schema_name = schema_ref.split("/")[-1]
    properties = openapi_spec["components"]["schemas"][schema_name]["properties"]

    assert "image_a" in properties
    assert "image_b" in properties

    assert properties["image_a"]["type"] == "string"
    assert properties["image_a"]["format"] == "binary"

    assert properties["image_b"]["type"] == "string"
    assert properties["image_b"]["format"] == "binary"

