"""
api/routes.py

FastAPI route definitions for lunar image matching API.
Exposes endpoints for health status, job submission, status polling, and live SSE progress streaming.
"""

import os
import asyncio
import logging
from typing import AsyncGenerator
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse

from api.schemas import (
    HealthResponse,
    JobCreatedResponse,
    JobStatusResponse,
)
from api.jobs import JobManager
from api.progress import progress_tracker, format_sse_event

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_VISUALIZATION_NAMES = {
    "confidence_map_a.png",
    "confidence_map_b.png",
    "correspondence_image.png",
    "registration_overlay.png",
    "checkerboard.png",
}
# Global JobManager reference set by main application
_job_manager: JobManager = None


def set_job_manager(mgr: JobManager):
    """Set the application-wide JobManager instance."""
    global _job_manager
    _job_manager = mgr


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    description="Returns API service health status and model availability.",
)
async def health_check():
    """Health check endpoint."""
    model_ready = (_job_manager is not None and _job_manager.matcher is not None)
    return HealthResponse(
        status="ok",
        service="lunar-image-matching-api",
        model_available=model_ready,
    )


@router.post(
    "/match",
    response_model=JobCreatedResponse,
    status_code=202,
    summary="Submit Image Pair for Location Matching",
    description="Asynchronously submit two lunar images (Image A and Image B) for feature matching and registration.",
)
async def submit_match(
    image_a: UploadFile = File(..., description="Reference lunar image (PNG, JPG, TIFF, etc.)"),
    image_b: UploadFile = File(..., description="Query lunar image to match (PNG, JPG, TIFF, etc.)"),
):
    """Submit image pair for matching."""
    if _job_manager is None:
        raise HTTPException(status_code=503, detail="Matching engine service unavailable.")

    if not image_a.filename or not image_b.filename:
        raise HTTPException(status_code=400, detail="Both image_a and image_b files must be provided.")

    try:
        bytes_a = await image_a.read()
        bytes_b = await image_b.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded image files: {str(exc)}")

    if not bytes_a or not bytes_b:
        raise HTTPException(status_code=400, detail="Uploaded image files must not be empty.")

    try:
        job_id, status = _job_manager.create_job(bytes_a, bytes_b)
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        logger.error("Failed to enqueue matching job: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to enqueue matching job.")

    return JobCreatedResponse(job_id=job_id, status=status)


@router.get(
    "/match/{job_id}",
    response_model=JobStatusResponse,
    summary="Poll Job Status and Registration Results",
    description="Retrieve the current status, progress, and registration results for a submitted job ID.",
)
async def get_match_status(job_id: str):
    """Poll job status."""
    if _job_manager is None:
        raise HTTPException(status_code=503, detail="Matching engine service unavailable.")

    status_resp = _job_manager.get_job_status(job_id)
    if status_resp is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return status_resp


@router.get(
    "/match/{job_id}/events",
    summary="Stream Live Progress Events via SSE",
    description="Establishes a Server-Sent Events (SSE) stream for live stage and tile progress updates.",
)
async def stream_match_events(job_id: str, request: Request):
    """SSE progress streaming endpoint."""
    if _job_manager is None:
        raise HTTPException(status_code=503, detail="Matching engine service unavailable.")

    status_resp = _job_manager.get_job_status(job_id)
    if status_resp is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    async def sse_generator() -> AsyncGenerator[str, None]:
        # Yield initial snapshot if available
        latest_evt = progress_tracker.get_latest_event(job_id)
        if latest_evt is not None:
            yield format_sse_event("progress", latest_evt.to_dict())

        # If job already completed or failed, yield final status and return
        st = _job_manager.get_job_status(job_id)
        if st and st.status in ("completed", "failed"):
            if st.status == "completed" and st.result:
                yield format_sse_event("completed", st.result.model_dump())
            elif st.status == "failed":
                yield format_sse_event("failed", {"job_id": job_id, "status": "failed", "error": st.error})
            return

        # Subscribe to live updates
        q = progress_tracker.subscribe(job_id)
        try:
            while True:
                # Check client disconnect
                if await request.is_disconnected():
                    break

                try:
                    payload = await asyncio.wait_for(q.get(), timeout=1.0)
                    yield payload

                    # Terminate stream on completed or failed wire event
                    if payload.startswith("event: completed") or payload.startswith("event: failed"):
                        break
                except asyncio.TimeoutError:
                    # Periodically check job status in case event was missed
                    current_st = _job_manager.get_job_status(job_id)
                    if current_st and current_st.status in ("completed", "failed"):
                        if current_st.status == "completed" and current_st.result:
                            yield format_sse_event("completed", current_st.result.model_dump())
                        elif current_st.status == "failed":
                            yield format_sse_event("failed", {"job_id": job_id, "status": "failed", "error": current_st.error})
                        break
        finally:
            progress_tracker.unsubscribe(job_id, q)

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/match/{job_id}/visualizations/{name}",
    summary="Retrieve Generated Visualization Asset",
    description="Safely serve generated PNG visualization assets for a given job ID.",
)
async def get_visualization(job_id: str, name: str):
    """Serve visualization PNG file."""
    if _job_manager is None:
        raise HTTPException(status_code=503, detail="Matching engine service unavailable.")

    if name not in ALLOWED_VISUALIZATION_NAMES:
        raise HTTPException(status_code=400, detail=f"Invalid visualization asset name '{name}'.")

    status_resp = _job_manager.get_job_status(job_id)
    if status_resp is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    base_dir = os.path.abspath(_job_manager.visualizer.config.get("output_dir", "data/processed/visualizations"))
    job_dir = os.path.abspath(os.path.join(base_dir, job_id))
    file_path = os.path.abspath(os.path.join(job_dir, name))

    # Security check: Prevent directory traversal
    if not file_path.startswith(job_dir):
        raise HTTPException(status_code=403, detail="Access denied to requested asset path.")

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"Visualization asset '{name}' not found for job '{job_id}'.")

    return FileResponse(file_path, media_type="image/png")
