"""
api/jobs.py

In-process background job manager with single-concurrency lock for memory safety.
Adapts model results into JSON-serializable MatchResultSummary structures.
"""

import asyncio
import logging
import threading
import time
import uuid
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, Tuple, Set

from src.matching.hybrid_matcher import HybridMatcher, HybridMatchResult
from src.registration.registration_engine import RegistrationEngine, RegistrationResult
from src.matching.progress import ProgressEvent
from src.visualization.match_visualizer import MatchVisualizer
from src.utils.memory_debug import log_memory_stage
from api.schemas import (
    MatchResultSummary,
    TransformSchema,
    TranslationSchema,
    MatchAcceptanceSchema,
    JobStatusResponse,
    ProgressEventSchema,
    VisualizationSchema,
)
from api.progress import progress_tracker

logger = logging.getLogger(__name__)


import gc

class JobManager:
    """
    In-process thread-safe manager for lunar image matching background jobs.
    Enforces single-job concurrency lock to prevent OOM during LoFTR inference.
    """

    def __init__(self, matcher: HybridMatcher, registration_engine: RegistrationEngine):
        self.matcher = matcher
        self.registration_engine = registration_engine
        self.visualizer = MatchVisualizer()
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._jobs_lock = threading.Lock()
        self._execution_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="lunar_job_worker")
        self.max_job_history: int = int(self.visualizer.config.get("max_job_history", 5))

    def _evict_old_completed_jobs(self):
        """Evict oldest finished jobs from in-memory _jobs dictionary if history limit is exceeded."""
        with self._jobs_lock:
            finished_jobs = [
                (jid, jdata)
                for jid, jdata in self._jobs.items()
                if jdata.get("status") in ("completed", "failed")
            ]
            if len(finished_jobs) > self.max_job_history:
                # Sort by completed_at (or created_at if completed_at is None)
                finished_jobs.sort(
                    key=lambda x: x[1].get("completed_at") or x[1].get("created_at") or 0
                )
                evict_count = len(finished_jobs) - self.max_job_history
                for i in range(evict_count):
                    evict_id = finished_jobs[i][0]
                    del self._jobs[evict_id]
                    logger.info("Evicted old job metadata from memory: %s", evict_id)

    def get_active_job_ids(self) -> Set[str]:
        """Return set of currently active/registered job IDs."""
        with self._jobs_lock:
            return set(self._jobs.keys())

    def create_job(self, bytes_a: bytes, bytes_b: bytes) -> Tuple[str, str]:
        """
        Validate image bytes, register job in state dictionary, and submit background worker.

        Returns:
            Tuple of (job_id, status).
        """
        # Validate image decoding
        img_a = cv2.imdecode(np.frombuffer(bytes_a, np.uint8), cv2.IMREAD_GRAYSCALE)
        img_b = cv2.imdecode(np.frombuffer(bytes_b, np.uint8), cv2.IMREAD_GRAYSCALE)

        if img_a is None or img_b is None or img_a.size == 0 or img_b.size == 0:
            raise ValueError("Invalid or corrupted image data. Could not decode grayscale image.")

        job_id = str(uuid.uuid4())
        created_at = time.time()

        with self._jobs_lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "created_at": created_at,
                "started_at": None,
                "completed_at": None,
                "result": None,
                "error": None,
                "image_a": img_a,
                "image_b": img_b,
            }

        # Submit background task
        self._executor.submit(self._run_job_wrapper, job_id)
        logger.info("Enqueued job %s (status=queued)", job_id)
        return job_id, "queued"

    def get_job_status(self, job_id: str) -> Optional[JobStatusResponse]:
        """
        Retrieve current job status response (or None if job_id does not exist).
        """
        with self._jobs_lock:
            job_data = self._jobs.get(job_id)
            if job_data is None:
                return None

            status = job_data["status"]
            result = job_data["result"]
            visualizations = job_data.get("visualizations")
            error = job_data["error"]

        latest_evt = progress_tracker.get_latest_event(job_id)
        progress_schema = (
            ProgressEventSchema(**latest_evt.to_dict()) if latest_evt is not None else None
        )

        return JobStatusResponse(
            job_id=job_id,
            status=status,
            progress=progress_schema,
            result=result,
            visualizations=visualizations,
            error=error,
        )

    def _run_job_wrapper(self, job_id: str):
        """Worker wrapper enforcing single-concurrency lock."""
        with self._jobs_lock:
            job_data = self._jobs.get(job_id)
            if not job_data:
                return

        # Enforce single heavy job concurrency
        logger.info("Job %s waiting for execution lock...", job_id)
        with self._execution_lock:
            logger.info("Job %s acquired execution lock.", job_id)
            with self._jobs_lock:
                self._jobs[job_id]["status"] = "running"
                self._jobs[job_id]["started_at"] = time.time()
                img_a = self._jobs[job_id]["image_a"]
                img_b = self._jobs[job_id]["image_b"]

            log_memory_stage("rss_after_image_loading")

            # Emit initial stage="loading" progress event
            init_event = ProgressEvent(
                stage="loading",
                current=0,
                total=1,
                progress=0.0,
                message="Loading input images",
            )
            progress_tracker.update_progress(job_id, init_event)

            try:

                def progress_cb(evt: ProgressEvent):
                    progress_tracker.update_progress(job_id, evt)

                # 1. Run Hybrid Matching
                hybrid_res: HybridMatchResult = self.matcher.match(
                    image_a=img_a,
                    image_b=img_b,
                    progress_callback=progress_cb,
                )

                # 2. Run Registration & Quality Evaluation
                reg_res: Optional[RegistrationResult] = None
                if hybrid_res.matched and hybrid_res.transform is not None:
                    try:
                        reg_res = self.registration_engine.register(
                            image_a=img_a,
                            image_b=img_b,
                            hybrid_result=hybrid_res,
                            progress_callback=progress_cb,
                        )
                    except Exception as reg_err:
                        logger.warning("Registration engine failed for job %s: %s", job_id, reg_err)
                log_memory_stage("rss_after_registration")

                # Convert to MatchResultSummary
                summary = self._build_result_summary(hybrid_res, reg_res)

                # 3. Generate Visualization Assets (isolated error handling)
                viz_schema: Optional[VisualizationSchema] = None
                try:
                    active_ids = self.get_active_job_ids()
                    viz_res = self.visualizer.generate(
                        image_a=img_a,
                        image_b=img_b,
                        match_result=hybrid_res,
                        registration_result=reg_res,
                        job_id=job_id,
                        active_job_ids=active_ids,
                    )
                    viz_schema = VisualizationSchema(
                        confidence_map_a=f"/match/{job_id}/visualizations/confidence_map_a.png" if viz_res.confidence_map_a else None,
                        confidence_map_b=f"/match/{job_id}/visualizations/confidence_map_b.png" if viz_res.confidence_map_b else None,
                        correspondence_image=f"/match/{job_id}/visualizations/correspondence_image.png" if viz_res.correspondence_image else None,
                        registration_overlay=f"/match/{job_id}/visualizations/registration_overlay.png" if viz_res.registration_overlay else None,
                        checkerboard=f"/match/{job_id}/visualizations/checkerboard.png" if viz_res.checkerboard else None,
                    )
                except Exception as viz_err:
                    logger.warning("MatchVisualizer failed for job %s: %s", job_id, viz_err, exc_info=True)
                log_memory_stage("rss_after_visualization")

                # Emit final completed event
                completed_evt = ProgressEvent(
                    stage="completed",
                    current=1,
                    total=1,
                    progress=1.0,
                    message="Matching and registration complete",
                )
                progress_tracker.update_progress(job_id, completed_evt)

                with self._jobs_lock:
                    self._jobs[job_id]["status"] = "completed"
                    self._jobs[job_id]["completed_at"] = time.time()
                    self._jobs[job_id]["result"] = summary
                    self._jobs[job_id]["visualizations"] = viz_schema

                progress_tracker.notify_job_finished(job_id, "completed", summary.model_dump())
                logger.info("Job %s completed successfully (matched=%s).", job_id, summary.matched)

            except Exception as exc:
                logger.error("Job %s failed with unexpected exception: %s", job_id, exc, exc_info=True)

                failed_evt = ProgressEvent(
                    stage="failed",
                    current=0,
                    total=1,
                    progress=0.0,
                    message=f"Job failed: {str(exc)}",
                )
                progress_tracker.update_progress(job_id, failed_evt)

                safe_err_msg = "Internal matching failure occurred during processing."
                with self._jobs_lock:
                    self._jobs[job_id]["status"] = "failed"
                    self._jobs[job_id]["completed_at"] = time.time()
                    self._jobs[job_id]["error"] = safe_err_msg

                progress_tracker.notify_job_finished(
                    job_id, "failed", {"job_id": job_id, "status": "failed", "error": safe_err_msg}
                )
            finally:
                # Guaranteed cleanup of raw input images in RAM on both success & failure
                with self._jobs_lock:
                    if job_id in self._jobs:
                        self._jobs[job_id]["image_a"] = None
                        self._jobs[job_id]["image_b"] = None
                img_a = None
                img_b = None
                self._evict_old_completed_jobs()
                gc.collect()
                log_memory_stage("rss_after_complete_job_cleanup")

    def _build_result_summary(
        self,
        hybrid_res: HybridMatchResult,
        reg_res: Optional[RegistrationResult],
    ) -> MatchResultSummary:
        """Construct JSON-serializable MatchResultSummary from model outputs."""
        matched = bool(hybrid_res.matched)
        corrs = int(hybrid_res.num_correspondences)
        inliers = int(len(hybrid_res.inlier_indices))
        ratio = round(float(hybrid_res.inlier_ratio), 4)

        # Build acceptance schema if available
        acc_schema: Optional[MatchAcceptanceSchema] = None
        if hybrid_res.acceptance is not None:
            acc_dict = hybrid_res.acceptance.to_dict()
            try:
                acc_schema = MatchAcceptanceSchema(
                    accepted=acc_dict["accepted"],
                    status=acc_dict["status"],
                    reason=acc_dict["reason"],
                    acceptance_score=acc_dict["acceptance_score"],
                    checks=acc_dict.get("checks", {}),
                    metrics=acc_dict.get("metrics", {}),
                )
            except Exception as acc_err:
                logger.warning("Failed to format acceptance schema: %s", acc_err)

        # Transform extraction
        tf_obj = reg_res.transform if (reg_res is not None and reg_res.transform is not None) else hybrid_res.transform
        tf_schema: Optional[TransformSchema] = None
        if tf_obj is not None:
            try:
                tf_schema = TransformSchema(
                    scale=round(float(tf_obj.scale), 6),
                    rotation_deg=round(float(tf_obj.rotation_deg), 4),
                    translation=TranslationSchema(
                        x=round(float(tf_obj.translation[0]), 4),
                        y=round(float(tf_obj.translation[1]), 4),
                    ),
                )
            except Exception as tf_err:
                logger.warning("Failed to format transform schema: %s", tf_err)

        if reg_res is not None and reg_res.success and matched:
            return MatchResultSummary(
                matched=True,
                correspondences=corrs,
                inliers=inliers,
                inlier_ratio=ratio,
                confidence=round(float(reg_res.confidence), 4),
                rmse=round(float(reg_res.rmse), 4) if reg_res.rmse is not None else None,
                mean_error=round(float(reg_res.mean_error), 4) if reg_res.mean_error is not None else None,
                median_error=round(float(reg_res.median_error), 4) if reg_res.median_error is not None else None,
                max_error=round(float(reg_res.max_error), 4) if reg_res.max_error is not None else None,
                coverage=round(float(reg_res.coverage), 4),
                quality=str(reg_res.quality),
                transform=tf_schema,
                acceptance=acc_schema,
                failure_reason=None,
            )

        if matched:
            return MatchResultSummary(
                matched=True,
                correspondences=corrs,
                inliers=inliers,
                inlier_ratio=ratio,
                confidence=round(float(ratio), 4),
                coverage=0.0,
                quality="GOOD" if ratio > 0.3 else "FAIR",
                transform=tf_schema,
                acceptance=acc_schema,
                failure_reason=None,
            )

        # Unmatched / Geometrically failed / Acceptance rejected case
        reason = (
            hybrid_res.acceptance.reason
            if (hybrid_res.acceptance is not None and hybrid_res.acceptance.reason)
            else (hybrid_res.metadata.get("reason") or "Insufficient geometric inliers during RANSAC verification.")
        )
        return MatchResultSummary(
            matched=False,
            correspondences=corrs,
            inliers=inliers,
            inlier_ratio=ratio,
            confidence=0.0,
            coverage=0.0,
            quality="FAILED",
            transform=None,
            acceptance=acc_schema,
            failure_reason=reason,
        )
