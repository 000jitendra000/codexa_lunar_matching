import os
import shutil
import time
import gc
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Set
import cv2
import numpy as np

from src.matching.hybrid_matcher import HybridMatchResult
from src.matching.transformation import SimilarityTransform2D
from src.registration.registration_engine import RegistrationResult
from src.visualization.confidence_map import generate_confidence_map
from src.visualization.correspondence_plot import plot_correspondences
from src.visualization.registration_overlay import generate_registration_overlay

logger = logging.getLogger(__name__)


def cleanup_old_visualizations(
    base_dir: str,
    max_job_directories: int = 5,
    retention_hours: float = 1.0,
    active_job_ids: Optional[Set[str]] = None,
):
    """
    Best-effort cleanup of historical job visualization directories.

    Args:
        base_dir: Base directory containing job_id subdirectories.
        max_job_directories: Maximum allowed historical job directories.
        retention_hours: Age threshold in hours after which job directories are removed.
        active_job_ids: Set of currently active/running job_ids to never delete.
    """
    try:
        base_path = os.path.abspath(base_dir)
        if not os.path.isdir(base_path):
            return

        active_set = set(active_job_ids or [])
        now = time.time()
        retention_sec = float(retention_hours) * 3600.0

        # Collect valid job directories with modification times
        dir_entries = []
        for entry in os.listdir(base_path):
            dir_path = os.path.join(base_path, entry)
            if not os.path.isdir(dir_path):
                continue
            if entry in active_set or entry == ".git":
                continue
            try:
                mtime = os.path.getmtime(dir_path)
                dir_entries.append((entry, dir_path, mtime))
            except Exception:
                continue

        # Sort by modification time (newest first)
        dir_entries.sort(key=lambda x: x[2], reverse=True)

        # Evict directories based on retention_hours or max_job_directories
        for idx, (entry, dir_path, mtime) in enumerate(dir_entries):
            is_expired = (now - mtime) > retention_sec
            exceeds_count = idx >= max_job_directories

            if is_expired or exceeds_count:
                try:
                    shutil.rmtree(dir_path, ignore_errors=True)
                    logger.info("Cleaned up old visualization directory: %s", entry)
                except Exception as exc:
                    logger.warning("Failed to clean up directory %s: %s", dir_path, exc)
    except Exception as exc:
        logger.warning("Error during visualization cleanup: %s", exc)


@dataclass
class VisualizationResult:
    """
    Lightweight, primitive container for visualization output file paths.
    Contains no NumPy arrays, PyTorch tensors, or complex models for clean serialization.
    """

    confidence_map_a: Optional[str] = None
    confidence_map_b: Optional[str] = None
    correspondence_image: Optional[str] = None
    registration_overlay: Optional[str] = None
    checkerboard: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to JSON-compatible dictionary."""
        return {
            "confidence_map_a": self.confidence_map_a,
            "confidence_map_b": self.confidence_map_b,
            "correspondence_image": self.correspondence_image,
            "registration_overlay": self.registration_overlay,
            "checkerboard": self.checkerboard,
            "metadata": self.metadata,
        }


class MatchVisualizer:
    """
    High-level match visualizer generating explainable evidence maps, correspondence line plots,
    and registration overlays. Keeps model results valid even if an individual visualization fails.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config: Dict[str, Any] = {}
        try:
            from configs.default import Config
            default_cfg = getattr(Config, "VISUALIZATION", {})
            self.config.update(default_cfg)
        except Exception:
            pass
        if config is not None:
            self.config.update(config)

    def generate(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        match_result: HybridMatchResult,
        registration_result: Optional[RegistrationResult] = None,
        job_id: Optional[str] = None,
        output_dir: Optional[str] = None,
        active_job_ids: Optional[Set[str]] = None,
    ) -> VisualizationResult:
        """
        Generate and save visualization assets under output_dir/{job_id}/.

        Args:
            image_a: Reference lunar image array.
            image_b: Query lunar image array.
            match_result: HybridMatchResult produced by HybridMatcher.
            registration_result: Optional RegistrationResult produced by RegistrationEngine.
            job_id: Unique string identifier for directory organization.
            output_dir: Override directory path for saving PNGs.
            active_job_ids: Optional set of active job IDs to preserve during retention cleanup.

        Returns:
            VisualizationResult containing asset relative/absolute file paths.
        """
        # If visualization is disabled in config, return clean empty result
        if not self.config.get("enabled", True):
            return VisualizationResult(metadata={"disabled": True})

        # Resolve output directory
        base_dir = output_dir or self.config.get("output_dir", "data/processed/visualizations")
        sub_dir = job_id or "default"
        target_dir = os.path.abspath(os.path.join(base_dir, sub_dir))
        os.makedirs(target_dir, exist_ok=True)

        # Execute bounded retention cleanup
        if self.config.get("retention_enabled", True):
            effective_active = set(active_job_ids or [])
            if job_id:
                effective_active.add(job_id)
            cleanup_old_visualizations(
                base_dir=base_dir,
                max_job_directories=int(self.config.get("max_job_directories", 5)),
                retention_hours=float(self.config.get("retention_hours", 1.0)),
                active_job_ids=effective_active,
            )

        res = VisualizationResult()

        corrs = getattr(match_result, "correspondences", [])
        inliers = getattr(match_result, "inlier_indices", [])

        # Extract available transform
        transform = None
        if registration_result is not None and getattr(registration_result, "transform", None) is not None:
            transform = registration_result.transform
        elif getattr(match_result, "transform", None) is not None:
            transform = match_result.transform

        # 1. Generate Match Confidence Map for Image A
        try:
            cmap_a = generate_confidence_map(
                image=image_a,
                correspondences=corrs,
                inlier_indices=inliers,
                is_target_b=False,
                config=self.config,
            )
            path_a = os.path.join(target_dir, "confidence_map_a.png")
            if cv2.imwrite(path_a, cv2.cvtColor(cmap_a, cv2.COLOR_RGB2BGR)):
                res.confidence_map_a = path_a
            del cmap_a
        except Exception as exc:
            logger.warning("Failed to generate confidence_map_a: %s", exc, exc_info=True)

        # 2. Generate Match Confidence Map for Image B
        try:
            cmap_b = generate_confidence_map(
                image=image_b,
                correspondences=corrs,
                inlier_indices=inliers,
                is_target_b=True,
                config=self.config,
            )
            path_b = os.path.join(target_dir, "confidence_map_b.png")
            if cv2.imwrite(path_b, cv2.cvtColor(cmap_b, cv2.COLOR_RGB2BGR)):
                res.confidence_map_b = path_b
            del cmap_b
        except Exception as exc:
            logger.warning("Failed to generate confidence_map_b: %s", exc, exc_info=True)

        # 3. Generate Correspondence Line Plot
        try:
            corr_plot = plot_correspondences(
                image_a=image_a,
                image_b=image_b,
                correspondences=corrs,
                inlier_indices=inliers,
                config=self.config,
            )
            path_corr = os.path.join(target_dir, "correspondence_image.png")
            if cv2.imwrite(path_corr, cv2.cvtColor(corr_plot, cv2.COLOR_RGB2BGR)):
                res.correspondence_image = path_corr
            del corr_plot
        except Exception as exc:
            logger.warning("Failed to generate correspondence_image: %s", exc, exc_info=True)

        # 4. Generate Registration Overlay & Checkerboard (always generate, fallback to identity transform if None)
        try:
            eff_transform = transform if transform is not None else SimilarityTransform2D(
                scale=1.0, rotation_rad=0.0, translation_x=0.0, translation_y=0.0
            )
            overlay_rgb, checker_rgb = generate_registration_overlay(
                image_a=image_a,
                image_b=image_b,
                transform=eff_transform,
                config=self.config,
            )
            path_ov = os.path.join(target_dir, "registration_overlay.png")
            if cv2.imwrite(path_ov, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)):
                res.registration_overlay = path_ov
            del overlay_rgb

            if checker_rgb is not None:
                path_ck = os.path.join(target_dir, "checkerboard.png")
                if cv2.imwrite(path_ck, cv2.cvtColor(checker_rgb, cv2.COLOR_RGB2BGR)):
                    res.checkerboard = path_ck
                del checker_rgb
        except Exception as exc:
            logger.warning("Failed to generate registration_overlay: %s", exc, exc_info=True)

        gc.collect()

        res.metadata = {
            "num_correspondences": len(corrs),
            "num_inliers": len(inliers),
            "has_transform": (transform is not None),
        }

        return res
